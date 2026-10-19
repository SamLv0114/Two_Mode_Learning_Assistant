"""Test doubles shared by the recommendation tests."""
import json
import threading
from types import SimpleNamespace

import numpy as np


class TaskClient:
    """
    An OpenAI client stand-in that answers each structured call by its task name
    (the response_format schema name), so parallel calls get the right reply in any order.
    A handler returns a dict (sent as JSON) or raises to simulate a provider error.
    """

    def __init__(self, **handlers):
        self.handlers = handlers
        self.requests = []
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        task = kwargs["response_format"]["json_schema"]["name"]
        with self._lock:
            self.requests.append((task, kwargs))
        reply = self.handlers[task](kwargs)
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                                                        message=SimpleNamespace(content=content, refusal=None))])

    def tasks(self):
        return [task for task, _ in self.requests]


def user_text(kwargs) -> str:
    return "\n".join(m["content"] for m in kwargs["messages"] if m["role"] == "user")


class WordEmbeddings:
    """
    Embeddings over a fixed vocabulary: a text's vector counts the topic words it contains,
    so similarity is predictable. Papers and the vector store live in memory.
    """

    TOPICS = ("diffusion", "language", "graph", "robot", "vision", "speech")

    def __init__(self):
        self.store = {}            # doc id -> (vector, metadata)
        self.collection = self

    def generate_embedding(self, text):
        lowered = text.lower()
        vec = np.array([lowered.count(t) for t in self.TOPICS], dtype=float) + 0.01
        return (vec / np.linalg.norm(vec)).tolist()

    def generate_embeddings(self, texts):
        return [self.generate_embedding(t) for t in texts]

    def embed_papers(self, titles, abstracts):
        return [self.generate_embedding(f"{t} {a}") for t, a in zip(titles, abstracts)]

    def index(self, arxiv_id, title, abstract):
        self.store[f"paper_{arxiv_id}"] = (self.generate_embedding(f"{title} {abstract}"), {"paper_id": arxiv_id})

    # The parts of the Chroma collection API the code under test uses.
    def get(self, ids, include=None):
        found = [i for i in ids if i in self.store]
        return {"ids": found, "embeddings": [self.store[i][0] for i in found]}

    def upsert(self, ids, embeddings, documents, metadatas):
        for doc_id, emb, meta in zip(ids, embeddings, metadatas):
            self.store[doc_id] = (emb, meta)

    def search(self, query, n_results=10, filter_type=None, user_id=None):
        q = np.asarray(self.generate_embedding(query))
        scored = sorted(((1 - float(np.dot(q, v)), meta) for v, meta in self.store.values()), key=lambda x: x[0])
        return [{"distance": d, "metadata": meta} for d, meta in scored[:n_results]]
