"""
LLM answer generation for RAG
"""
from typing import List, Dict
from src.utils.config import settings
import openai
from anthropic import Anthropic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator:
    """Generates answers using LLM with retrieved context"""
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        
        if self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment")
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate_answer(self, question: str, context: List[Dict], 
                        user_interests: List[str] = None) -> Dict:
        """
        Generate answer with citations from context
        """
        if user_interests is None:
            user_interests = settings.USER_INTERESTS
        
        # Format context
        context_text = "\n\n".join([
            f"[Source {i+1}]: {doc['document']}\n"
            f"Metadata: {doc.get('metadata', {})}"
            for i, doc in enumerate(context)
        ])
        
        interests_str = ", ".join(user_interests)
        
        prompt = f"""You are a helpful AI teaching assistant. Answer the following question using the provided context.

User interests: {interests_str}

Question: {question}

Context:
{context_text}

Instructions:
1. Provide a clear, comprehensive answer
2. Cite specific sources using [Source N] format
3. If the context doesn't fully answer the question, say so
4. Explain concepts in a way that's accessible but technically accurate
5. Relate the answer to the user's interests when relevant

Answer:"""
        
        answer = self._generate(prompt)
        
        # Extract citations
        citations = []
        for doc in context:
            metadata = doc.get("metadata", {})
            if metadata.get("type") == "paper":
                citations.append({
                    "type": "paper",
                    "title": metadata.get("title", "Unknown"),
                    "arxiv_id": metadata.get("paper_id", ""),
                    "url": metadata.get("url", "")
                })
            elif metadata.get("type") == "article":
                citations.append({
                    "type": "article",
                    "title": metadata.get("title", "Unknown"),
                    "url": metadata.get("url", ""),
                    "source": metadata.get("source", "")
                })
            elif metadata.get("type") == "user_doc":
                citations.append({
                    "type": "user_doc",
                    "title": metadata.get("title", "User document"),
                    "source": metadata.get("source", "")
                })
        
        return {
            "answer": answer,
            "citations": citations
        }
    
    def _generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text using the configured LLM"""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant specialized in machine learning and research."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=max_tokens,
                    # temperature=0.7
                )
                return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_summary(self, title: str, content: str, user_interests: List[str] = None) -> str:
        """
        Generate personalized summary for a paper/article
        """
        if user_interests is None:
            user_interests = settings.USER_INTERESTS
        
        interests_str = ", ".join(user_interests)
        
        prompt = f"""Summarize this for an ML grad student focused on {interests_str}.

Title: {title}
Content: {content[:2000]}

Provide:
- One-sentence key insight
- Why they should care (2-3 sentences)
- How it relates to their interests

Keep under 100 words. Be concise and actionable."""
        
        return self._generate(prompt, max_tokens=200)

