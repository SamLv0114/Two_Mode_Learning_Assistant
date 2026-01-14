"""
MODE 2: Q&A Assistant with RAG
"""
from typing import Dict, Optional
import logging
from uuid import uuid4

from src.rag import Retriever, Generator
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAAssistant:
    """Q&A Assistant using RAG (Retrieval-Augmented Generation)"""
    
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
    
    def answer_question(self, question: str, n_context: int = 5, filter_type: Optional[str] = None) -> Dict:
        """
        Answer a question using RAG
        
        Args:
            question: User's question
            n_context: Number of relevant documents to retrieve
            filter_type: Optional filter by "paper", "article", or "user_doc"
            
        Returns:
            Dict with 'answer' and 'citations'
        """
        logger.info(f"Answering question: {question}")
        
        # Step 1: Retrieve relevant context
        logger.info("Step 1: Retrieving relevant context...")
        context = self.retriever.retrieve(
            question,
            n_results=n_context,
            filter_type=filter_type
        )
        
        if not context:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base to answer this question. "
                         "Try rephrasing your question or adding more content to the knowledge base.",
                "citations": []
            }
        
        # Step 2: Generate answer with citations
        logger.info("Step 2: Generating answer...")
        result = self.generator.generate_answer(
            question=question,
            context=context
        )
        
        logger.info("Answer generated successfully")
        return result

    def add_user_document(self, title: str, content: str, source: str = "") -> Dict:
        """
        Add a user document to the vector database.
        """
        doc_id = uuid4().hex
        chunk_count = self.retriever.embedding_manager.add_user_document(
            doc_id=doc_id,
            title=title,
            content=content,
            metadata={"source": source}
        )
        return {"doc_id": doc_id, "chunks": chunk_count}

    
    def format_answer(self, result: Dict) -> str:
        """Format answer for display"""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("ANSWER:")
        lines.append("=" * 60)
        lines.append(f"\n{result['answer']}\n")
        
        if result['citations']:
            lines.append("\n" + "=" * 60)
            lines.append("SOURCES:")
            lines.append("=" * 60)
            for i, citation in enumerate(result['citations'], 1):
                if citation['type'] == 'paper':
                    lines.append(f"\n{i}. Paper: {citation['title']}")
                    lines.append(f"   arXiv: {citation.get('arxiv_id', 'N/A')}")
                    lines.append(f"   URL: {citation.get('url', 'N/A')}")
                elif citation['type'] == 'article':
                    lines.append(f"\n{i}. Article: {citation['title']}")
                    lines.append(f"   Source: {citation.get('source', 'N/A')}")
                    lines.append(f"   URL: {citation.get('url', 'N/A')}")
                else:
                    lines.append(f"\n{i}. User Document: {citation['title']}")
                    lines.append(f"   Source: {citation.get('source', 'N/A')}")
        
        return "\n".join(lines)
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "=" * 60)
        print("Q&A Assistant - Interactive Mode")
        print("Type 'quit' or 'exit' to end the session")
        print("=" * 60 + "\n")
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            result = self.answer_question(question)
            print(self.format_answer(result))


if __name__ == "__main__":
    assistant = QAAssistant()
    assistant.interactive_mode()

