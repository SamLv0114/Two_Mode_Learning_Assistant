"""
DocumentAgent: helps the user manage and search their uploaded documents.
"""
from src.agents.base_agent import BaseAgent
from src.agents.tools import LIST_USER_DOCUMENTS, SEARCH_USER_DOCUMENTS


class DocumentAgent(BaseAgent):
    name = "DocumentAgent"

    system_prompt = """\
You are a document management assistant for a researcher's personal knowledge base.

Your role: Help the user understand, navigate, and search their uploaded documents.

Guidelines:
- Call list_user_documents to show what files the user has
- Call search_user_documents when the user wants to find specific information
- Quote relevant passages directly from search results
- Tell the user how many documents they have and what topics they cover
- If a search returns nothing, tell the user clearly and suggest uploading relevant material
- Keep responses concise and organized"""

    tool_schemas = [LIST_USER_DOCUMENTS, SEARCH_USER_DOCUMENTS]
