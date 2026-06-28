"""
Q&A Assistant endpoints
"""
import hashlib
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel

from src.database.models import User, UserDocument
from src.api.deps import get_db_session, get_current_user, get_embedding_manager
from src.models.embeddings import EmbeddingManager
from src.utils.config import settings

router = APIRouter(prefix="/qa", tags=["Q&A Assistant"])


class QuestionRequest(BaseModel):
    """Schema for asking a question"""
    question: str
    n_context: int = 5
    filter_type: Optional[str] = None  # "paper", "article", "user_doc", or None for all


class QuestionResponse(BaseModel):
    """Schema for Q&A response"""
    answer: str
    citations: List[dict]
    sources_used: int


class DocumentUploadResponse(BaseModel):
    """Schema for document upload response"""
    doc_id: int
    title: str
    chunks: int
    message: str


class DocumentListItem(BaseModel):
    """Schema for document list item"""
    id: int
    title: str
    source: Optional[str]
    chunk_count: int
    created_at: str


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """
    Ask a question and get an answer from the knowledge base

    - **question**: The question to ask
    - **n_context**: Number of context documents to retrieve (3-10)
    - **filter_type**: Optional filter by content type
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )

    try:
        # Import QA components
        from src.rag.retriever import Retriever
        from src.rag.generator import Generator

        retriever = Retriever(embedding_manager)
        generator = Generator()

        # Retrieve relevant documents
        results = retriever.retrieve(
            query=request.question,
            n_results=request.n_context,
            filter_type=request.filter_type
        )

        if not results:
            return QuestionResponse(
                answer="I couldn't find any relevant information in the knowledge base to answer your question.",
                citations=[],
                sources_used=0
            )

        # Generate answer
        answer_result = generator.generate_answer(
            question=request.question,
            context=results
        )

        return QuestionResponse(
            answer=answer_result.get("answer", ""),
            citations=answer_result.get("citations", []),
            sources_used=len(results)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}"
        )


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """
    Upload a document to the user's knowledge base

    Supported formats: .txt, .md, .pdf
    """
    # Validate file type
    allowed_extensions = {".txt", ".md", ".pdf"}
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        # Read file content
        content = await file.read()

        # Extract text based on file type
        if file_ext == ".pdf":
            from src.utils.preprocessing import extract_text_from_pdf
            text_content = extract_text_from_pdf(content)
        else:
            text_content = content.decode("utf-8", errors="ignore")

        if not text_content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document appears to be empty"
            )

        # Calculate content hash for deduplication
        content_hash = hashlib.sha256(text_content.encode()).hexdigest()

        # Check for duplicate
        existing = db.query(UserDocument).filter(
            UserDocument.user_id == current_user.id,
            UserDocument.content_hash == content_hash
        ).first()

        if existing:
            return DocumentUploadResponse(
                doc_id=existing.id,
                title=existing.title,
                chunks=existing.chunk_count,
                message="Document already exists in your knowledge base"
            )

        # Use provided title or filename
        doc_title = title or file.filename

        # Add to vector database
        doc_id = f"userdoc_{current_user.id}_{content_hash[:8]}"
        chunk_count = embedding_manager.add_user_document(
            doc_id=doc_id,
            title=doc_title,
            content=text_content,
            metadata={
                "user_id": current_user.id,
                "source": file.filename
            }
        )

        # Save to database
        user_doc = UserDocument(
            user_id=current_user.id,
            title=doc_title,
            source=file.filename,
            content_hash=content_hash,
            chunk_count=chunk_count
        )
        db.add(user_doc)
        db.commit()
        db.refresh(user_doc)

        return DocumentUploadResponse(
            doc_id=user_doc.id,
            title=doc_title,
            chunks=chunk_count,
            message=f"Successfully added {chunk_count} chunks to your knowledge base"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get("/documents", response_model=List[DocumentListItem])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    List all documents uploaded by the user
    """
    docs = db.query(UserDocument).filter(
        UserDocument.user_id == current_user.id
    ).order_by(UserDocument.created_at.desc()).all()

    return [
        DocumentListItem(
            id=doc.id,
            title=doc.title,
            source=doc.source,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at.isoformat()
        )
        for doc in docs
    ]


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Delete a user-uploaded document

    Note: This removes the document from the database but may not
    immediately remove it from the vector database.
    """
    doc = db.query(UserDocument).filter(
        UserDocument.id == doc_id,
        UserDocument.user_id == current_user.id
    ).first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    db.delete(doc)
    db.commit()
    return None
