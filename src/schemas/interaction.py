"""
Interaction-related Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class InteractionCreate(BaseModel):
    """Schema for creating a user interaction"""
    item_type: Literal["paper", "article"]
    item_id: int = Field(..., gt=0)
    interaction_type: Literal["viewed", "saved", "dismissed"]


class InteractionResponse(BaseModel):
    """Schema for interaction response"""
    id: int
    item_type: str
    item_id: int
    interaction_type: str
    timestamp: datetime

    class Config:
        from_attributes = True


class InteractionStats(BaseModel):
    """Schema for user interaction statistics"""
    total: int = 0
    saved: int = 0
    viewed: int = 0
    dismissed: int = 0
    ready_for_training: bool = False
    interactions_until_training: int = 0


class InteractionList(BaseModel):
    """Schema for paginated list of interactions"""
    items: list[InteractionResponse]
    total: int
    page: int
    per_page: int
    has_more: bool


class ModelStatus(BaseModel):
    """Schema for user's model training status"""
    is_trained: bool = False
    last_trained_at: Optional[datetime] = None
    interaction_count: int = 0
    min_interactions_required: int = 50
    train_ndcg: Optional[float] = None
    train_mrr: Optional[float] = None
    val_ndcg: Optional[float] = None
    val_mrr: Optional[float] = None


class RetrainResponse(BaseModel):
    """Schema for model retraining response"""
    success: bool
    message: str
    metrics: Optional[dict] = None
