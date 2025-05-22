from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class AgentQueryRequest(BaseModel):
    """Schema for API request to process a query."""
    query: str = Field(..., description="The user's query for the agent system.")

class ErrorResponse(BaseModel):
    """Schema for a generic error response."""
    detail: str
    error_code: Optional[str] = None

# --- Schemas for data within Server-Sent Events (SSE) ---
class SSEBaseData(BaseModel):
    pass

class SSEStartData(SSEBaseData):
    """Data for the 'start' SSE event."""
    query: str
    message: str = "Processing started..."

class SSEAgentUpdateData(SSEBaseData):
    """Data for an 'agent_update' SSE event, detailing an agent's progress."""
    agent_name: str
    status: str
    details: Optional[Any] = None # Can be a Pydantic model's dict representation or text
    task_id: Optional[str] = None

class SSEErrorData(SSEBaseData):
    """Data for an 'error' SSE event (system or agent error)."""
    source_agent: Optional[str] = None
    error_message: str
    original_query: Optional[str] = None

class SSECompleteData(SSEBaseData):
    """Data for the 'complete' SSE event, when the entire graph finishes."""
    original_query: str
    final_summary: Optional[Dict[str, Any]] # Dict representation of the FinalSummary Pydantic model
    has_errors: bool = False
    error_details: Optional[str] = None