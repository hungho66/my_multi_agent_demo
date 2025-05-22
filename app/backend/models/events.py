from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, Literal
from datetime import datetime, timezone # Thêm import này

class BaseEvent(BaseModel):
    """Base class for all Server-Sent Event (SSE) structures."""
    type: str # The event type (e.g., "start", "agent_update")

    def to_sse(self) -> str:
        """Converts the event data to SSE message format."""
        event_type_lower = self.type.lower()
        # Ensure model_dump_json is used for Pydantic v2
        json_data = self.model_dump_json(exclude_none=True)
        return f"event: {event_type_lower}\ndata: {json_data}\n\n"

class StartEvent(BaseEvent):
    """Event sent when processing begins."""
    type: Literal["start"] = "start"
    query: str
    message: str = "Processing started..."
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class AgentUpdateEvent(BaseEvent):
    """Event sent when an agent provides a progress update."""
    type: Literal["agent_update"] = "agent_update"
    agent_name: str
    status: str # Current status message from the agent
    details: Optional[Any] = None # More detailed data, e.g., a plan object or tool output summary
    task_id: Optional[str] = None # Relevant task ID, if applicable
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ErrorEvent(BaseEvent):
    """Event sent when an error occurs during processing."""
    type: Literal["error"] = "error"
    source_agent: Optional[str] = None # The agent that reported or caused the error
    error_message: str
    original_query: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class CompleteEvent(BaseEvent):
    """Event sent when the entire agent graph processing is complete."""
    type: Literal["complete"] = "complete"
    original_query: str
    final_summary: Optional[Dict[str, Any]] # The final result, typically a dict from FinalSummary model
    has_errors: bool = False # Indicates if any errors occurred during the overall process
    error_details: Optional[str] = None # Summary of errors if has_errors is true
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())