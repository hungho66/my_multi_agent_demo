from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from ..models.schemas import AgentQueryRequest, ErrorResponse
from ..services.graph_service import GraphService, get_graph_service
from ..models.events import StartEvent, AgentUpdateEvent, ErrorEvent, CompleteEvent
from src.utils.progress import progress_tracker # Ensure this path is correct
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone

router = APIRouter()

sse_progress_queue = asyncio.Queue()

def sse_progress_handler(agent_name: str, optional_data: Optional[Dict[str, Any]], status_message: str, timestamp: str):
    """
    Handler called by progress_tracker on updates.
    Puts an AgentUpdateEvent into the SSE queue.
    """
    details_for_event = {}
    task_id_for_event = None

    if optional_data:
        task_id_for_event = optional_data.get("task_id")
        # You can add other fields from optional_data to details_for_event if needed
        if "current_step" in optional_data: # Example from your progress_tracker
            details_for_event["current_step_info"] = optional_data["current_step"]
        # Add other relevant details if present in optional_data
        for key, value in optional_data.items():
            if key not in ["task_id", "current_step"]: # Avoid duplication
                details_for_event[key] = value


    event = AgentUpdateEvent(
        agent_name=agent_name,
        status=status_message,
        details=details_for_event if details_for_event else None,
        task_id=task_id_for_event,
        timestamp=timestamp # Use the timestamp from progress_tracker
    )
    try:
        sse_progress_queue.put_nowait(event)
    except asyncio.QueueFull:
        print(f"WARNING: SSE progress queue is full. Skipping update from {agent_name}.")


@router.post(
    "/run_main_flow",
    responses={
        200: {"description": "Agent flow started, results will be streamed via SSE."},
        400: {"model": ErrorResponse, "description": "Invalid parameters."},
        500: {"model": ErrorResponse, "description": "Internal server error."}
    }
)
async def run_main_agent_flow(
    request_data: AgentQueryRequest,
    graph_service: GraphService = Depends(get_graph_service)
):
    """
    Main endpoint to run the multi-agent flow.
    Receives a query, executes the graph, and streams progress and final results.
    """
    original_query = request_data.query

    async def event_stream_generator():
        # Unique ID for this request, could be useful for more advanced queue management
        # request_id = str(uuid.uuid4())

        # Register handler specifically for this request's lifetime
        progress_tracker.register_handler(sse_progress_handler)
        print(f"SSE progress handler registered for query: {original_query[:50]}...")

        try:
            start_event = StartEvent(query=original_query, timestamp=datetime.now(timezone.utc).isoformat())
            yield start_event.to_sse()

            graph_execution_task = asyncio.create_task(
                graph_service.execute_graph_async(original_query)
            )

            while not graph_execution_task.done() or not sse_progress_queue.empty():
                try:
                    progress_event = await asyncio.wait_for(sse_progress_queue.get(), timeout=0.2) # Increased timeout slightly
                    yield progress_event.to_sse()
                    sse_progress_queue.task_done()
                except asyncio.TimeoutError:
                    if graph_execution_task.done() and sse_progress_queue.empty(): # Check if queue is also empty
                        break
                    await asyncio.sleep(0.1) # Brief pause if no event but task not done
                except Exception as e_queue:
                    err_event = ErrorEvent(
                        error_message=f"Error reading SSE progress queue: {str(e_queue)}",
                        original_query=original_query,
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                    yield err_event.to_sse()
                    break

            final_state_dict = await graph_execution_task

            if final_state_dict.get("error_message"):
                error_msg = final_state_dict["error_message"]
                final_error_event = ErrorEvent(
                    source_agent=final_state_dict.get("sender_agent"),
                    error_message=str(error_msg),
                    original_query=original_query,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                yield final_error_event.to_sse()
            elif final_state_dict.get("summary"):
                summary_pydantic_obj = final_state_dict["summary"]
                summary_dict = summary_pydantic_obj.model_dump() if hasattr(summary_pydantic_obj, 'model_dump') else summary_pydantic_obj

                complete_event = CompleteEvent(
                    original_query=original_query,
                    final_summary=summary_dict,
                    has_errors=bool(final_state_dict.get("error_message")),
                    error_details=str(final_state_dict.get("error_message")) if final_state_dict.get("error_message") else None,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                yield complete_event.to_sse()
            else:
                unknown_error_event = ErrorEvent(
                    error_message="Processing completed but no final summary or specific error was found.",
                    original_query=original_query,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                yield unknown_error_event.to_sse()

        except asyncio.CancelledError:
            print(f"Client disconnected for query: {original_query}")
            try:
                cancel_err_event = ErrorEvent(
                    error_message="Client disconnected.",
                    original_query=original_query,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                yield cancel_err_event.to_sse()
            except Exception:
                pass
        except Exception as e:
            print(f"Critical error in event_stream_generator for query '{original_query}': {type(e).__name__} - {e}")
            critical_err_event = ErrorEvent(
                error_message=f"Critical server error: {str(e)}",
                original_query=original_query,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            yield critical_err_event.to_sse()
        finally:
            progress_tracker.unregister_handler(sse_progress_handler)
            print(f"SSE progress handler unregistered for query: {original_query[:50]}...")

    return StreamingResponse(event_stream_generator(), media_type="text/event-stream")