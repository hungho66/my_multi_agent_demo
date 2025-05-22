import asyncio
from typing import Dict, Any, Optional
from src.graph.builder import build_graph
from src.graph.state import AgentState, FinalSummary
from src.utils.progress import progress_tracker
import json
from datetime import datetime, timezone

_compiled_graph = None

def get_compiled_graph():
    """Gets or creates the compiled LangGraph."""
    global _compiled_graph
    if _compiled_graph is None:
        progress_tracker.update_status("GraphService", status_message="Compiling LangGraph for the first time...")
        graph = build_graph()
        _compiled_graph = graph.compile()
        progress_tracker.update_status("GraphService", status_message="LangGraph compiled and ready.")
    return _compiled_graph

class GraphService:
    """
    Service responsible for executing the agent graph and managing the flow.
    """
    def __init__(self):
        self.app = get_compiled_graph() # Get the compiled graph

    async def execute_graph_async(self, query: str) -> Dict[str, Any]:
        """
        Executes the graph asynchronously.
        This function is intended to be run, for example, in a separate thread
        by FastAPI to avoid blocking the event loop for synchronous parts of the graph.
        """
        initial_state: AgentState = {
            "query": query,
            "messages": [],
            "plan": None,
            "executed_tool_results": [],
            "analysis_result": None,
            "summary": None,
            "sender_agent": "User", # Initial sender
            "error_message": None
        }

        final_state_dict: Dict[str, Any] = {}
        progress_tracker.update_status("GraphService", status_message=f"Starting graph execution for query: {query[:50]}...")

        try:
            # Using app.astream for asynchronous iteration over graph states
            async for state_snapshot_dict in self.app.astream(initial_state, stream_mode="values"):
                final_state_dict = state_snapshot_dict # Keep track of the latest state

                # Optional: Log or update progress based on intermediate states if needed
                # agent_name = state_snapshot_dict.get("sender_agent", "System")
                # current_task_msg = ""
                # if "plan" in state_snapshot_dict and state_snapshot_dict["plan"]:
                #     current_task_msg = f"Current plan steps: {len(state_snapshot_dict['plan'].steps)}"

                # progress_tracker.update_status(
                #     "GraphService",
                #     status_message=f"Graph progressing. Current sender: {agent_name}. {current_task_msg}"
                # )

            # Ensure 'summary' is a Pydantic model instance if present
            if "summary" in final_state_dict and final_state_dict["summary"] is not None:
                 if not isinstance(final_state_dict["summary"], FinalSummary):
                     try:
                         # Attempt to parse if it's a dictionary (e.g., from a previous non-Pydantic state)
                         final_state_dict["summary"] = FinalSummary(**final_state_dict["summary"])
                     except Exception as parse_error:
                         print(f"Warning: Could not parse 'summary' into FinalSummary model: {parse_error}")
                         final_state_dict["summary"] = None # or handle as error

            progress_tracker.update_status("GraphService", status_message=f"Graph execution finished for query: {query[:50]}.")
            return final_state_dict

        except Exception as e:
            error_message = f"Critical error during graph execution: {type(e).__name__} - {str(e)}"
            progress_tracker.update_status("GraphService", status_message=f"ERROR: {error_message}")
            # Return an error state
            return {
                "query": query,
                "messages": initial_state.get("messages", []),
                "plan": None,
                "executed_tool_results": [],
                "analysis_result": None,
                "summary": None,
                "sender_agent": "SystemError",
                "error_message": error_message
            }

_graph_service_instance: Optional[GraphService] = None

def get_graph_service() -> GraphService:
    """
    Factory function to provide a GraphService instance for FastAPI dependencies.
    Currently creates a singleton for the compiled graph.
    """
    global _graph_service_instance
    if _graph_service_instance is None:
        _graph_service_instance = GraphService()
    return _graph_service_instance