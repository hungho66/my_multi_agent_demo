from langgraph.graph import StateGraph, END
from .state import AgentState, show_agent_reasoning
from ..agents.planner_agent import planner_agent_node
from ..agents.execution_agent import execution_agent_node
from ..agents.analysis_agent import analysis_agent_node
from ..agents.summary_agent import summary_agent_node
from ..utils.progress import progress_tracker
from rich.console import Console
from typing import Dict, Any

console = Console()

def _route_after_planning(state: AgentState) -> str:
    """Decides the next step after PlannerAgent completes."""
    agent_name_log = "RoutingLogic (AfterPlanner)"
    progress_tracker.update_status(agent_name_log, status_message="Checking planning results...")
    show_agent_reasoning(f"Checking state after PlannerAgent.", agent_name_log, color="grey50")

    if state.get("error_message"):
        console.print(f"[bold red]Error from Planner: {state['error_message']}. Routing to Summary.[/]", style="dim red")
        progress_tracker.update_status(agent_name_log, status_message="Planner error, routing to Summary.")
        return "summary_agent"

    plan = state.get("plan")
    if not plan or not plan.steps:
        console.print("[bold yellow]Planner did not generate valid execution steps. Routing to Summary.[/]", style="dim yellow")
        progress_tracker.update_status(agent_name_log, status_message="Empty plan, routing to Summary.")
        if not state.get("error_message"):
             state["error_message"] = "Planner Agent did not generate any execution steps."
        return "summary_agent"

    console.print("[green]Plan created successfully, routing to ExecutionAgent.[/]", style="dim green")
    progress_tracker.update_status(agent_name_log, status_message="Valid plan, routing to Execution.")
    return "execution_agent"

def _route_after_execution(state: AgentState) -> str:
    """Decides the next step after ExecutionAgent completes."""
    agent_name_log = "RoutingLogic (AfterExecution)"
    progress_tracker.update_status(agent_name_log, status_message="Checking execution results...")
    show_agent_reasoning(f"Checking state after ExecutionAgent.", agent_name_log, color="grey50")

    if state.get("error_message"):
        console.print(f"[bold red]Error from Execution: {state['error_message']}. Routing to Summary.[/]", style="dim red")
        progress_tracker.update_status(agent_name_log, status_message="Execution error, routing to Summary.")
        return "summary_agent"

    if not state.get("executed_tool_results"):
        console.print("[bold yellow]ExecutionAgent produced no tool results. Routing to Analysis.[/]", style="dim yellow")
        progress_tracker.update_status(agent_name_log, status_message="No tool results, routing to Analysis.")
    else:
        console.print("[green]Tool execution complete, routing to AnalysisAgent.[/]", style="dim green")
        progress_tracker.update_status(agent_name_log, status_message="Tool execution done, routing to Analysis.")
    return "analysis_agent"

def _route_after_analysis(state: AgentState) -> str:
    """Decides the next step after AnalysisAgent completes."""
    agent_name_log = "RoutingLogic (AfterAnalysis)"
    progress_tracker.update_status(agent_name_log, status_message="Checking analysis results...")
    show_agent_reasoning(f"Checking state after AnalysisAgent.", agent_name_log, color="grey50")

    if state.get("error_message"):
        console.print(f"[bold red]Error from Analysis: {state['error_message']}. Routing to Summary.[/]", style="dim red")
        progress_tracker.update_status(agent_name_log, status_message="Analysis error, routing to Summary.")
    else:
        console.print("[green]Analysis complete, routing to SummaryAgent.[/]", style="dim green")
        progress_tracker.update_status(agent_name_log, status_message="Analysis done, routing to Summary.")
    return "summary_agent"

def build_graph() -> StateGraph:
    """
    Builds and returns the StateGraph for the multi-agent system.
    """
    progress_tracker.update_status("GraphBuilder", status_message="Starting graph construction...")
    workflow = StateGraph(AgentState)

    workflow.add_node("planner_agent", planner_agent_node)
    workflow.add_node("execution_agent", execution_agent_node)
    workflow.add_node("analysis_agent", analysis_agent_node)
    workflow.add_node("summary_agent", summary_agent_node)

    workflow.set_entry_point("planner_agent")

    workflow.add_conditional_edges(
        "planner_agent",
        _route_after_planning,
        {"execution_agent": "execution_agent", "summary_agent": "summary_agent"}
    )
    workflow.add_conditional_edges(
        "execution_agent",
        _route_after_execution,
        {"analysis_agent": "analysis_agent", "summary_agent": "summary_agent"}
    )
    workflow.add_conditional_edges(
        "analysis_agent",
        _route_after_analysis,
        {"summary_agent": "summary_agent"}
    )
    workflow.add_edge("summary_agent", END)

    app = workflow.compile()
    console.print("[bold green]LangGraph compiled successfully.[/]", style="dim green")
    progress_tracker.update_status("GraphBuilder", status_message="Graph compiled successfully.")
    return app 