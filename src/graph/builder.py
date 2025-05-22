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
    """Quyết định bước tiếp theo sau khi PlannerAgent hoàn thành."""
    agent_name_log = "RoutingLogic (AfterPlanner)"
    progress_tracker.update_status(agent_name_log, status_message="Kiểm tra kết quả lập kế hoạch...")
    show_agent_reasoning(f"Kiểm tra trạng thái sau PlannerAgent.", agent_name_log, color="grey50")

    if state.get("error_message"):
        console.print(f"[bold red]Lỗi từ Planner: {state['error_message']}. Chuyển sang Summary.[/]", style="dim red")
        progress_tracker.update_status(agent_name_log, status_message="Lỗi Planner, chuyển sang Summary.")
        return "summary_agent"

    plan = state.get("plan")
    if not plan or not plan.steps:
        console.print("[bold yellow]Planner không tạo được các bước thực thi hợp lệ. Chuyển sang Summary.[/]", style="dim yellow")
        progress_tracker.update_status(agent_name_log, status_message="Kế hoạch rỗng, chuyển sang Summary.")
        if not state.get("error_message"):
             state["error_message"] = "Planner Agent không tạo được bước thực thi nào."
        return "summary_agent"

    console.print("[green]Kế hoạch đã được tạo thành công, chuyển sang ExecutionAgent.[/]", style="dim green")
    progress_tracker.update_status(agent_name_log, status_message="Kế hoạch hợp lệ, chuyển sang Execution.")
    return "execution_agent"

def _route_after_execution(state: AgentState) -> str:
    """Quyết định bước tiếp theo sau khi ExecutionAgent hoàn thành."""
    agent_name_log = "RoutingLogic (AfterExecution)"
    progress_tracker.update_status(agent_name_log, status_message="Kiểm tra kết quả thực thi...")
    show_agent_reasoning(f"Kiểm tra trạng thái sau ExecutionAgent.", agent_name_log, color="grey50")

    if state.get("error_message"):
        console.print(f"[bold red]Lỗi từ Execution: {state['error_message']}. Chuyển sang Summary.[/]", style="dim red")
        progress_tracker.update_status(agent_name_log, status_message="Lỗi thực thi, chuyển sang Summary.")
        return "summary_agent"

    if not state.get("executed_tool_results"):
        console.print("[bold yellow]ExecutionAgent không tạo ra kết quả thực thi nào. Chuyển sang Analysis.[/]", style="dim yellow")
        progress_tracker.update_status(agent_name_log, status_message="Không có kết quả tool, chuyển sang Analysis.")
    else:
        console.print("[green]Thực thi công cụ hoàn tất, chuyển sang AnalysisAgent.[/]", style="dim green")
        progress_tracker.update_status(agent_name_log, status_message="Thực thi xong, chuyển sang Analysis.")
    return "analysis_agent"

def _route_after_analysis(state: AgentState) -> str:
    """Quyết định bước tiếp theo sau khi AnalysisAgent hoàn thành."""
    agent_name_log = "RoutingLogic (AfterAnalysis)"
    progress_tracker.update_status(agent_name_log, status_message="Kiểm tra kết quả phân tích...")
    show_agent_reasoning(f"Kiểm tra trạng thái sau AnalysisAgent.", agent_name_log, color="grey50")

    if state.get("error_message"):
        console.print(f"[bold red]Lỗi từ Analysis: {state['error_message']}. Chuyển sang Summary.[/]", style="dim red")
        progress_tracker.update_status(agent_name_log, status_message="Lỗi phân tích, chuyển sang Summary.")
    else:
        console.print("[green]Phân tích hoàn tất, chuyển sang SummaryAgent.[/]", style="dim green")
        progress_tracker.update_status(agent_name_log, status_message="Phân tích xong, chuyển sang Summary.")
    return "summary_agent"

def build_graph() -> StateGraph:
    """
    Xây dựng và trả về StateGraph cho hệ thống đa agent.
    """
    progress_tracker.update_status("GraphBuilder", status_message="Bắt đầu xây dựng graph...")
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
    console.print("[bold green]LangGraph biên dịch thành công.[/]", style="dim green")
    progress_tracker.update_status("GraphBuilder", status_message="Graph biên dịch thành công.")
    return app 