from langgraph.graph import StateGraph, END
from .state import AgentState, show_agent_reasoning
from ..agents.planner_agent import planner_agent_node
from ..agents.execution_agent import execution_agent_node
from ..agents.analysis_agent import analysis_agent_node
from ..agents.summary_agent import summary_agent_node
from ..utils.progress import progress_tracker
from typing import Dict, Any

def _route_after_planning(state: AgentState) -> str:
    """Quyết định bước tiếp theo sau khi PlannerAgent hoàn thành."""
    agent_name_log = "RoutingLogic (AfterPlanner)"
    progress_tracker.update_status(agent_name_log, status_message="Kiểm tra kết quả lập kế hoạch...")
    show_agent_reasoning(f"Kiểm tra trạng thái sau PlannerAgent.", agent_name_log)

    if state.get("error_message"):
        print(f"• Routing: Lỗi từ Planner: {state['error_message'][:50]}... Chuyển sang Summary.")
        progress_tracker.update_status(agent_name_log, status_message="Lỗi Planner, chuyển sang Summary.")
        return "summary_agent"

    plan = state.get("plan")
    if not plan or not plan.steps:
        print("• Routing: Planner không tạo được các bước thực thi hợp lệ. Chuyển sang Summary.")
        progress_tracker.update_status(agent_name_log, status_message="Kế hoạch rỗng, chuyển sang Summary.")
        if not state.get("error_message"):
             state["error_message"] = "Planner Agent không tạo được bước thực thi nào."
        return "summary_agent"

    print("• Routing: Kế hoạch đã được tạo thành công, chuyển sang ExecutionAgent.")
    progress_tracker.update_status(agent_name_log, status_message="Kế hoạch hợp lệ, chuyển sang Execution.")
    return "execution_agent"

def _route_after_execution(state: AgentState) -> str:
    """Quyết định bước tiếp theo sau khi ExecutionAgent hoàn thành."""
    agent_name_log = "RoutingLogic (AfterExecution)"
    progress_tracker.update_status(agent_name_log, status_message="Kiểm tra kết quả thực thi...")
    show_agent_reasoning(f"Kiểm tra trạng thái sau ExecutionAgent.", agent_name_log)

    if state.get("error_message"):
        print(f"• Routing: Lỗi từ Execution: {state['error_message'][:50]}... Chuyển sang Summary.")
        progress_tracker.update_status(agent_name_log, status_message="Lỗi thực thi, chuyển sang Summary.")
        return "summary_agent"

    if not state.get("executed_tool_results"):
        print("• Routing: ExecutionAgent không tạo ra kết quả thực thi. Chuyển sang Analysis.")
        progress_tracker.update_status(agent_name_log, status_message="Không có kết quả tool, chuyển sang Analysis.")
    else:
        print("• Routing: Thực thi công cụ hoàn tất, chuyển sang AnalysisAgent.")
        progress_tracker.update_status(agent_name_log, status_message="Thực thi xong, chuyển sang Analysis.")
    return "analysis_agent"

def _route_after_analysis(state: AgentState) -> str:
    """Quyết định bước tiếp theo sau khi AnalysisAgent hoàn thành."""
    agent_name_log = "RoutingLogic (AfterAnalysis)"
    progress_tracker.update_status(agent_name_log, status_message="Kiểm tra kết quả phân tích...")
    show_agent_reasoning(f"Kiểm tra trạng thái sau AnalysisAgent.", agent_name_log)

    if state.get("error_message"):
        print(f"• Routing: Lỗi từ Analysis: {state['error_message'][:50]}... Chuyển sang Summary.")
        progress_tracker.update_status(agent_name_log, status_message="Lỗi phân tích, chuyển sang Summary.")
    else:
        print("• Routing: Phân tích hoàn tất, chuyển sang SummaryAgent.")
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
    print("• LangGraph biên dịch thành công.")
    progress_tracker.update_status("GraphBuilder", status_message="Graph biên dịch thành công.")
    return app 