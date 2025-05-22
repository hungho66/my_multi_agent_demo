from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from ..graph.state import AgentState, show_agent_reasoning, PlanStep, ToolExecutionResult
from ..tools.search_tool import get_tavily_search_tool
from ..tools.weather_tool import get_current_weather
from ..utils.progress import progress_tracker
from typing import List, Dict, Any, Optional
import json

def execution_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    ExecutionAgent: Nhận kế hoạch từ PlannerAgent và thực thi từng bước.
    Với mỗi PlanStep, xác định công cụ cần thiết và gọi nó với input tương ứng.
    Kết quả (hoặc lỗi) của mỗi lần thực thi được ghi lại.
    """
    plan = state.get("plan")
    agent_name_log = "ExecutionAgent"
    progress_tracker.update_status(agent_name_log, status_message="Bắt đầu thực thi kế hoạch.")

    if not plan or not plan.steps:
        message_content = "Bỏ qua thực thi: Không có kế hoạch hoặc kế hoạch rỗng."
        show_agent_reasoning(message_content, agent_name_log, color="yellow")
        progress_tracker.update_status(agent_name_log, status_message="LỖI: Kế hoạch không hợp lệ.")
        return {
            "executed_tool_results": [],
            "messages": [AIMessage(content=message_content, name=agent_name_log)],
            "sender_agent": agent_name_log,
            "error_message": state.get("error_message") or "Không có kế hoạch để thực thi."
        }

    show_agent_reasoning(f"Đang thực thi kế hoạch với {len(plan.steps)} bước cho câu hỏi: '{plan.original_query}'", agent_name_log, color="blue")

    tool_execution_results_for_this_node_run: List[ToolExecutionResult] = []
    search_tool = get_tavily_search_tool(max_results=2)
    overall_execution_successful = True

    for i, step in enumerate(plan.steps):
        task_id = step.task_id
        task_description = step.task_description
        progress_tracker.update_status(agent_name_log,
                                       optional_data={"task_id": task_id, "current_step": f"{i+1}/{len(plan.steps)}"},
                                       status_message=f"Đang thực thi: \"{step.task_description[:30]}...\"")
        show_agent_reasoning(f"Bước {i+1}/{len(plan.steps)} (ID: {task_id}): \"{step.task_description}\" sử dụng '{step.agent_or_tool_name}'", agent_name_log, color="blue")

        tool_output: Any = None
        error_msg_step: Optional[str] = None
        tool_name_from_plan = step.agent_or_tool_name.lower()
        tool_input_from_plan = step.required_input
        
        # Xử lý trường hợp tool_input_from_plan không phải dictionary
        if isinstance(tool_input_from_plan, str):
            try:
                # Thử chuyển đổi từ chuỗi JSON
                tool_input_from_plan = json.loads(tool_input_from_plan)
            except json.JSONDecodeError:
                # Nếu không phải JSON hợp lệ, tạo dictionary phù hợp
                if tool_name_from_plan == "search_executor":
                    tool_input_from_plan = {"query": tool_input_from_plan}
                elif tool_name_from_plan == "weather_executor":
                    tool_input_from_plan = {"city": tool_input_from_plan}
                else:
                    tool_input_from_plan = {"input": tool_input_from_plan}
        elif not isinstance(tool_input_from_plan, dict):
            # Nếu không phải string hoặc dict, chuyển thành dict
            tool_input_from_plan = {"query": str(tool_input_from_plan)}

        if not tool_input_from_plan:
            error_msg_step = f"Input không hợp lệ hoặc thiếu cho công cụ '{tool_name_from_plan}' (Task ID: {task_id})."
            show_agent_reasoning(error_msg_step, agent_name_log, color="red")
        else:
            try:
                if tool_name_from_plan == "search_executor":
                    query_input = tool_input_from_plan.get("query")
                    if not query_input or not isinstance(query_input, str):
                        error_msg_step = f"Thiếu hoặc không hợp lệ trường 'query' (phải là chuỗi) cho search_executor. Input: {tool_input_from_plan}"
                    else:
                        show_agent_reasoning(f"Gọi Tavily Search với truy vấn: '{query_input}'", agent_name_log, color="cyan")
                        tool_output = search_tool(query_input)
                elif tool_name_from_plan == "weather_executor":
                    city_input = tool_input_from_plan.get("city")
                    if not city_input or not isinstance(city_input, str):
                        error_msg_step = f"Thiếu hoặc không hợp lệ trường 'city' (phải là chuỗi) cho weather_executor. Input: {tool_input_from_plan}"
                    else:
                        show_agent_reasoning(f"Gọi Weather Tool cho thành phố: '{city_input}'", agent_name_log, color="cyan")
                        tool_output = get_current_weather.invoke({"city": city_input})
                else:
                    error_msg_step = f"Không xác định được công cụ/agent '{tool_name_from_plan}' được chỉ định trong kế hoạch (ID: {task_id})."

                if error_msg_step:
                     show_agent_reasoning(error_msg_step, agent_name_log, color="red")

            except Exception as e:
                error_msg_step = f"Lỗi khi thực thi công cụ '{tool_name_from_plan}' (Task ID: {task_id}) với input '{tool_input_from_plan}': {type(e).__name__} - {str(e)}"
                show_agent_reasoning(error_msg_step, agent_name_log, color="red")
                tool_output = None

        current_tool_run = ToolExecutionResult(
            task_id=task_id,
            task_description=task_description,
            tool_name=tool_name_from_plan,
            tool_input=tool_input_from_plan,
            raw_output=tool_output if not error_msg_step else None,
            error=error_msg_step,
            is_successful=not bool(error_msg_step)
        )
        tool_execution_results_for_this_node_run.append(current_tool_run)

        if error_msg_step:
            overall_execution_successful = False
            progress_tracker.update_status(agent_name_log,
                                           optional_data={"task_id": task_id},
                                           status_message=f"LỖI: \"{step.task_description[:30]}...\"")
            show_agent_reasoning(f"Công cụ '{tool_name_from_plan}' (Task ID: {task_id}) THẤT BẠI: {error_msg_step}", agent_name_log, color="red")
        else:
            output_summary = str(tool_output)[:100] + "..." if tool_output and len(str(tool_output)) > 100 else str(tool_output)
            progress_tracker.update_status(agent_name_log,
                                           optional_data={"task_id": task_id},
                                           status_message=f"Hoàn thành: \"{step.task_description[:30]}...\"")
            show_agent_reasoning(f"Kết quả công cụ '{tool_name_from_plan}' (Task ID: {task_id}): {output_summary}", agent_name_log, color="blue")

    num_tasks = len(plan.steps)
    num_successful_tasks = sum(1 for res in tool_execution_results_for_this_node_run if res.is_successful)
    execution_summary_message_content = f"Hoàn thành thực thi {num_successful_tasks}/{num_tasks} bước cho câu hỏi: '{plan.original_query}'."

    if not overall_execution_successful:
        num_errors = num_tasks - num_successful_tasks
        execution_summary_message_content += f" Gặp {num_errors} lỗi trong quá trình thực thi."

    update_data: Dict[str, Any] = {
        "executed_tool_results": tool_execution_results_for_this_node_run,
        "messages": [AIMessage(content=execution_summary_message_content, name=agent_name_log)],
        "sender_agent": agent_name_log
    }

    first_error_encountered = next((res.error for res in tool_execution_results_for_this_node_run if res.error), None)
    if first_error_encountered:
        update_data["error_message"] = state.get("error_message") or first_error_encountered

    final_status_msg = f"Thực thi kế hoạch hoàn tất ({num_successful_tasks}/{num_tasks} thành công)."
    if not overall_execution_successful:
        final_status_msg += " Có lỗi xảy ra."
    progress_tracker.update_status(agent_name_log, status_message=final_status_msg)
    return update_data 