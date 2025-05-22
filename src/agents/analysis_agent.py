from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from ..graph.state import AgentState, show_agent_reasoning, ToolExecutionResult, Analysis
from ..llm.models import init_chat_base_model
from ..utils.progress import progress_tracker
from typing import List, Dict, Any, Optional
import json

def analysis_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    AnalysisAgent: Analyzes results from executed tools.
    Extracts key insights, determines overall sentiment, and assesses data source quality.
    """
    query = state["query"]
    executed_tool_results: List[ToolExecutionResult] = state.get("executed_tool_results", [])
    agent_name_log = "AnalysisAgent"
    progress_tracker.update_status(agent_name_log, status_message="Starting results analysis.")

    if not executed_tool_results:
        message_content = "No tool execution results to analyze. Cannot proceed."
        show_agent_reasoning(message_content, agent_name_log, color="yellow")
        progress_tracker.update_status(agent_name_log, status_message="ERROR: No input data.")
        analysis_output_obj = Analysis(
            original_query=query,
            analysis_summary=message_content,
            key_insights=[],
            sentiment="neutral",
            data_sources_quality="cannot be assessed due to missing input data.",
            reasoning_process="No data from tools to perform analysis."
        )
        return {
            "analysis_result": analysis_output_obj,
            "messages": [AIMessage(content=message_content, name=agent_name_log)],
            "sender_agent": agent_name_log,
            "error_message": state.get("error_message") or message_content
        }

    show_agent_reasoning(f"Received {len(executed_tool_results)} tool execution results for query: '{query}'", agent_name_log, color="green")

    llm = init_chat_base_model()
    structured_llm = llm.with_structured_output(Analysis)

    formatted_tool_outputs_for_llm = []
    has_successful_tool_execution = False
    for i, result in enumerate(executed_tool_results):
        task_desc = result.task_description if hasattr(result, 'task_description') and result.task_description else "N/A"
        content_line = f"Information from Task ID '{result.task_id}' (Tool: {result.tool_name}, Description: \"{task_desc}\", Input: {result.tool_input}):\n"

        if result.is_successful and result.raw_output is not None:
            has_successful_tool_execution = True
            output_str = ""
            if isinstance(result.raw_output, str):
                try:
                    parsed_json = json.loads(result.raw_output)
                    if isinstance(parsed_json, list):
                         output_str = f"  Parsed result is a list of {len(parsed_json)} items. Example: {str(parsed_json[:2]) if len(parsed_json) > 0 else 'Empty list.'}"
                    elif isinstance(parsed_json, dict):
                         output_str = f"  Parsed result is a dict with keys: {list(parsed_json.keys())}. Example: {str(list(parsed_json.values())[:2]) if len(parsed_json.values()) > 0 else 'Empty dict.'}"
                    else:
                        output_str = str(parsed_json)
                except json.JSONDecodeError:
                    output_str = str(result.raw_output)
            elif isinstance(result.raw_output, list):
                output_str = f"  Result is a list of {len(result.raw_output)} items. Example: {str(result.raw_output[:2]) if len(result.raw_output) > 0 else 'Empty list.'}"
            elif isinstance(result.raw_output, dict):
                output_str = f"  Result is a dict with keys: {list(result.raw_output.keys())}. Example: {str(list(result.raw_output.values())[:2]) if len(result.raw_output.values()) > 0 else 'Empty dict.'}"
            else:
                output_str = str(result.raw_output)
            content_line += f"  Output (summary): {output_str[:1500]}\n"
        elif result.error:
            content_line += f"  EXECUTION ERROR: {result.error}\n"
        else:
            content_line += "  Tool executed successfully but provided no significant output.\n"
        formatted_tool_outputs_for_llm.append(content_line)

    if not has_successful_tool_execution:
        formatted_tool_outputs_for_llm.append("IMPORTANT NOTE: All executed tools encountered errors or returned no useful results. Analyze based on this fact.")

    combined_content_for_llm = "\n---\n".join(formatted_tool_outputs_for_llm)
    system_prompt = """Bạn là một Information Analyst sắc bén. Nhiệm vụ của bạn là tổng hợp và phân tích thông tin từ các kết quả thực thi tool liên quan đến câu hỏi gốc của người dùng.
Dựa trên "Thông tin tổng hợp từ các tool" được cung cấp, hãy:
1.  `analysis_summary`: Viết một đoạn tóm tắt phân tích, nêu bật các phát hiện chính.
2.  `key_insights`: Liệt kê các thông tin, kết luận hoặc insight quan trọng nhất.
3.  `sentiment`: Đánh giá cảm xúc tổng thể (ví dụ: 'tích cực', 'tiêu cực', 'trung lập', 'hỗn hợp', 'không xác định') của thông tin liên quan đến câu hỏi gốc.
4.  `data_sources_quality`: Đánh giá ngắn gọn về chất lượng, độ tin cậy của nguồn dữ liệu (tức là output của tool). Nếu có lỗi, hãy nêu rõ.
5.  `reasoning_process`: Mô tả ngắn gọn quá trình suy luận để đưa ra kết luận phân tích. Bạn đã kết nối các thông tin như thế nào? Có mâu thuẫn gì không?

Tập trung vào thông tin liên quan nhất cho câu hỏi gốc. Nếu thiếu dữ liệu hoặc tool lỗi, hãy nêu rõ trong `data_sources_quality` và `analysis_summary`.
Kết quả PHẢI đúng định dạng JSON của model Pydantic `Analysis`.

LUÔN LUÔN trả lời bằng tiếng Việt.
"""
    user_prompt_content = (
        f"Câu hỏi gốc của người dùng: '{query}'\n\n"
        f"Thông tin tổng hợp từ các tool:\n{combined_content_for_llm}\n\n"
        "Dựa trên các thông tin trên, hãy phân tích và trả về kết quả đúng định dạng model Pydantic `Analysis`."
    )
    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt_content)
    ]

    progress_tracker.update_status(agent_name_log, status_message=f"Requesting LLM ({llm.model_name if hasattr(llm, 'model_name') else 'N/A'}) for analysis...")
    show_agent_reasoning(f"Requesting LLM (Model: {llm.model_name if hasattr(llm, 'model_name') else 'N/A'}) for analysis...", agent_name_log, color="green")

    analysis_llm_output: Optional[Analysis] = None
    error_message_for_state: Optional[str] = None

    try:
        ai_response_analysis = structured_llm.invoke(prompt_messages)
        if isinstance(ai_response_analysis, Analysis):
            analysis_llm_output = ai_response_analysis
            show_agent_reasoning(analysis_llm_output, agent_name_log, color="green")
            progress_tracker.update_status(agent_name_log, status_message="Analysis successful.")
        else:
            error_message_for_state = "LLM did not return a valid Analysis object."
            show_agent_reasoning(error_message_for_state, agent_name_log, color="red")
            progress_tracker.update_status(agent_name_log, status_message=f"ERROR: {error_message_for_state}")

    except Exception as e:
        error_message_for_state = f"Error calling LLM for analysis: {type(e).__name__} - {str(e)}"
        show_agent_reasoning(error_message_for_state, agent_name_log, color="red")
        progress_tracker.update_status(agent_name_log, status_message=f"ERROR: {error_message_for_state}")

    if not analysis_llm_output:
        analysis_llm_output = Analysis(
            original_query=query,
            analysis_summary=f"Analysis failed: {error_message_for_state or 'Unknown LLM error.'}",
            key_insights=["No insights due to analysis error."],
            sentiment="error",
            data_sources_quality="Cannot be assessed due to analysis error.",
            reasoning_process=f"Analysis process encountered an error: {error_message_for_state or 'Unknown LLM error.'}"
        )
        if not error_message_for_state:
             error_message_for_state = "Analysis failed, using default error response."

    analysis_aimessage_content = analysis_llm_output.model_dump_json(indent=2)
    update_data: Dict[str, Any] = {
        "analysis_result": analysis_llm_output,
        "messages": [AIMessage(content=analysis_aimessage_content, name=agent_name_log)],
        "sender_agent": agent_name_log
    }
    if error_message_for_state:
        update_data["error_message"] = state.get("error_message") or error_message_for_state

    progress_tracker.update_status(agent_name_log, status_message="Analysis complete.")
    return update_data 