from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from ..graph.state import AgentState, show_agent_reasoning, Analysis, FinalSummary
from ..llm.models import init_chat_base_model
from ..utils.progress import progress_tracker
from typing import Dict, Any, Optional

def summary_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    SummaryAgent: Synthesizes all prior information, especially from AnalysisAgent,
    to create a final, coherent, and user-friendly summary answering the original query.
    Also provides a confidence assessment and limitations, if any.
    """
    query = state["query"]
    analysis_data: Optional[Analysis] = state.get("analysis_result")
    agent_name_log = "SummaryAgent"
    progress_tracker.update_status(agent_name_log, status_message="Starting final summary generation.")
    show_agent_reasoning("Compiling final summary...", agent_name_log, color="yellow")

    llm = init_chat_base_model()
    structured_llm = llm.with_structured_output(FinalSummary)

    analysis_summary_for_prompt = "No detailed analysis results available or analysis process encountered an error."
    if analysis_data:
        analysis_summary_for_prompt = (
            f"Key conclusions from Analysis Agent: \"{analysis_data.analysis_summary}\"\n"
            f"Key Insights: {'; '.join(analysis_data.key_insights) if analysis_data.key_insights else 'None extracted.'}\n"
            f"Overall Sentiment: {analysis_data.sentiment}\n"
            f"Data Source Quality Assessment: \"{analysis_data.data_sources_quality}\"\n"
            f"Analysis Reasoning (if available): \"{analysis_data.reasoning_process or 'Not provided.'}\""
        )
    elif state.get("error_message"): # If no analysis_data but there's a general error message in state
        analysis_summary_for_prompt = f"The preceding process encountered an error: {state.get('error_message')}. No detailed analysis is available."


    system_prompt = """Bạn là chuyên gia tổng hợp thông tin. Nhiệm vụ của bạn là sử dụng thông tin từ các agent khác (đặc biệt là AnalysisAgent) để tạo ra câu trả lời cuối cùng, dễ hiểu, đầy đủ cho câu hỏi gốc của người dùng.
Trong trường `overall_answer`:
- Trả lời trực tiếp, đầy đủ cho câu hỏi gốc.
- Tổng hợp các insight chính từ phần phân tích.
- Trình bày rõ ràng, ngắn gọn, dễ hiểu.

Trong `confidence_level`:
- Đánh giá mức độ tin cậy của câu trả lời (Cao, Trung bình, Thấp) dựa trên chất lượng dữ liệu và phân tích.

Trong `limitations`:
- Nêu rõ các giới hạn, thiếu sót, giả định hoặc lỗi nếu có.

Kết quả PHẢI đúng định dạng JSON của model Pydantic `FinalSummary`.

LUÔN LUÔN trả lời bằng tiếng Việt.
"""
    human_prompt_content = (
        f"Câu hỏi gốc của người dùng: '{query}'\n\n"
        f"Thông tin từ AnalysisAgent:\n{analysis_summary_for_prompt}\n\n"
        "Dựa trên tất cả thông tin trên, hãy trả về bản tổng hợp cuối cùng đúng định dạng model Pydantic `FinalSummary`."
    )
    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt_content)
    ]

    progress_tracker.update_status(agent_name_log, status_message=f"Requesting LLM ({llm.model_name if hasattr(llm, 'model_name') else 'N/A'}) for final summary...")
    show_agent_reasoning(f"Requesting LLM (Model: {llm.model_name if hasattr(llm, 'model_name') else 'N/A'}) for final summary...", agent_name_log, color="yellow")

    summary_llm_output: Optional[FinalSummary] = None
    error_message_for_state: Optional[str] = None

    try:
        ai_response_summary = structured_llm.invoke(prompt_messages)
        if isinstance(ai_response_summary, FinalSummary):
            summary_llm_output = ai_response_summary
            show_agent_reasoning(summary_llm_output, agent_name_log, color="yellow")
            progress_tracker.update_status(agent_name_log, status_message="Summary generated successfully.")
        else:
            error_message_for_state = "LLM did not return a valid FinalSummary object."
            show_agent_reasoning(error_message_for_state, agent_name_log, color="red")
            progress_tracker.update_status(agent_name_log, status_message=f"ERROR: {error_message_for_state}")

    except Exception as e:
        error_message_for_state = f"Error calling LLM for summary: {type(e).__name__} - {str(e)}"
        show_agent_reasoning(error_message_for_state, agent_name_log, color="red")
        progress_tracker.update_status(agent_name_log, status_message=f"ERROR: {error_message_for_state}")

    if not summary_llm_output:
        summary_llm_output = FinalSummary(
            original_query=query,
            overall_answer=f"Could not generate a complete summary for query '{query}' due to an error: {error_message_for_state or 'Unknown LLM error.'}. Analysis information (if any): {analysis_summary_for_prompt}",
            confidence_level="Low",
            limitations=f"Summary generation failed: {error_message_for_state or 'Unknown LLM error.'}. " + (state.get("error_message") or "")
        )
        if not error_message_for_state:
             error_message_for_state = "Summary generation failed, using default error response."

    summary_aimessage_content = summary_llm_output.model_dump_json(indent=2)
    update_data: Dict[str, Any] = {
        "summary": summary_llm_output,
        "messages": [AIMessage(content=summary_aimessage_content, name=agent_name_log)],
        "sender_agent": agent_name_log
    }

    current_state_error = state.get("error_message")
    if error_message_for_state:
        update_data["error_message"] = error_message_for_state
    elif current_state_error :
        update_data["error_message"] = current_state_error

    progress_tracker.update_status(agent_name_log, status_message="Final summary complete.")
    return update_data 