from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from ..graph.state import AgentState, show_agent_reasoning, Plan, PlanStep
from ..llm.models import init_chat_base_model
from ..utils.progress import progress_tracker
from typing import List, Dict, Any, Optional
import uuid

def planner_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    PlannerAgent: Receives user query and creates a detailed plan with steps (PlanStep).
    Each PlanStep specifies the agent_or_tool_name and required_input.
    This agent will articulate its thought process in the Plan object.
    """
    current_query = state["query"]
    agent_name_log = "PlannerAgent"
    progress_tracker.update_status(agent_name_log, status_message=f"Received query for planning: '{current_query[:50]}...'")

    show_agent_reasoning(f"Starting to plan for: '{current_query}'", agent_name_log, color="magenta")

    llm = init_chat_base_model()
    structured_llm = llm.with_structured_output(Plan)

    system_prompt = """Bạn là một PlannerAgent chuyên nghiệp. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và tạo ra một kế hoạch chi tiết, từng bước để trả lời câu hỏi đó.
Mỗi bước trong kế hoạch cần bao gồm:
1.  `task_id`: Mã định danh duy nhất cho bước, ví dụ: 'task_001'.
2.  `task_description`: Mô tả chi tiết công việc cần thực hiện.
3.  `agent_or_tool_name`: Tên agent hoặc tool sẽ thực hiện bước này. Chỉ được chọn một trong các giá trị sau:
    * `search_executor`: Dùng để tìm kiếm thông tin trên internet, tin tức, sự kiện, định nghĩa, v.v.
    * `weather_executor`: Dùng để lấy thông tin thời tiết hiện tại của một thành phố cụ thể.
4.  `required_input`: Dictionary chứa input cần thiết cho agent/tool đó.
    * Với `search_executor`, key phải là `query` và value là chuỗi tìm kiếm. Ví dụ: {"query": "Lịch sử Việt Nam"}.
    * Với `weather_executor`, key phải là `city` và value là tên thành phố. Ví dụ: {"city": "Hà Nội"}.
5.  `reasoning`: Giải thích ngắn gọn tại sao bước này cần thiết để trả lời câu hỏi gốc.

Phần `thought` trong output cần giải thích cách bạn phân tích câu hỏi để xây dựng các bước kế hoạch.
Nếu câu hỏi phức tạp (ví dụ: vừa cần tìm kiếm vừa cần thời tiết), hãy tách thành nhiều `PlanStep`.
Đảm bảo `agent_or_tool_name` chỉ là 'search_executor' hoặc 'weather_executor'.
Đảm bảo `required_input` đúng key ('query' cho search, 'city' cho weather).
Output PHẢI đúng định dạng JSON của model Pydantic `Plan`.

LUÔN LUÔN trả lời bằng tiếng Việt.
"""
    user_prompt = f"Hãy lập kế hoạch chi tiết để trả lời câu hỏi sau: '{current_query}'"
    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    progress_tracker.update_status(agent_name_log, status_message=f"Requesting LLM ({llm.model_name if hasattr(llm, 'model_name') else 'N/A'}) to create plan...")
    show_agent_reasoning(f"Requesting LLM (Model: {llm.model_name if hasattr(llm, 'model_name') else 'N/A'}) to create plan...", agent_name_log, color="magenta")

    plan_output: Optional[Plan] = None
    error_message_for_state: Optional[str] = None

    try:
        ai_response_plan = structured_llm.invoke(prompt_messages)
        if isinstance(ai_response_plan, Plan):
            plan_output = ai_response_plan
            for i, step in enumerate(plan_output.steps):
                step.task_id = f"task_{str(uuid.uuid4())[:6]}_{i+1}" # Ensure unique, short task_id
            show_agent_reasoning(plan_output, agent_name_log, color="magenta")
            progress_tracker.update_status(agent_name_log, status_message=f"Plan created with {len(plan_output.steps)} steps.")
        else:
            error_message_for_state = "LLM did not return a valid Plan object."
            show_agent_reasoning(error_message_for_state, agent_name_log, color="red")
            progress_tracker.update_status(agent_name_log, status_message=f"ERROR: {error_message_for_state}")

    except Exception as e:
        error_message_for_state = f"Error calling LLM for planning: {type(e).__name__} - {str(e)}"
        show_agent_reasoning(error_message_for_state, agent_name_log, color="red")
        progress_tracker.update_status(agent_name_log, status_message=f"ERROR: {error_message_for_state}")

    if not plan_output or not plan_output.steps:
        fallback_thought = "Fallback plan: performing a simple search for the entire query."
        fallback_steps = [PlanStep(
            task_id=f"task_fallback_{str(uuid.uuid4())[:6]}",
            task_description=f"Search for general information about: '{current_query}'.",
            agent_or_tool_name="search_executor",
            required_input={"query": current_query},
            reasoning="Fallback step due to planning error."
        )]
        plan_output = Plan(
            original_query=current_query,
            thought=fallback_thought,
            steps=fallback_steps
        )
        if not error_message_for_state:
            error_message_for_state = "Could not create a detailed plan, using fallback."
        show_agent_reasoning("Using fallback plan.", agent_name_log, color="yellow")
        progress_tracker.update_status(agent_name_log, status_message="Using fallback plan.")

    planner_aimessage_content = plan_output.model_dump_json(indent=2)
    update_data: Dict[str, Any] = {
        "plan": plan_output,
        "messages": [AIMessage(content=planner_aimessage_content, name=agent_name_log)],
        "sender_agent": agent_name_log
    }
    if error_message_for_state:
        update_data["error_message"] = error_message_for_state

    progress_tracker.update_status(agent_name_log, status_message="Planning complete.")
    return update_data 