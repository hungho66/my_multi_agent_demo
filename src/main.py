import argparse
from dotenv import load_dotenv
from .graph.builder import build_graph
from .utils.progress import progress_tracker
import os
import json
from .graph.state import AgentState, Plan, ToolExecutionResult, Analysis, FinalSummary

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

def display_state_details(state_snapshot: AgentState, step_number: int):
    """Hiển thị thông tin trạng thái chi tiết tại một bước cụ thể."""
    print(f"\n--- Trạng thái chi tiết tại bước {step_number} ---")

    if state_snapshot.get('plan'):
        plan_obj = state_snapshot['plan']
        if isinstance(plan_obj, Plan):
            print("\n--- KẾ HOẠCH ---")
            print(json.dumps(plan_obj.model_dump(), indent=2, ensure_ascii=False))
        else:
             print("\n--- KẾ HOẠCH (Thô) ---")
             print(str(plan_obj))

    if state_snapshot.get('executed_tool_results'):
        results = state_snapshot['executed_tool_results']
        print("\n--- KẾT QUẢ THỰC THI CÔNG CỤ ---")
        if isinstance(results, list) and all(isinstance(r, ToolExecutionResult) for r in results):
            for res in results:
                print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))
                print("---")
        else:
            for res_dict in results if isinstance(results, list) else [results]: # Handle single dict if not list
                 print(str(res_dict))
                 print("---")

    if state_snapshot.get('analysis_result'):
        analysis_obj = state_snapshot['analysis_result']
        if isinstance(analysis_obj, Analysis):
            print("\n--- KẾT QUẢ PHÂN TÍCH ---")
            print(json.dumps(analysis_obj.model_dump(), indent=2, ensure_ascii=False))
        else:
            print("\n--- KẾT QUẢ PHÂN TÍCH (Thô) ---")
            print(str(analysis_obj))

    if state_snapshot.get('error_message'):
        print(f"\n--- LỖI ---")
        print(f"Thông báo lỗi hiện tại: {state_snapshot['error_message']}")

    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Run multi-agent demo with Gemini, LangGraph, and reasoning display.")
    parser.add_argument("query", type=str, help="Query for the agent system to process.")
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Display detailed graph state after each agent step."
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("HỆ THỐNG ĐA AGENT VỚI LANGCHAIN & LANGGRAPH")
    print("=" * 50)
    print(f"\nXử lý câu hỏi: \"{args.query}\"\n")

    app = build_graph()

    initial_inputs: AgentState = {
        "query": args.query,
        "messages": [],
        "plan": None,
        "executed_tool_results": [],
        "analysis_result": None,
        "summary": None,
        "sender_agent": "User",
        "error_message": None
    }

    print("--- BẮT ĐẦU QUÁ TRÌNH XỬ LÝ ---")
    final_response_summary_text: str = "Không có tóm tắt nào được tạo hoặc quá trình gặp lỗi."
    final_summary_obj: Optional[FinalSummary] = None

    progress_tracker.start()

    try:
        for i, current_state_snapshot in enumerate(app.stream(initial_inputs, stream_mode="values")):
            active_agent = current_state_snapshot.get('sender_agent', 'System')
            # Progress tracker is updated within agent nodes and routing logic.

            if args.show_steps:
                display_state_details(current_state_snapshot, i + 1)

            if current_state_snapshot.get("error_message"):
                error_msg = current_state_snapshot["error_message"]
                print(f"\n!!! LỖI NGHIÊM TRỌNG !!!")
                print(f"Lỗi từ Agent '{active_agent}': {error_msg}")

            current_summary = current_state_snapshot.get("summary")
            if isinstance(current_summary, FinalSummary):
                 final_summary_obj = current_summary

        if final_summary_obj:
            final_response_summary_text = final_summary_obj.overall_answer
            if final_summary_obj.confidence_level:
                final_response_summary_text += f"\n\nĐộ tin cậy: {final_summary_obj.confidence_level}"
            if final_summary_obj.limitations:
                final_response_summary_text += f"\nHạn chế: {final_summary_obj.limitations}"
        elif initial_inputs.get("error_message"):
             final_response_summary_text = f"Lỗi xử lý: {initial_inputs['error_message']}"
        elif current_state_snapshot and current_state_snapshot.get("error_message"): # Check last state for error
             final_response_summary_text = f"Lỗi xử lý: {current_state_snapshot['error_message']}"

        progress_tracker.stop()

        print("\n" + "=" * 50)
        print("KẾT QUẢ CUỐI CÙNG TỪ HỆ THỐNG")
        print("=" * 50)
        print(f"\n{final_response_summary_text}\n")

    except Exception as e:
        progress_tracker.stop()
        print(f"\n!!! LỖI HỆ THỐNG !!!")
        print(f"Lỗi không mong đợi trong quá trình thực thi: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()