import argparse
from dotenv import load_dotenv
from .graph.builder import build_graph
from .utils.progress import progress_tracker
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.json import JSON
import os
import json
from .graph.state import AgentState, Plan, ToolExecutionResult, Analysis, FinalSummary

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

console = Console(width=140)

def display_state_details(state_snapshot: AgentState, step_number: int):
    """Displays detailed state information at a specific step."""
    console.print(Panel(f"[bold #1E90FF]State Details at Step {step_number}[/]", border_style="#1E90FF", expand=False))

    if state_snapshot.get('plan'):
        plan_obj = state_snapshot['plan']
        if isinstance(plan_obj, Plan):
            console.print(Panel(JSON(plan_obj.model_dump_json(indent=2)), title="[bold magenta]Plan[/]", border_style="magenta", expand_all=False))
        else:
             console.print(Panel(str(plan_obj), title="[bold magenta]Plan (Raw)[/]", border_style="magenta"))

    if state_snapshot.get('executed_tool_results'):
        results = state_snapshot['executed_tool_results']
        console.print(Panel(title="[bold blue]Tool Execution Results[/]", border_style="blue", expand_all=False))
        if isinstance(results, list) and all(isinstance(r, ToolExecutionResult) for r in results):
            for res in results:
                console.print(JSON(res.model_dump_json(indent=2)))
                console.print("---")
        else:
            for res_dict in results if isinstance(results, list) else [results]: # Handle single dict if not list
                 console.print(str(res_dict))
                 console.print("---")

    if state_snapshot.get('analysis_result'):
        analysis_obj = state_snapshot['analysis_result']
        if isinstance(analysis_obj, Analysis):
            console.print(Panel(JSON(analysis_obj.model_dump_json(indent=2)), title="[bold green]Analysis Result[/]", border_style="green", expand_all=False))
        else:
            console.print(Panel(str(analysis_obj), title="[bold green]Analysis Result (Raw)[/]", border_style="green"))

    if state_snapshot.get('error_message'):
        console.print(Panel(f"[bold red]Current Error Message:[/bold red] {state_snapshot['error_message']}", title="[red]Error[/red]", border_style="red"))

    console.print("-" * console.width)

def main():
    parser = argparse.ArgumentParser(description="Run multi-agent demo with Gemini, LangGraph, and reasoning display.")
    parser.add_argument("query", type=str, help="Query for the agent system to process.")
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Display detailed graph state after each agent step."
    )
    args = parser.parse_args()

    console.rule(f"[bold #4682B4]MULTI-AGENT AI DEMO (GEMINI & LANGGRAPH)[/]", style="#4682B4")
    console.print(Panel(f"[bold yellow]Processing query:[/] \"{args.query}\"", expand=False, border_style="yellow", padding=(1,2)))

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

    console.print(f"\n[bold #FF69B4]--- STARTING AGENT FLOW ---[/bold #FF69B4]")
    final_response_summary_text: str = "No summary generated or process encountered an error."
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
                console.print(Panel(f"[bold red]ERROR ENCOUNTERED![/]\nError from Agent '{active_agent}': {error_msg}",
                                    title="[red]CRITICAL ERROR[/red]", border_style="red", expand=True))

            current_summary = current_state_snapshot.get("summary")
            if isinstance(current_summary, FinalSummary):
                 final_summary_obj = current_summary

        if final_summary_obj:
            final_response_summary_text = final_summary_obj.overall_answer
            if final_summary_obj.confidence_level:
                final_response_summary_text += f"\n\n[italic]Confidence Level:[/italic] {final_summary_obj.confidence_level}"
            if final_summary_obj.limitations:
                final_response_summary_text += f"\n[italic]Limitations:[/italic] {final_summary_obj.limitations}"
        elif initial_inputs.get("error_message"):
             final_response_summary_text = f"Processing error: {initial_inputs['error_message']}"
        elif current_state_snapshot and current_state_snapshot.get("error_message"): # Check last state for error
             final_response_summary_text = f"Processing error: {current_state_snapshot['error_message']}"


        progress_tracker.stop()

        console.print(Panel(Text(final_response_summary_text, justify="left"),
                              title="[bold #1E90FF]FINAL RESULT FROM AGENT SYSTEM[/]",
                              expand=True, border_style="#1E90FF", padding=(1,2)))

    except Exception as e:
        progress_tracker.stop()
        console.print(Panel(f"[bold red]Unexpected error during graph execution:[/] {type(e).__name__} - {str(e)}",
                              title="[red]SYSTEM ERROR[/red]", border_style="red", expand=True))
        console.print_exception(show_locals=False)

if __name__ == "__main__":
    main()