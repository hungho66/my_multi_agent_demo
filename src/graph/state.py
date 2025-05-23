from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
import operator
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator
import json
from ..utils.progress import progress_tracker

class PlanStep(BaseModel):
    """A specific step in the execution plan generated by PlannerAgent."""
    task_id: str = Field(description="Unique identifier for this task, e.g., 'task_001'.")
    task_description: str = Field(description="Detailed description of the task to be performed.")
    agent_or_tool_name: str = Field(description="Name of the agent or tool. E.g., 'search_executor', 'weather_executor'.")
    required_input: Dict[str, Any] = Field(description="Dictionary containing input for the agent/tool. E.g., {'query': 'AI news'} or {'city': 'Hanoi'}.")
    reasoning: Optional[str] = Field(default=None, description="Brief explanation of why this step is necessary.")
    
    # Validator để tự động chuyển đổi chuỗi JSON thành dictionary
    @field_validator('required_input', mode='before')
    @classmethod
    def process_required_input(cls, v):
        # Nếu required_input là chuỗi, thử parse như JSON
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Nếu không phải JSON hợp lệ, tạo dict đơn giản
                return {"query": v}
        
        # Nếu không phải dictionary, bọc lại thành dictionary
        if not isinstance(v, dict):
            return {"query": str(v)}
            
        return v

class Plan(BaseModel):
    """The overall execution plan generated by PlannerAgent."""
    original_query: str = Field(description="The original query from the user.")
    thought: str = Field(description="The thought process of PlannerAgent to generate this plan.")
    steps: List[PlanStep] = Field(description="A list of PlanStep objects.")

class ToolExecutionResult(BaseModel):
    """Stores the result (or error) from executing a tool by ExecutionAgent."""
    task_id: str = Field(description="Identifier of the corresponding task in the plan.")
    task_description: Optional[str] = Field(None, description="Description of the executed task (from PlanStep).")
    tool_name: str = Field(description="Name of the tool that was called.")
    tool_input: Dict[str, Any] = Field(description="Input data provided to the tool.")
    raw_output: Any = Field(description="Raw output returned from the tool.")
    error: Optional[str] = Field(default=None, description="Error message if any issue occurred during tool execution.")
    is_successful: bool = Field(default=True, description="True if the tool executed successfully.")

class Analysis(BaseModel):
    """Result of information analysis from AnalysisAgent."""
    original_query: str = Field(description="The original user query.")
    analysis_summary: str = Field(description="Summary of the analysis based on collected information.")
    key_insights: List[str] = Field(description="List of key information points or insights derived.")
    sentiment: str = Field(description="Overall sentiment of the analyzed information (e.g., 'positive', 'negative', 'neutral').")
    data_sources_quality: str = Field(description="Brief assessment of the quality and reliability of the data sources.")
    reasoning_process: Optional[str] = Field(default=None, description="Description of AnalysisAgent's reasoning process.")

class FinalSummary(BaseModel):
    """Final summary generated by SummaryAgent to answer the user."""
    original_query: str = Field(description="The original user query.")
    overall_answer: str = Field(description="Comprehensive and coherent answer to the query.")
    confidence_level: str = Field(description="Estimated confidence level in the answer (High, Medium, Low).")
    limitations: Optional[str] = Field(default=None, description="Any limitations, missing information, or assumptions made.")

def merge_tool_execution_results(
    left: List[ToolExecutionResult], right: List[ToolExecutionResult]
) -> List[ToolExecutionResult]:
    return left + right

class AgentState(TypedDict):
    """
    Represents the state of the graph. Includes all information passed between agents.
    """
    query: str
    plan: Optional[Plan]
    executed_tool_results: Annotated[List[ToolExecutionResult], merge_tool_execution_results]
    analysis_result: Optional[Analysis]
    summary: Optional[FinalSummary]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender_agent: Optional[str]
    error_message: Optional[str]

def convert_to_serializable(obj):
    """Convert objects to JSON-serializable format recursively"""
    if hasattr(obj, "model_dump"):  # Handle Pydantic models
        return obj.model_dump()
    elif hasattr(obj, "to_dict"):  # Handle Pandas Series/DataFrame
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):  # Handle custom objects
        return obj.__dict__
    elif isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return str(obj)  # Fallback to string representation

def show_agent_reasoning(output: Any, agent_name: str, color: str = "cyan"):
    """
    Hiển thị suy luận của agent trong dạng đơn giản.
    Chỉ hiển thị thông tin quan trọng và quá trình suy luận.
    """
    print(f"\n{'=' * 20} {agent_name} {'=' * 20}")
    
    try:
        # Các đối tượng Pydantic
        if hasattr(output, 'model_dump') and callable(output.model_dump):
            data = convert_to_serializable(output)
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            print(json_str)
        # Chuỗi và các dạng dữ liệu khác
        elif isinstance(output, str):
            try:
                # Thử xử lý như JSON nếu có thể
                parsed_json = json.loads(output)
                json_str = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                print(json_str)
            except json.JSONDecodeError:
                # Nếu không phải JSON, in dạng text thường
                print(output)
        else:
            # Dict, list hoặc các dạng dữ liệu khác
            json_str = json.dumps(convert_to_serializable(output), indent=2, ensure_ascii=False)
            print(json_str)
    except Exception as e:
        # Fallback nếu không thể xử lý
        print(f"[Không thể hiển thị dạng dữ liệu: {type(output)}] {str(output)[:200]}")
        
    print("") # Dòng trống để tạo khoảng cách 