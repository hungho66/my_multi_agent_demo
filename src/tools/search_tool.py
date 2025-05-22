from langchain_community.tools.tavily_search import TavilySearchResults
import os
from ..utils.progress import progress_tracker

def get_tavily_search_tool(max_results: int = 3):
    """
    Initializes and returns the Tavily Search tool.
    Ensures TAVILY_API_KEY is set in your environment.
    """
    tool_name = "TavilySearchTool"
    progress_tracker.update_status(tool_name, status_message="Khởi tạo công cụ tìm kiếm Tavily...")
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        error_msg = "TAVILY_API_KEY không được tìm thấy trong biến môi trường."
        progress_tracker.update_status(tool_name, status_message=f"LỖI: {error_msg}")
        raise ValueError(error_msg)

    try:
        tool_instance = TavilySearchResults(max_results=max_results, tavily_api_key=api_key)
        progress_tracker.update_status(tool_name, status_message="Công cụ tìm kiếm Tavily đã sẵn sàng.")
        return tool_instance
    except Exception as e:
        error_msg = f"Lỗi khi khởi tạo TavilySearchResults: {e}"
        progress_tracker.update_status(tool_name, status_message=f"LỖI: {error_msg}")
        raise ConnectionError(error_msg) from e 