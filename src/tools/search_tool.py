from langchain_community.tools.tavily_search import TavilySearchResults
import os
from ..utils.progress import progress_tracker
from typing import Dict, Any, Optional
from datetime import datetime
from langchain_core.tools import tool

@tool
def tavily_search(query: str) -> str:
    """
    Công cụ tìm kiếm web sử dụng Tavily API.
    Nhận vào một câu truy vấn (query) và trả về kết quả tìm kiếm.
    Hữu ích để tìm thông tin cập nhật, sự kiện mới, hoặc kiến thức cụ thể.
    
    Args:
        query: Truy vấn cần tìm kiếm
        
    Returns:
        Kết quả tìm kiếm dạng văn bản
    """
    tool_name = "TavilySearchTool"
    progress_tracker.update_status(tool_name, status_message=f"Đang tìm kiếm: '{query}'...")
    
    # Lấy thời gian hiện tại theo định dạng Việt Nam
    current_time_vi = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        error_msg = "TAVILY_API_KEY không được tìm thấy trong biến môi trường."
        progress_tracker.update_status(tool_name, status_message=f"LỖI: {error_msg}")
        return error_msg
    
    try:
        # Khởi tạo tool Tavily với số kết quả tối đa và cấu hình để yêu cầu thông tin mới nhất
        tavily_tool = TavilySearchResults(
            max_results=3, 
            tavily_api_key=api_key,
            search_depth="advanced",
            include_raw_content=True,
            include_images=False
        )
        
        # Thêm hậu tố yêu cầu thông tin mới nhất vào truy vấn
        search_query = f"{query} (cập nhật mới nhất {current_time_vi})"
        
        # Thực hiện tìm kiếm với truy vấn đã tăng cường
        results = tavily_tool.invoke({"query": search_query})
        progress_tracker.update_status(tool_name, status_message="Tìm kiếm thành công.")
        
        # Thêm thông tin về thời gian tìm kiếm vào kết quả
        return f"Kết quả tìm kiếm (thời điểm: {current_time_vi}):\n\n{results}"
    except Exception as e:
        error_msg = f"Lỗi khi thực hiện tìm kiếm Tavily: {str(e)}"
        progress_tracker.update_status(tool_name, status_message=f"LỖI: {error_msg}")
        return error_msg

def get_tavily_search_tool(max_results: int = 3):
    """
    Khởi tạo và trả về công cụ tìm kiếm Tavily.
    Đảm bảo TAVILY_API_KEY được thiết lập trong môi trường.
    """
    tool_name = "TavilySearchTool"
    progress_tracker.update_status(tool_name, status_message="Khởi tạo công cụ tìm kiếm Tavily...")
    
    try:
        # Đăng ký tool đã được decorator
        progress_tracker.update_status(tool_name, status_message="Công cụ tìm kiếm Tavily đã sẵn sàng.")
        return tavily_search
    except Exception as e:
        error_msg = f"Lỗi khi khởi tạo công cụ tìm kiếm Tavily: {e}"
        progress_tracker.update_status(tool_name, status_message=f"LỖI: {error_msg}")
        raise ConnectionError(error_msg) from e 