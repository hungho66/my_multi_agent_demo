from datetime import datetime, timezone
from typing import Dict, Optional, Callable, List, Any

class AgentProgress:
    """Quản lý và hiển thị tiến trình cho nhiều agent."""

    def __init__(self):
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.started = False
        self.update_handlers: List[Callable[[str, Optional[Dict[str, Any]], str, str], None]] = []
        self.show_all_logs = False  # Tùy chọn hiển thị tất cả logs

    def register_handler(self, handler: Callable[[str, Optional[Dict[str, Any]], str, str], None]):
        """Đăng ký handler được gọi khi cập nhật trạng thái agent."""
        self.update_handlers.append(handler)
        return handler

    def unregister_handler(self, handler: Callable[[str, Optional[Dict[str, Any]], str, str], None]):
        """Hủy đăng ký handler đã đăng ký trước đó."""
        if handler in self.update_handlers:
            self.update_handlers.remove(handler)

    def start(self):
        """Bắt đầu hiển thị tiến trình."""
        if not self.started:
            self.started = True
            print("\n--- BẮT ĐẦU THEO DÕI TIẾN TRÌNH ---")

    def stop(self):
        """Dừng hiển thị tiến trình."""
        if self.started:
            self.started = False
            print("\n--- KẾT THÚC THEO DÕI TIẾN TRÌNH ---\n")

    def update_status(self, agent_name: str, optional_data: Optional[Dict[str, Any]] = None, status_message: str = ""):
        """Cập nhật trạng thái của một agent."""
        if agent_name not in self.agent_status:
            self.agent_status[agent_name] = {"status_message": "", "optional_data": {}}

        if optional_data:
            self.agent_status[agent_name]["optional_data"].update(optional_data)
        if status_message:
            self.agent_status[agent_name]["status_message"] = status_message

        timestamp = datetime.now(timezone.utc).isoformat()
        self.agent_status[agent_name]["timestamp"] = timestamp

        for handler in self.update_handlers:
            handler(agent_name, self.agent_status[agent_name].get("optional_data"), status_message, timestamp)

        if self.started:
            self._print_status_update(agent_name, status_message, optional_data)

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Lấy trạng thái hiện tại của tất cả agents."""
        return {
            agent_name: {
                "optional_data": info.get("optional_data", {}),
                "status_message": info.get("status_message", ""),
                "display_name": self._get_display_name(agent_name),
                "timestamp": info.get("timestamp", "")
            }
            for agent_name, info in self.agent_status.items()
        }

    def _get_display_name(self, agent_name: str) -> str:
        """Chuyển agent_name thành định dạng hiển thị thân thiện."""
        if agent_name.endswith("Agent"):
            return agent_name[:-5].replace("_", " ").title()
        if agent_name.endswith("_agent"):
            return agent_name[:-6].replace("_", " ").title()
        return agent_name.replace("_", " ").title()

    def _print_status_update(self, agent_name: str, status_message: str, optional_data: Optional[Dict[str, Any]] = None):
        """In cập nhật trạng thái của agent."""
        # Chỉ hiển thị log quan trọng nếu không bật show_all_logs
        if not self.show_all_logs:
            if any(skip_term in agent_name.lower() for skip_term in ["routing", "internal"]):
                return
            if status_message and (
                not any(important_term in status_message.lower() for important_term in 
                      ["lỗi", "bắt đầu", "hoàn thành", "thành công", "thất bại", "đang thực thi"])
            ):
                return

        display_name = self._get_display_name(agent_name)
        
        # Tạo indicator dựa trên trạng thái
        indicator = "•"
        if "lỗi" in status_message.lower() or "thất bại" in status_message.lower():
            indicator = "✗"
        elif "hoàn thành" in status_message.lower() or "thành công" in status_message.lower():
            indicator = "✓"
        
        # Tạo thông tin thêm từ optional_data
        data_info = ""
        if optional_data:
            parts = []
            if task_id := optional_data.get("task_id"):
                parts.append(f"Task: {task_id}")
            if current_step := optional_data.get("current_step"):
                parts.append(f"Bước: {current_step}")
            if parts:
                data_info = f" [{', '.join(parts)}]"
        
        # In trạng thái
        print(f"{indicator} {display_name:<20}{data_info} - {status_message}")

progress_tracker = AgentProgress() 