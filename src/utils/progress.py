from datetime import datetime, timezone
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.style import Style
from rich.text import Text
from typing import Dict, Optional, Callable, List, Any

console = Console(width=120)

class AgentProgress:
    """Manages and displays progress for multiple agents."""

    def __init__(self):
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.table = Table(show_header=False, box=None, padding=(0, 1))
        self.live = Live(self.table, console=console, refresh_per_second=10, vertical_overflow="visible")
        self.started = False
        self.update_handlers: List[Callable[[str, Optional[Dict[str, Any]], str, str], None]] = []

    def register_handler(self, handler: Callable[[str, Optional[Dict[str, Any]], str, str], None]):
        """Registers a handler to be called on agent status updates."""
        self.update_handlers.append(handler)
        return handler

    def unregister_handler(self, handler: Callable[[str, Optional[Dict[str, Any]], str, str], None]):
        """Unregisters a previously registered handler."""
        if handler in self.update_handlers:
            self.update_handlers.remove(handler)

    def start(self):
        """Starts the progress display."""
        if not self.started:
            self.live.start(refresh=True)
            self.started = True

    def stop(self):
        """Stops the progress display."""
        if self.started:
            self.live.stop()
            self.started = False
            console.print()

    def update_status(self, agent_name: str, optional_data: Optional[Dict[str, Any]] = None, status_message: str = ""):
        """Updates the status of an agent."""
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
            self._refresh_display()

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Gets the current status of all agents."""
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
        """Converts agent_name to a display-friendly format."""
        if agent_name.endswith("Agent"):
            return agent_name[:-5].replace("_", " ").title()
        if agent_name.endswith("_agent"):
            return agent_name[:-6].replace("_", " ").title()
        return agent_name.replace("_", " ").title()


    def _refresh_display(self):
        """Refreshes the progress display table."""
        self.table = Table(show_header=False, box=None, padding=(0,1))
        self.table.add_column(width=120)

        agent_display_order = ["PlannerAgent", "ExecutionAgent", "AnalysisAgent", "SummaryAgent", "GraphBuilder", "RoutingLogic (AfterPlanner)", "RoutingLogic (AfterExecution)", "RoutingLogic (AfterAnalysis)"]

        sorted_agent_names = sorted(
            self.agent_status.keys(),
            key=lambda x: agent_display_order.index(x) if x in agent_display_order else float('inf')
        )

        for agent_name in sorted_agent_names:
            info = self.agent_status[agent_name]
            status_msg = info.get("status_message", "")
            opt_data = info.get("optional_data", {})

            style = Style(color="yellow")
            symbol = "⏳"

            if "lỗi" in status_msg.lower() or "thất bại" in status_msg.lower() or "error" in status_msg.lower():
                style = Style(color="red", bold=True)
                symbol = "❌"
            elif "hoàn thành" in status_msg.lower() or "done" in status_msg.lower() or "thành công" in status_msg.lower() or "sẵn sàng" in status_msg.lower() or "hợp lệ" in status_msg.lower():
                style = Style(color="green", bold=True)
                symbol = "✅"
            elif "bắt đầu" in status_msg.lower() or "đang" in status_msg.lower():
                style = Style(color="blue")
                symbol = "⚙️"


            agent_display_name = self._get_display_name(agent_name)

            status_text = Text()
            status_text.append(f"{symbol} ", style=style)
            status_text.append(f"{agent_display_name:<25}", style=Style(bold=True, color="default"))

            optional_data_parts = []
            if task_id_val := opt_data.get("task_id"):
                optional_data_parts.append(f"Task: {task_id_val}")
            if current_step_val := opt_data.get("current_step"):
                optional_data_parts.append(f"Bước: {current_step_val}")


            if optional_data_parts:
                 status_text.append(f"[{', '.join(optional_data_parts)}] ", style=Style(color="cyan"))

            status_text.append(status_msg, style=style)

            self.table.add_row(status_text)

        if self.started:
            self.live.update(self.table, refresh=True)

progress_tracker = AgentProgress() 