# Demo Multi-Agent với Reasoning (LangGraph & Gemini)

Dự án này minh họa cách xây dựng một hệ thống multi-agent sử dụng LangGraph để điều phối, LLM (Google Gemini làm mặc định) để xử lý ngôn ngữ và đưa ra quyết định, cùng với các công cụ (tools) để tương tác với thế giới bên ngoài. Hệ thống có khả năng lập kế hoạch (planning), thực thi (execution), phân tích (analysis) và tổng hợp (summary) để trả lời các câu truy vấn phức tạp của người dùng.

Một backend API được xây dựng bằng FastAPI cho phép tương tác với hệ thống agent và stream các bước xử lý cũng như kết quả về client thông qua Server-Sent Events (SSE). Frontend (React + Vite) được cung cấp để người dùng có thể nhập truy vấn và theo dõi quá trình.

Điểm nhấn của dự án là khả năng hiển thị "reasoning" (quá trình suy nghĩ và output chi tiết) của từng agent, giúp hiểu rõ hơn về cách hệ thống đưa ra quyết định.

## Các tính năng chính

* **Kiến trúc Multi-Agent:**
    * `PlannerAgent`: Phân tích query và tạo kế hoạch chi tiết.
    * `ExecutionAgent`: Thực thi kế hoạch, gọi tools (Tavily Search, Weather API).
    * `AnalysisAgent`: Phân tích kết quả từ tools, rút ra insights.
    * `SummaryAgent`: Tổng hợp thông tin, tạo câu trả lời cuối cùng.
* **Tích hợp LLM:**
    * Sử dụng hàm `init_chat_model` tùy chỉnh để khởi tạo LLM.
    * Mặc định sử dụng Google Gemini (`google_google_genai` provider).
    * Hỗ trợ tùy chọn OpenAI và Anthropic qua biến môi trường.
    * Các agent sử dụng LLM với structured output (Pydantic models).
* **LangGraph Orchestration:**
    * Sử dụng `StateGraph` để định nghĩa luồng công việc.
    * `conditional_edges` để xử lý lỗi và định tuyến.
    * `AgentState` để lưu trữ và truyền tải thông tin.
* **Hiển thị Reasoning & Tiến trình:**
    * Hàm `show_agent_reasoning` sử dụng `rich` để hiển thị output trong CLI.
    * Module `src/utils/progress.py` cung cấp lớp `AgentProgress` để theo dõi và hiển thị tiến trình trong CLI và gửi về frontend qua SSE.
* **Công cụ (Tools):**
    * `TavilySearchResults`
    * `get_current_weather` (Custom tool)
* **Backend API (FastAPI):**
    * Endpoint `/agents/run_main_flow` (POST).
    * Server-Sent Events (SSE) để stream cập nhật.
    * Schemas trong `app/backend/models/`.
* **Frontend (React + Vite):**
    * Giao diện nhập query, hiển thị graph ReactFlow và kết quả.
    * Xử lý SSE để cập nhật UI.
* **Quản lý Dependencies:** `pyproject.toml` (uv/Poetry).
* **Theo dõi với LangSmith.**

## Cấu trúc thư mục dự án

```
/my_multi_agent_demo
|-- app/                                # Ứng dụng web (backend và frontend)
|   |-- backend/                        # Backend API sử dụng FastAPI
|   |   |-- __init__.py                 # Đánh dấu là Python package, thiết lập PYTHONPATH
|   |   |-- main.py                     # Điểm khởi chạy FastAPI, cấu hình CORS, routers
|   |   |-- models/                     # Pydantic models cho cấu trúc dữ liệu
|   |   |   |-- __init__.py
|   |   |   |-- schemas.py              # Schemas cho API request/response và SSE Data
|   |   |   |-- events.py               # Schemas cho Server-Sent Events
|   |   |-- routes/                     # API routes (endpoints)
|   |   |   |-- __init__.py             # Gom các routers con
|   |   |   |-- agent_router.py         # Endpoints cho agent graph
|   |   |   |-- health_router.py        # Endpoints kiểm tra "sức khỏe" API
|   |   |-- services/                   # Business logic, dịch vụ cốt lõi
|   |       |-- __init__.py
|   |       |-- graph_service.py        # Logic khởi tạo, chạy LangGraph, xử lý stream
|   |-- frontend/                       # Frontend (React + Vite)
|       |-- public/                     # Tài sản tĩnh (favicon, images)
|       |   |-- favicon.ico
|       |-- src/                        # Mã nguồn chính frontend
|       |   |-- App.tsx                 # Component React gốc, thiết lập layout và ReactFlow
|       |   |-- main.tsx                # Điểm khởi chạy của ứng dụng React
|       |   |-- index.css               # File CSS toàn cục, import TailwindCSS
|       |   |-- vite-env.d.ts           # Khai báo type cho Vite
|       |   |-- components/             # Các UI components tái sử dụng
|       |   |   |-- ui/                 # Các component cơ bản từ shadcn/ui (Button, Card, Dialog, etc.)
|       |   |   |-- Layout.tsx          # Component layout chính của trang
|       |   |   |-- Flow.tsx              # Component hiển thị graph bằng ReactFlow
|       |   |   |-- QueryInput.tsx        # Component cho ô nhập query và nút gửi
|       |   |   |-- OutputDisplay.tsx     # Component hiển thị kết quả cuối cùng
|       |   |-- contexts/               # React Contexts
|       |   |   |-- NodeStatusContext.tsx # Context quản lý trạng thái của các node agent
|       |   |-- data/                   # Dữ liệu tĩnh cho frontend
|       |   |   |-- agent-nodes-config.ts # Cấu hình các node agent cho ReactFlow
|       |   |-- edges/                  # Định nghĩa các loại cạnh (edge) cho ReactFlow
|       |   |   |-- index.ts
|       |   |-- hooks/                  # Các custom React hooks
|       |   |   |-- useSSE.ts           # Hook quản lý kết nối SSE
|       |   |-- lib/                    # Các hàm tiện ích cho frontend
|       |   |   |-- utils.ts            # Ví dụ: cn (classnames utility)
|       |   |-- nodes/                  # Định nghĩa và components cho các node ReactFlow
|       |   |   |-- types.ts            # Định nghĩa kiểu dữ liệu cho các node
|       |   |   |-- index.ts            # Export các node types và initial nodes
|       |   |   |-- CustomNode.tsx      # Component cho một node agent tùy chỉnh
|       |   |   |-- InputNode.tsx       # Component cho node nhập liệu
|       |   |   |-- OutputNode.tsx      # Component cho node hiển thị kết quả
|       |   |-- providers/              # React Context Providers
|       |   |   |-- ThemeProvider.tsx   # Quản lý theme (dark/light)
|       |   |-- services/               # Logic gọi API backend
|       |       |-- api.ts              # Hàm gọi API backend và xử lý SSE
|       |-- .github/
|       |   |-- dependabot.yml
|       |-- components.json             # Cấu hình shadcn/ui
|       |-- index.html                  # File HTML template gốc
|       |-- package.json                # Quản lý dependencies frontend
|       |-- postcss.config.mjs          # Cấu hình PostCSS
|       |-- tailwind.config.ts          # Cấu hình TailwindCSS
|       |-- tsconfig.json               # Cấu hình TypeScript cho src
|       |-- tsconfig.node.json          # Cấu hình TypeScript cho file cấu hình Node.js
|       |-- vite.config.ts              # Cấu hình Vite
|       |-- README.md                   # Hướng dẫn riêng cho frontend
|       |-- pnpm-lock.yaml              # (Hoặc package-lock.json / yarn.lock)
|-- src/                                # Mã nguồn Python cốt lõi của hệ thống multi-agent
|   |-- __init__.py
|   |-- agents/                       # Định nghĩa các individual agents
|   |   |-- __init__.py
|   |   |-- planner_agent.py          # Agent tạo kế hoạch
|   |   |-- execution_agent.py        # Agent thực thi kế hoạch và tools
|   |   |-- analysis_agent.py         # Agent phân tích kết quả
|   |   |-- summary_agent.py          # Agent tạo tóm tắt cuối cùng
|   |-- graph/                        # Logic xây dựng và điều phối LangGraph
|   |   |-- __init__.py
|   |   |-- state.py                  # Định nghĩa AgentState, Pydantic models, show_agent_reasoning
|   |   |-- builder.py                # Hàm build_graph
|   |-- llm/                          # Cấu hình và khởi tạo LLMs
|   |   |-- __init__.py
|   |   |-- models.py                 # Hàm init_chat_model
|   |-- tools/                        # Định nghĩa các tools cho agents
|   |   |-- __init__.py
|   |   |-- search_tool.py            # Tavily Search tool
|   |   |-- weather_tool.py           # Open-Meteo Weather tool
|   |-- utils/                        # Các module tiện ích chung
|   |   |-- __init__.py
|   |   |-- progress.py               # Theo dõi và hiển thị tiến trình agent
|   |-- main.py                       # Điểm khởi chạy CLI
|-- .env.example                      # File mẫu biến môi trường
|-- pyproject.toml                    # Cấu hình dự án Python (uv/Poetry)
|-- README.md                         # Hướng dẫn dự án
```


-----
## Cách chạy dự án

### Yêu cầu

* Python 3.9+
* `uv` (khuyến nghị) hoặc `Poetry`
* Node.js (LTS) và `npm` (hoặc `yarn`/`pnpm`)
* API Keys (chi tiết trong `.env.example`):
    * Google AI Studio API Key (cho Gemini)
    * Tavily API Key (cho Search tool)
    * (Tùy chọn) LangSmith, OpenAI, Anthropic API Keys

### 1. Cài đặt chung

a.  **Clone dự án** và `cd my_multi_agent_demo`
b.  **Cấu hình biến môi trường**: Sao chép `.env.example` thành `.env` và điền API keys.

### 2. Chạy Backend (FastAPI & LangGraph Agents)

a.  **Cài đặt dependencies Python**:
    * **Với `uv`**:
        ```bash
        python -m venv .venv
        source .venv/bin/activate  # Linux/macOS (hoặc .venv\Scripts\activate cho Windows)
        uv pip compile pyproject.toml -o requirements.txt
        uv pip sync requirements.txt
        ```
b.  **Khởi chạy server FastAPI**:
    (Trong thư mục gốc `my_multi_agent_demo` và đã kích hoạt virtual environment)
    ```bash
    uvicorn app.backend.main:app --reload --port 8000
    # hoặc: poetry run uvicorn app.backend.main:app --reload --port 8000
    ```
    Backend API sẽ chạy tại `http://localhost:8000`. Docs: `http://localhost:8000/docs`.

### 3. Chạy Frontend (React + Vite)

a.  **Mở terminal mới**, `cd app/frontend`
b.  **Cài đặt dependencies Node.js**: `npm install` (hoặc `yarn`/`pnpm`)
c.  **Khởi chạy server frontend**: `npm run dev`
    Frontend sẽ chạy tại `http://localhost:5173`.

### 4. Sử dụng ứng dụng

* Mở `http://localhost:5173` trong trình duyệt.
* Nhập câu truy vấn (ví dụ: "Thời tiết ở Hà Nội hôm nay và tin tức mới nhất về AI là gì?").
* Nhấn "Gửi Truy Vấn".
* Theo dõi các node agent trên graph thay đổi trạng thái và xem kết quả cuối cùng ở panel output.

### 5. (Tùy chọn) Chạy từ dòng lệnh (CLI)

(Từ thư mục gốc `my_multi_agent_demo` và đã kích hoạt virtual environment)
```bash
python -m src.main "Câu truy vấn của bạn" [--show-steps]
# hoặc: poetry run python -m src.main "Câu truy vấn của bạn" [--show-steps]
