import os
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from typing import Optional

load_dotenv(override=True)

console = Console()

def init_chat_base_model(provider_name: Optional[str] = None, model_name_override: Optional[str] = None) -> BaseChatModel:
    """
    Khởi tạo và trả về instance của chat model dựa trên provider được chỉ định
    hoặc biến môi trường LLM_PROVIDER. Hiện tại chỉ hỗ trợ Google Gemini.
    """
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDVaY8V0R-jDmLzTbQQZCS7Mzy8ZPetPXI")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    LLM_PROVIDER = provider_name or os.getenv("LLM_PROVIDER", "google_genai")
    GEMINI_MODEL = model_name_override or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Nếu provider là google_genai thì trả về ChatGoogleGenerativeAI
    if LLM_PROVIDER == "google_genai":
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
    else:
        raise NotImplementedError(f"Chưa hỗ trợ provider: {LLM_PROVIDER}")