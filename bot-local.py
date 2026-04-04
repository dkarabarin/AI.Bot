"""
Локальная RAG система для работы с PDF-документами по термодинамике
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple
import time
from datetime import datetime
import re

from dotenv import load_dotenv

# LangChain для Ollama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Tavily для веб-поиска
from tavily import TavilyClient

# Langfuse для наблюдаемости
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

# Наш RAG модуль
from rag import ThermodynamicsKnowledgeBase

# Отключаем предупреждения
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================================
# Конфигурация
# ============================================================================

BOOKS_DIR = Path("./books")

# Ollama (локальная модель)
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_ENABLED = False

# Параметры RAG
K_RETRIEVAL = 5
MAX_HISTORY = 10

# ============================================================================
# УЛУЧШЕННЫЙ СИСТЕМНЫЙ ПРОМПТ
# ============================================================================

SYSTEM_PROMPT = """Ты — преподаватель по технической термодинамике.

ВАЖНЫЕ ПРАВИЛА ФОРМАТИРОВАНИЯ:
1. Все формулы ОБЯЗАТЕЛЬНО заключай в $$...$$ для отдельных формул или $...$ для формул в тексте
2. Пример: "Формула теплопроводности: $$k = \\frac{Q \\cdot \\ln(r_2/r_1)}{2\\pi L \\Delta T}$$"
3. Пример в тексте: "Коэффициент $k$ измеряется в Вт/(м·К)"
4. Используй \\frac{}{} для дробей, \\cdot для умножения
5. Греческие буквы: \\alpha, \\beta, \\gamma, \\Delta, \\pi

Пример правильной формулы:
$$\\Delta S = \\int \\frac{dQ}{T}$$

Для лабораторных работ давай пошаговые инструкции с явными формулами.
"""

# ============================================================================
# Инициализация Langfuse
# ============================================================================

langfuse = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    try:
        langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        langfuse.auth_check()
        LANGFUSE_ENABLED = True
        print("✅ Langfuse инициализирован")
    except Exception as e:
        print(f"⚠️ Langfuse не инициализирован: {e}")
else:
    print("⚠️ Langfuse отключен")

# ============================================================================
# Веб-поиск через Tavily
# ============================================================================

class TavilySearch:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key) if api_key else None
        self._available = self.client is not None

    def search(self, query: str, max_results: int = 3) -> Optional[str]:
        if not self._available:
            return None
        try:
            response = self.client.search(
                query, search_depth="basic",
                include_answer=False, max_results=max_results,
            )
            results = response.get("results", [])
            if not results:
                return None
            formatted = []
            for r in results[:max_results]:
                title = r.get("title", "Без названия")
                content = r.get("content", "")
                formatted.append(f"📄 **{title}**\n{content[:500]}")
            return "\n\n---\n\n".join(formatted)
        except Exception as e:
            logger.error(f"Ошибка Tavily: {e}")
            return None

    def is_available(self) -> bool:
        return self._available

# ============================================================================
# Инициализация
# ============================================================================

llm = ChatOpenAI(
    openai_api_key="fake_key",
    openai_api_base=OLLAMA_BASE,
    model_name=OLLAMA_MODEL,
    temperature=0.7,
    max_tokens=2048,
)

tavily = TavilySearch(TAVILY_API_KEY)
knowledge_base = ThermodynamicsKnowledgeBase(BOOKS_DIR)
knowledge_base.load()

# ============================================================================
# Функции ответов
# ============================================================================

@observe()
def answer_from_pdf(question: str, trace_id: str = None) -> Optional[str]:
    if not knowledge_base.vectorstore:
        return None
    
    context_chunks = knowledge_base.get_relevant_chunks(question, k=K_RETRIEVAL)
    if not context_chunks:
        return None
    
    context = "\n\n---\n\n".join(context_chunks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"\n\n--- ИЗ PDF-ДОКУМЕНТОВ ---\n{context}"),
        HumanMessage(content=question),
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Ошибка RAG: {e}")
        return None

@observe()
def answer_from_web(question: str, trace_id: str = None) -> Optional[str]:
    if not tavily.is_available():
        return None
    
    search_results = tavily.search(question, max_results=3)
    if not search_results:
        return None
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"\n\n--- ВЕБ-ПОИСК (Tavily) ---\n{search_results}"),
        HumanMessage(content=question),
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Ошибка веб-ответа: {e}")
        return None

@observe()
def answer_direct(question: str, trace_id: str = None) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Ошибка LLM: {e}")
        return f"Ошибка: {e}"

@observe()
def get_answer(question: str, session_id: str = None) -> Tuple[str, str]:
    if LANGFUSE_ENABLED:
        langfuse_context.update_current_trace(
            name="question_answer",
            user_id=session_id or "web_user",
            session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    
    # Пробуем PDF
    print("  🔍 Поиск в PDF...")
    answer = answer_from_pdf(question)
    if answer:
        if LANGFUSE_ENABLED:
            langfuse_context.score_current_trace(name="source", value=1.0)
        return answer, "📚 PDF"
    
    # Пробуем веб-поиск
    print("  🌐 Поиск в интернете...")
    answer = answer_from_web(question)
    if answer:
        if LANGFUSE_ENABLED:
            langfuse_context.score_current_trace(name="source", value=0.8)
        return answer, "🌐 Интернет"
    
    # Используем LLM
    print("  🤖 Использование LLM...")
    answer = answer_direct(question)
    if LANGFUSE_ENABLED:
        langfuse_context.score_current_trace(name="source", value=0.6)
    return answer, "🎓 LLM"

# ============================================================================
# Консольный интерфейс (оставлен для обратной совместимости)
# ============================================================================

def main():
    print("Запуск консольного интерфейса...")
    while True:
        try:
            question = input("\n❓ Вопрос: ").strip()
            if question.lower() in ['/quit', '/exit']:
                break
            if question:
                answer, source = get_answer(question)
                print(f"\n{source} ОТВЕТ:\n{answer}\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()