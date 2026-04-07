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

# Web Search
from tavily import TavilyClient
from ddgs import DDGS  # DuckDuckGo Search как fallback

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
# Гибридный веб-поиск (Tavily с fallback на DuckDuckGo)
# ============================================================================

class HybridWebSearch:
    """Гибридный поиск: Tavily как основной, DuckDuckGo как fallback"""
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        self.tavily_client = None
        self.tavily_available = False
        self.ddg_available = False
        self.active_search = None
        
        # Инициализация Tavily
        if tavily_api_key:
            try:
                self.tavily_client = TavilyClient(api_key=tavily_api_key)
                # Проверяем работу Tavily
                test_response = self.tavily_client.search("test", max_results=1)
                self.tavily_available = True
                self.active_search = "tavily"
                print("✅ Tavily поиск доступен (основной)")
            except Exception as e:
                print(f"⚠️ Tavily недоступен: {e}")
        
        # Инициализация DuckDuckGo как fallback
        try:
            with DDGS() as ddgs:
                test = list(ddgs.text("test", max_results=1))
            self.ddg_available = True
            if not self.tavily_available:
                self.active_search = "duckduckgo"
                print("✅ DuckDuckGo поиск доступен (fallback)")
            else:
                print("📌 DuckDuckGo доступен как резервный")
        except Exception as e:
            print(f"⚠️ DuckDuckGo недоступен: {e}")
        
        if not self.tavily_available and not self.ddg_available:
            print("❌ Веб-поиск полностью недоступен!")
    
    def search(self, query: str, max_results: int = 3) -> Optional[str]:
        """Выполняет поиск, автоматически выбирая доступный источник"""
        
        # Сначала пробуем Tavily
        if self.tavily_available:
            result = self._search_tavily(query, max_results)
            if result:
                return result
        
        # Если Tavily не сработал, пробуем DuckDuckGo
        if self.ddg_available:
            result = self._search_duckduckgo(query, max_results)
            if result:
                return result
        
        return None
    
    def _search_tavily(self, query: str, max_results: int = 3) -> Optional[str]:
        """Поиск через Tavily"""
        try:
            response = self.tavily_client.search(
                query, 
                search_depth="basic",
                include_answer=False, 
                max_results=max_results,
            )
            results = response.get("results", [])
            if not results:
                return None
            
            formatted = []
            for r in results[:max_results]:
                title = r.get("title", "Без названия")
                content = r.get("content", "")
                url = r.get("url", "")
                formatted.append(f"📄 **{title}**\n{content[:500]}\n🔗 {url}")
            
            return "\n\n---\n\n".join(formatted)
        except Exception as e:
            logger.error(f"Ошибка Tavily: {e}")
            return None
    
    def _search_duckduckgo(self, query: str, max_results: int = 3) -> Optional[str]:
        """Поиск через DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return None
            
            formatted = []
            for r in results[:max_results]:
                title = r.get("title", "Без названия")
                body = r.get("body", "")
                href = r.get("href", "")
                formatted.append(f"📄 **{title}**\n{body[:500]}\n🔗 {href}")
            
            return "\n\n---\n\n".join(formatted)
        except Exception as e:
            logger.error(f"Ошибка DuckDuckGo: {e}")
            return None
    
    def is_available(self) -> bool:
        """Проверяет доступность хотя бы одного поискового сервиса"""
        return self.tavily_available or self.ddg_available
    
    def get_active_source(self) -> str:
        """Возвращает активный источник поиска"""
        if self.tavily_available:
            return "Tavily"
        elif self.ddg_available:
            return "DuckDuckGo"
        else:
            return "None"

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

web_search = HybridWebSearch(TAVILY_API_KEY)
knowledge_base = ThermodynamicsKnowledgeBase(BOOKS_DIR)
knowledge_base.load()

# Для обратной совместимости с api.py
tavily = web_search  # Переименовываем для совместимости

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
    if not web_search.is_available():
        return None
    
    search_results = web_search.search(question, max_results=3)
    if not search_results:
        return None
    
    source_name = web_search.get_active_source()
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"\n\n--- ВЕБ-ПОИСК ({source_name}) ---\n{search_results}"),
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
    source_name = web_search.get_active_source()
    print(f"  🌐 Поиск в интернете ({source_name})...")
    answer = answer_from_web(question)
    if answer:
        if LANGFUSE_ENABLED:
            langfuse_context.score_current_trace(name="source", value=0.8)
        return answer, f"🌐 {source_name}"
    
    # Используем LLM
    print("  🤖 Использование LLM...")
    answer = answer_direct(question)
    if LANGFUSE_ENABLED:
        langfuse_context.score_current_trace(name="source", value=0.6)
    return answer, "🎓 LLM"

# ============================================================================
# Консольный интерфейс
# ============================================================================

def main():
    print("\n" + "="*50)
    print("🔧 Термодинамический консультант")
    print("="*50)
    print(f"📚 PDF база: {'загружена' if knowledge_base.vectorstore else 'не загружена'}")
    print(f"🌐 Веб-поиск: {web_search.get_active_source()}")
    print(f"🤖 LLM модель: {OLLAMA_MODEL}")
    print("="*50 + "\n")
    
    while True:
        try:
            question = input("\n❓ Вопрос (или /quit для выхода): ").strip()
            if question.lower() in ['/quit', '/exit', '/q']:
                break
            if question:
                answer, source = get_answer(question)
                print(f"\n{source} ОТВЕТ:\n{answer}\n")
                print("-"*50)
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("\n👋 До свидания!")

if __name__ == "__main__":
    main()