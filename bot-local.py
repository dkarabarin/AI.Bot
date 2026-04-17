"""
Локальная RAG система для работы с PDF-документами по термодинамике
С поддержкой стриминга (постепенной печати)
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, AsyncGenerator
import time
import re
import asyncio
import json

from dotenv import load_dotenv

# LangChain для Ollama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Tavily для веб-поиска
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# DuckDuckGo поиск
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# Наш RAG модуль
from rag import ThermodynamicsKnowledgeBase

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================================
# Конфигурация
# ============================================================================

BOOKS_DIR = Path("./books")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
K_RETRIEVAL = 5
SEARCH_MAX_RESULTS = 3

# ============================================================================
# СИСТЕМНЫЙ ПРОМПТ
# ============================================================================

SYSTEM_PROMPT = """Ты — преподаватель по технической термодинамике (ТТД) и тепломассообмену (ТМО).

ОБЛАСТЬ ЗНАНИЙ:
- Техническая термодинамика (ТТД): циклы, процессы, законы термодинамики
- Тепломассообмен (ТМО): теплопроводность, конвекция, излучение, массообмен

ПРИОРИТЕТ ИСТОЧНИКОВ:
1. В ПЕРВУЮ ОЧЕРЕДЬ используй материал из PDF-документов в папке books/
2. Если информации недостаточно — используй веб-поиск
3. Не придумывай факты. Если ответа нет — честно скажи об этом

ПРАВИЛА ФОРМАТИРОВАНИЯ ФОРМУЛ:
1. Все формулы ОБЯЗАТЕЛЬНО заключай в $$...$$ для отдельных формул
2. Для формул в тексте используй $...$
3. Используй \\frac{}{} для дробей, \\cdot для умножения
4. Греческие буквы: \\alpha, \\beta, \\gamma, \\Delta, \\pi

Пример: "Первый закон термодинамики: $$\\Delta U = Q - A$$"

Отвечай на русском языке. Будь полезным и точным.
"""

# ============================================================================
# ВЕБ-ПОИСК
# ============================================================================

class WebSearch:
    def __init__(self, api_key: Optional[str] = None):
        self.use_tavily = False
        self.use_duckduckgo = False
        
        if api_key and TAVILY_AVAILABLE:
            try:
                self.tavily_client = TavilyClient(api_key=api_key)
                self.use_tavily = True
                logger.info("✅ Веб-поиск: Tavily")
            except:
                pass
        
        if not self.use_tavily and DDGS_AVAILABLE:
            self.use_duckduckgo = True
            logger.info("✅ Веб-поиск: DuckDuckGo")
    
    def search(self, query: str, max_results: int = 3) -> Optional[str]:
        if self.use_tavily:
            return self._search_tavily(query, max_results)
        elif self.use_duckduckgo:
            return self._search_duckduckgo(query, max_results)
        return None
    
    def _search_tavily(self, query: str, max_results: int) -> Optional[str]:
        try:
            response = self.tavily_client.search(query, search_depth="basic", max_results=max_results)
            results = response.get("results", [])
            if not results:
                return None
            formatted = []
            for r in results[:max_results]:
                title = r.get("title", "")
                content = r.get("content", "")
                formatted.append(f"**{title}**\n{content[:500]}")
            return "\n\n---\n\n".join(formatted)
        except Exception as e:
            logger.error(f"Tavily error: {e}")
            return None
    
    def _search_duckduckgo(self, query: str, max_results: int) -> Optional[str]:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return None
                formatted = []
                for r in results[:max_results]:
                    title = r.get('title', '')
                    body = r.get('body', '')
                    body = re.sub(r'\s+', ' ', body).strip()
                    formatted.append(f"**{title}**\n{body[:500]}")
                return "\n\n---\n\n".join(formatted)
        except Exception as e:
            logger.error(f"DuckDuckGo error: {e}")
            return None
    
    def is_available(self) -> bool:
        return self.use_tavily or self.use_duckduckgo
    
    def get_engine_name(self) -> str:
        if self.use_tavily:
            return "Tavily"
        elif self.use_duckduckgo:
            return "DuckDuckGo"
        return "Недоступен"

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

web_search = WebSearch(TAVILY_API_KEY)
knowledge_base = ThermodynamicsKnowledgeBase(BOOKS_DIR)
knowledge_base.load()

# ============================================================================
# Функции ответов
# ============================================================================

def answer_from_pdf(question: str) -> Optional[str]:
    if not knowledge_base.vectorstore:
        return None
    
    context_chunks = knowledge_base.get_relevant_chunks(question, k=K_RETRIEVAL)
    if not context_chunks:
        return None
    
    context = "\n\n---\n\n".join(context_chunks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"Используй информацию из PDF:\n{context}"),
        HumanMessage(content=question),
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return None

def answer_from_web(question: str) -> Optional[str]:
    if not web_search.is_available():
        return None
    
    search_results = web_search.search(question)
    if not search_results:
        return None
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"Результаты поиска:\n{search_results}"),
        HumanMessage(content=question),
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Web error: {e}")
        return None

def answer_direct(question: str) -> str:
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Ошибка: {e}"

def is_educational_query(question: str) -> bool:
    """Проверка на образовательный запрос"""
    educational_keywords = [
        "термодинамик", "тепломассообмен", "энтропи", "энтальпи",
        "формула", "расчет", "закон", "цикл", "кпд", "нуссельт",
        "лабораторн", "экзамен", "помоги", "объясни", "расскажи",
        "thermodynamics", "heat transfer", "entropy", "enthalpy",
        "nusselt", "reynolds", "prandtl", "fourier"
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in educational_keywords)

def get_answer(question: str, session_id: str = None) -> Tuple[str, str]:
    """Получает ответ с указанием источника (без стриминга)"""
    
    # Проверка на образовательный запрос
    if not is_educational_query(question) and len(question) > 15:
        return """📚 Я специализируюсь на вопросах по **технической термодинамике (ТТД)** и **тепломассообмену (ТМО)**.

**Примеры вопросов:**
• Объясни первый закон термодинамики
• Что такое число Нуссельта?
• Как рассчитать КПД цикла Карно?
• Напиши формулу теплопроводности

Задайте конкретный вопрос по этим темам!""", "🎓 Совет"
    
    print(f"  🔍 Поиск в PDF: {question[:50]}...")
    answer = answer_from_pdf(question)
    if answer:
        print("  ✅ Найдено в PDF")
        return answer, "📚 PDF"
    
    if web_search.is_available():
        print(f"  🌐 Поиск в интернете ({web_search.get_engine_name()})...")
        answer = answer_from_web(question)
        if answer:
            print(f"  ✅ Найдено через {web_search.get_engine_name()}")
            return answer, f"🌐 {web_search.get_engine_name()}"
    
    print("  🤖 Использование LLM...")
    answer = answer_direct(question)
    return answer, "🎓 LLM"


# ============================================================================
# СТРИМИНГ (постепенная печать)
# ============================================================================

async def get_answer_stream(question: str, session_id: str = None) -> AsyncGenerator[str, None]:
    """
    Стриминг ответа с постепенной печатью.
    Отправляет JSON строки с полями: chunk, source, done
    """
    
    # Проверка на образовательный запрос
    if not is_educational_query(question) and len(question) > 15:
        yield json.dumps({
            "chunk": "📚 Я специализируюсь на вопросах по **технической термодинамике (ТТД)** и **тепломассообмену (ТМО)**.\n\n**Примеры вопросов:**\n• Объясни первый закон термодинамики\n• Что такое число Нуссельта?\n• Как рассчитать КПД цикла Карно?\n• Напиши формулу теплопроводности\n\nЗадайте конкретный вопрос по этим темам!",
            "source": "🎓 Совет",
            "done": True
        }) + "\n"
        return
    
    # Пробуем получить ответ из PDF
    print(f"  🔍 Поиск в PDF: {question[:50]}...")
    answer = answer_from_pdf(question)
    source = "📚 PDF"
    
    if not answer and web_search.is_available():
        print(f"  🌐 Поиск в интернете ({web_search.get_engine_name()})...")
        answer = answer_from_web(question)
        source = f"🌐 {web_search.get_engine_name()}"
    
    if not answer:
        print("  🤖 Использование LLM...")
        answer = answer_direct(question)
        source = "🎓 LLM"
    
    # Разбиваем ответ на части для постепенной печати
    # По словам или по символам для плавного эффекта
    words = answer.split()
    total_words = len(words)
    
    # Отправляем источник первым чанком
    yield json.dumps({
        "chunk": "",
        "source": source,
        "done": False
    }) + "\n"
    
    # Стримим по 2-3 слова за раз
    for i in range(0, total_words, 3):
        chunk = ' '.join(words[i:i+3])
        if chunk:
            yield json.dumps({
                "chunk": chunk + " ",
                "source": None,
                "done": False
            }) + "\n"
        await asyncio.sleep(0.03)  # Небольшая задержка для эффекта печати
    
    # Финальный сигнал
    yield json.dumps({
        "chunk": "",
        "source": source,
        "done": True
    }) + "\n"
    
    print(f"  ✅ Стриминг завершен ({total_words} слов)")


# ============================================================================
# Запуск
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔥 ИИ-преподаватель по ТТД и ТМО")
    print("="*60)
    
    stats = knowledge_base.get_stats()
    print(f"📚 RAG: {'✅' if stats.get('loaded') else '❌'} ({stats.get('vectors', 0)} векторов)")
    print(f"🌐 Поиск: {web_search.get_engine_name()}")
    print(f"🤖 Модель: {OLLAMA_MODEL}")
    print(f"✨ Стриминг: включен (постепенная печать)")
    print("="*60)
    print("Введите вопрос или 'quit' для выхода\n")
    
    async def async_main():
        while True:
            try:
                question = input("❓ Вопрос: ").strip()
                if question.lower() in ['quit', 'exit']:
                    break
                if not question:
                    continue
                
                print("  ⏳ Думаю...")
                start = time.time()
                
                # Для консоли используем обычный режим (без стриминга)
                answer, source = get_answer(question)
                elapsed = time.time() - start
                
                print(f"\n{source} ({elapsed:.1f}с):")
                print(answer)
                print("-"*60)
            except KeyboardInterrupt:
                break
    
    asyncio.run(async_main())
    print("\n👋 До свидания!")