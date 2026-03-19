import os
import logging
import warnings
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import time
import json
import requests

# Подавление предупреждений
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", message=".*unauthenticated.*", module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Отключаем логирование библиотек
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Импорты
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

class RAGConfig:
    """Конфигурация RAG-системы"""
    
    # Пути
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME = "knowledge_base"
    BOOKS_FOLDER = os.getenv("BOOKS_FOLDER", "./books")
    CACHE_FOLDER = "./cache"
    
    # Модели
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # OpenRouter API - используем актуальные бесплатные модели
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Список бесплатных моделей на 2026 год
    FREE_MODELS = [
        "google/gemma-3-12b-it:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-3-4b-it:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "openrouter/free"
    ]
    
    # Модель по умолчанию (первая из списка)
    OPENROUTER_MODEL = FREE_MODELS[0]
    
    # Tavily API (опционально)
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    TAVILY_SEARCH_DEPTH = "advanced"
    
    # Linear API (опционально)
    LINEAR_API_KEY = os.getenv("LINEAR_API_KEY", "")
    LINEAR_TEAM_ID = os.getenv("LINEAR_TEAM_ID", "")
    
    # Чанкование
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    
    # Поиск
    DEFAULT_K_RESULTS = 5
    MIN_SIMILARITY_SCORE = 0.3
    
    # RAG настройки
    RAG_TEMPERATURE = 0.7
    RAG_MAX_TOKENS = 2000
    
    # PDF загрузка
    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]
    USE_MULTITHREADING = True
    SHOW_PROGRESS = True

# Создаем экземпляр конфигурации
config = RAGConfig()

# Создаем необходимые директории
os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
os.makedirs(config.BOOKS_FOLDER, exist_ok=True)
os.makedirs(config.CACHE_FOLDER, exist_ok=True)

# =============================================================================
# OPENROUTER КЛИЕНТ (для RAG)
# =============================================================================

class OpenRouterRAG:
    """Клиент OpenRouter для RAG системы"""
    
    def __init__(self):
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = config.OPENROUTER_BASE_URL
        self.model = self._get_best_available_model()
        self.temperature = config.RAG_TEMPERATURE
        self.max_tokens = config.RAG_MAX_TOKENS
        
        # Системный промпт для преподавателя
        self.system_prompt = """Ты — преподаватель по технической термодинамике. Отвечай на вопросы студента, используя ТОЛЬКО информацию из предоставленного контекста.

ПРАВИЛА:
1. Если информация есть в контексте — дай точный ответ
2. Если информации нет в контексте — честно скажи "В предоставленных материалах нет информации по этому вопросу"
3. НЕ придумывай факты и не используй свои знания вне контекста
4. Отвечай кратко, четко и по делу
5. Если вопрос не по термодинамике, вежливо направь к теме

Контекст из учебных материалов:
{context}

Вопрос студента: {question}

Твой ответ (только на основе контекста):"""
    
    def _get_best_available_model(self) -> str:
        """Получает лучшую доступную бесплатную модель"""
        if not self.is_available():
            return config.OPENROUTER_MODEL
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            models = response.json().get("data", [])
            available_free = []
            
            for model in models:
                model_id = model.get("id", "")
                pricing = model.get("pricing", {})
                
                # Проверяем, бесплатная ли модель
                if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
                    available_free.append(model_id)
            
            # Ищем модель из нашего списка
            for preferred in config.FREE_MODELS:
                for available in available_free:
                    if preferred in available or available in preferred:
                        logger.info(f"✅ Выбрана модель: {available}")
                        return available
            
            # Если ничего не нашли, берем первую доступную бесплатную
            if available_free:
                logger.info(f"✅ Выбрана модель: {available_free[0]}")
                return available_free[0]
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения списка моделей: {e}")
        
        # Возвращаем модель по умолчанию
        logger.info(f"ℹ️ Использую модель по умолчанию: {config.OPENROUTER_MODEL}")
        return config.OPENROUTER_MODEL
    
    def is_available(self) -> bool:
        """Проверяет доступность API"""
        return bool(self.api_key) and self.api_key != ""
    
    def generate_answer(self, question: str, context: str) -> Optional[str]:
        """Генерирует ответ на основе контекста"""
        
        if not self.is_available():
            logger.error("❌ OpenRouter API ключ не настроен")
            return None
        
        # Формируем промпт
        prompt = self.system_prompt.format(
            context=context,
            question=question
        )
        
        # Пробуем разные модели если первая не сработает
        models_to_try = [self.model] + config.FREE_MODELS
        
        for attempt, model in enumerate(models_to_try[:3]):  # Пробуем максимум 3 модели
            try:
                logger.info(f"🤖 Пробую модель: {model}")
                
                # Создаем LLM через LangChain
                llm = ChatOpenAI(
                    openai_api_key=self.api_key,
                    openai_api_base=self.base_url,
                    model_name=model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    request_timeout=60
                )
                
                # Отправляем запрос
                messages = [HumanMessage(content=prompt)]
                response = llm.invoke(messages)
                
                # Если успешно, запоминаем эту модель для будущих запросов
                if attempt > 0:
                    self.model = model
                    logger.info(f"✅ Модель обновлена на: {model}")
                
                return response.content
                
            except Exception as e:
                logger.warning(f"⚠️ Модель {model} не сработала: {e}")
                if attempt < len(models_to_try[:3]) - 1:
                    continue
                else:
                    logger.error("❌ Все модели не сработали")
                    return None
        
        return None
    
    def get_free_models(self) -> List[Dict]:
        """Получает список бесплатных моделей"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            models = response.json().get("data", [])
            free_models = []
            
            for model in models:
                pricing = model.get("pricing", {})
                if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
                    free_models.append({
                        "id": model.get("id"),
                        "name": model.get("name"),
                        "context_length": model.get("context_length")
                    })
            
            return free_models
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения моделей: {e}")
            return []

# Инициализируем OpenRouter для RAG
rag_llm = OpenRouterRAG()

# =============================================================================
# ЗАГРУЗКА PDF ИЗ ПАПКИ BOOKS
# =============================================================================

def get_pdf_files() -> List[Path]:
    """Возвращает список всех PDF файлов в папке books"""
    books_path = Path(config.BOOKS_FOLDER)
    
    if not books_path.exists():
        logger.warning(f"⚠️ Папка {config.BOOKS_FOLDER} не существует")
        return []
    
    # Ищем все поддерживаемые файлы
    all_files = []
    for ext in config.SUPPORTED_EXTENSIONS:
        all_files.extend(books_path.glob(f"**/*{ext}"))
    
    if all_files:
        logger.info(f"📚 Найдено файлов: {len(all_files)}")
        for f in sorted(all_files)[:5]:
            logger.info(f"   • {f.relative_to(books_path)}")
        if len(all_files) > 5:
            logger.info(f"   • ... и ещё {len(all_files) - 5}")
    else:
        logger.warning(f"⚠️ В папке {config.BOOKS_FOLDER} нет поддерживаемых файлов")
        logger.info(f"📝 Поддерживаемые форматы: {', '.join(config.SUPPORTED_EXTENSIONS)}")
    
    return all_files

def load_single_pdf(file_path: Path) -> List[Document]:
    """Загружает один PDF файл"""
    try:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        
        # Добавляем метаданные
        for doc in documents:
            doc.metadata.update({
                "source": str(file_path),
                "filename": file_path.name,
                "folder": str(file_path.parent),
                "type": "pdf",
                "loaded_at": datetime.now().isoformat()
            })
        
        return documents
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки {file_path.name}: {e}")
        return []

def load_all_documents() -> List[Document]:
    """Загружает все документы из папки books"""
    all_documents = []
    files = get_pdf_files()
    
    if not files:
        # Создаем пример для демонстрации
        logger.info("📝 Создаю пример документа для демонстрации")
        sample_text = """
        ТЕХНИЧЕСКАЯ ТЕРМОДИНАМИКА
        Основные понятия и определения
        
        Рабочее тело - вещество, с помощью которого осуществляется термодинамический процесс.
        Основные параметры состояния: давление P [Па], температура T [K], удельный объем v [м³/кг].
        
        Уравнение состояния идеального газа: P·v = R·T, где R - газовая постоянная.
        
        Первый закон термодинамики: Q = ΔU + L, где Q - теплота, ΔU - изменение внутренней энергии, L - работа.
        
        Энтропия - функция состояния, характеризующая меру неупорядоченности системы.
        Второй закон термодинамики: энтропия изолированной системы не убывает.
        
        Цикл Карно - идеальный цикл, состоящий из двух изотерм и двух адиабат.
        КПД цикла Карно: η = 1 - T₂/T₁.
        """
        
        doc = Document(
            page_content=sample_text,
            metadata={
                "source": "sample.txt",
                "filename": "sample.txt",
                "type": "sample"
            }
        )
        return [doc]
    
    # Загружаем все файлы с прогресс-баром
    for file_path in tqdm(files, desc="📚 Загрузка документов", disable=not config.SHOW_PROGRESS):
        documents = load_single_pdf(file_path)
        all_documents.extend(documents)
    
    logger.info(f"✅ Всего загружено страниц: {len(all_documents)}")
    return all_documents

# =============================================================================
# ЧАНКОВАНИЕ ДОКУМЕНТОВ
# =============================================================================

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Разбивает документы на чанки"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=config.CHUNK_SEPARATORS
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Добавляем индексы чанков
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    logger.info(f"🔪 Создано {len(chunks)} чанков из {len(documents)} документов")
    
    # Статистика
    if chunks:
        lengths = [len(c.page_content) for c in chunks]
        logger.info(f"   • Средний размер: {np.mean(lengths):.0f} символов")
        logger.info(f"   • Мин/Макс: {min(lengths)} / {max(lengths)}")
    
    return chunks

# =============================================================================
# РАБОТА С CHROMADB
# =============================================================================

def get_embedding_function():
    """Возвращает функцию эмбеддингов"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )

def create_or_get_collection():
    """Создает или получает существующую коллекцию ChromaDB"""
    
    client = chromadb.PersistentClient(
        path=config.CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    embedding_func = get_embedding_function()
    
    # Проверяем существование коллекции
    existing_collections = [col.name for col in client.list_collections()]
    
    if config.COLLECTION_NAME in existing_collections:
        logger.info(f"📚 Загружаем существующую коллекцию '{config.COLLECTION_NAME}'")
        collection = client.get_collection(name=config.COLLECTION_NAME)
        return collection, False
    else:
        logger.info(f"📚 Создаем новую коллекцию '{config.COLLECTION_NAME}'")
        collection = client.create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embedding_func,
            metadata={"created_at": datetime.now().isoformat()}
        )
        return collection, True

# Инициализация коллекции
collection, is_new = create_or_get_collection()

def index_documents(documents: List[Document], force_reindex: bool = False):
    """Индексирует документы в ChromaDB"""
    
    if not documents:
        logger.warning("⚠️ Нет документов для индексации")
        return False
    
    if not is_new and not force_reindex:
        logger.info("🟢 Коллекция уже существует. Используйте force_reindex=True для переиндексации")
        return True
    
    chunks = chunk_documents(documents)
    
    # Подготовка данных для ChromaDB
    ids = [hashlib.md5(f"{chunk.metadata.get('filename', 'unknown')}_{i}".encode()).hexdigest()[:16] 
           for i, chunk in enumerate(chunks)]
    
    documents_text = [chunk.page_content for chunk in chunks]
    
    # Метаданные
    metadatas = []
    for chunk in chunks:
        meta = chunk.metadata.copy()
        meta = {k: v for k, v in meta.items() if v is not None}
        metadatas.append(meta)
    
    # Добавляем в коллекцию батчами
    logger.info(f"📤 Добавляю {len(chunks)} чанков в ChromaDB...")
    
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Индексация", disable=not config.SHOW_PROGRESS):
        end_idx = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end_idx],
            documents=documents_text[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
    
    logger.info(f"✅ Индексация завершена! Всего чанков: {collection.count()}")
    return True

# =============================================================================
# ПОИСК И ИЗВЛЕЧЕНИЕ
# =============================================================================

def get_relevant_chunks(query: str, k: int = None, min_score: float = None) -> List[str]:
    """Возвращает k наиболее релевантных фрагментов из базы знаний по запросу."""
    
    if k is None:
        k = config.DEFAULT_K_RESULTS
    
    if min_score is None:
        min_score = config.MIN_SIMILARITY_SCORE
    
    try:
        total_chunks = collection.count()
        if total_chunks == 0:
            logger.warning("⚠️ База знаний пуста")
            return []
        
        results = collection.query(
            query_texts=[query], 
            n_results=min(k * 2, total_chunks)
        )
        
        if not results or not results.get("documents") or not results["documents"][0]:
            return []
        
        documents = results["documents"][0]
        distances = results.get("distances", [[]])[0]
        
        # Фильтруем по минимальному сходству
        filtered_results = []
        for doc, dist in zip(documents, distances):
            similarity = 1 - (dist / 2)
            if similarity >= min_score:
                filtered_results.append(doc)
            if len(filtered_results) >= k:
                break
        
        return filtered_results[:k]
        
    except Exception as e:
        logger.error(f"❌ Ошибка при запросе к ChromaDB: {e}")
        return []

def get_relevant_chunks_with_scores(query: str, k: int = None) -> List[Tuple[str, float]]:
    """Возвращает чанки с оценками сходства"""
    
    if k is None:
        k = config.DEFAULT_K_RESULTS
    
    try:
        total_chunks = collection.count()
        if total_chunks == 0:
            return []
        
        results = collection.query(
            query_texts=[query], 
            n_results=min(k, total_chunks),
            include=["documents", "distances", "metadatas"]
        )
        
        if not results or not results.get("documents") or not results["documents"][0]:
            return []
        
        documents = results["documents"][0]
        distances = results.get("distances", [[]])[0]
        
        result_with_scores = []
        for doc, dist in zip(documents, distances):
            similarity = 1 - (dist / 2)
            result_with_scores.append((doc, similarity))
        
        return result_with_scores
        
    except Exception as e:
        logger.error(f"❌ Ошибка при запросе к ChromaDB: {e}")
        return []

# =============================================================================
# RAG ФУНКЦИЯ (использует OpenRouter)
# =============================================================================

def ask_rag(question: str, k: int = None) -> Dict[str, Any]:
    """
    Основная функция RAG: ищет релевантные чанки и генерирует ответ через OpenRouter
    
    Args:
        question: Вопрос пользователя
        k: Количество чанков для контекста
    
    Returns:
        Dict с ответом и метаданными
    """
    
    if k is None:
        k = config.DEFAULT_K_RESULTS
    
    # Проверяем доступность OpenRouter
    if not rag_llm.is_available():
        return {
            "success": False,
            "error": "OpenRouter API не настроен",
            "answer": None,
            "chunks": []
        }
    
    # Получаем релевантные чанки
    chunks = get_relevant_chunks(question, k=k)
    
    if not chunks:
        return {
            "success": True,
            "answer": "В предоставленных учебных материалах нет информации по этому вопросу.",
            "chunks": [],
            "sources": []
        }
    
    # Формируем контекст из чанков
    context = "\n\n---\n\n".join(chunks)
    
    # Получаем ответ от OpenRouter
    answer = rag_llm.generate_answer(question, context)
    
    if not answer:
        return {
            "success": False,
            "error": "Не удалось получить ответ от OpenRouter",
            "answer": None,
            "chunks": chunks
        }
    
    # Извлекаем источники из метаданных
    sources = []
    try:
        # Получаем метаданные для найденных чанков
        results = collection.query(
            query_texts=[question],
            n_results=k,
            include=["metadatas"]
        )
        if results and results.get("metadatas") and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                if meta and "filename" in meta:
                    sources.append(meta["filename"])
    except:
        pass
    
    return {
        "success": True,
        "answer": answer,
        "chunks": chunks,
        "sources": list(set(sources)) if sources else []
    }

# =============================================================================
# API КЛИЕНТЫ (для бота)
# =============================================================================

class TavilyClient:
    """Клиент для работы с Tavily Search API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.TAVILY_API_KEY
        self.search_depth = config.TAVILY_SEARCH_DEPTH
        self.base_url = "https://api.tavily.com"
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
    
    def is_available(self) -> bool:
        """Проверяет доступность API"""
        return bool(self.api_key) and self.api_key != ""
    
    def search(self, query: str, max_results: int = 5) -> Optional[List[Dict]]:
        """Выполняет поиск"""
        
        if not self.is_available():
            return None
        
        try:
            response = self.session.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "search_depth": self.search_depth,
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False
                },
                timeout=15
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("results", [])
            
        except Exception as e:
            logger.error(f"❌ Tavily API error: {e}")
            return None


class LinearClient:
    """Клиент для работы с Linear API"""
    
    def __init__(self, api_key: str = None, team_id: str = None):
        self.api_key = api_key or config.LINEAR_API_KEY
        self.team_id = team_id or config.LINEAR_TEAM_ID
        self.base_url = "https://api.linear.app/graphql"
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": self.api_key,
                "Content-Type": "application/json"
            })
    
    def is_available(self) -> bool:
        """Проверяет доступность API"""
        return bool(self.api_key) and self.api_key != ""
    
    def execute_query(self, query: str, variables: Dict = None) -> Optional[Dict]:
        """Выполняет GraphQL запрос"""
        
        if not self.is_available():
            return None
        
        try:
            response = self.session.post(
                self.base_url,
                json={
                    "query": query,
                    "variables": variables or {}
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Linear API error: {e}")
            return None
    
    def create_issue(self, title: str, description: str = "", 
                     priority: int = 1) -> Optional[Dict]:
        """Создает задачу в Linear"""
        
        if not self.team_id:
            logger.error("❌ TEAM_ID не настроен")
            return None
        
        query = """
        mutation IssueCreate($title: String!, $description: String!, $teamId: String!, $priority: Int) {
          issueCreate(
            input: {
              title: $title,
              description: $description,
              teamId: $teamId,
              priority: $priority
            }
          ) {
            success
            issue {
              id
              title
              url
              identifier
            }
          }
        }
        """
        
        variables = {
            "title": title,
            "description": description,
            "teamId": self.team_id,
            "priority": priority
        }
        
        result = self.execute_query(query, variables)
        if result and "data" in result:
            return result["data"]["issueCreate"]
        return None
    
    def get_issues(self, limit: int = 10) -> Optional[List[Dict]]:
        """Получает список задач"""
        
        query = """
        query Issues($teamId: String!, $limit: Int) {
          team(id: $teamId) {
            issues(first: $limit) {
              nodes {
                id
                title
                description
                url
                identifier
                priority
                state {
                  name
                }
              }
            }
          }
        }
        """
        
        variables = {
            "teamId": self.team_id,
            "limit": limit
        }
        
        result = self.execute_query(query, variables)
        if result and "data" in result:
            return result["data"]["team"]["issues"]["nodes"]
        return None

# Инициализация клиентов для бота
tavily = TavilyClient()
linear = LinearClient()

# =============================================================================
# СТАТИСТИКА И УПРАВЛЕНИЕ
# =============================================================================

def get_collection_stats() -> Dict:
    """Возвращает статистику коллекции"""
    try:
        count = collection.count()
        
        # Получаем информацию о метаданных
        sources = {}
        if count > 0:
            meta_results = collection.get(limit=min(100, count))
            metadatas = meta_results.get('metadatas', [])
            
            for meta in metadatas:
                if meta and 'filename' in meta:
                    filename = meta['filename']
                    sources[filename] = sources.get(filename, 0) + 1
        
        return {
            "total_chunks": count,
            "collection_name": config.COLLECTION_NAME,
            "db_path": config.CHROMA_DB_PATH,
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "sources": sources,
            "has_data": count > 0,
            "rag_available": rag_llm.is_available(),
            "rag_model": rag_llm.model if rag_llm.is_available() else None
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}")
        return {"error": str(e)}

def get_api_status() -> Dict:
    """Возвращает статус всех API"""
    return {
        "openrouter": {
            "available": rag_llm.is_available(),
            "model": rag_llm.model if rag_llm.is_available() else None
        },
        "tavily": {
            "available": tavily.is_available()
        },
        "linear": {
            "available": linear.is_available(),
            "team_id": linear.team_id if linear.is_available() else None
        }
    }

def reset_collection():
    """Сбрасывает коллекцию"""
    global collection, is_new
    
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        client.delete_collection(config.COLLECTION_NAME)
        collection, is_new = create_or_get_collection()
        logger.info("🔄 Коллекция сброшена")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка сброса коллекции: {e}")
        return False

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ
# =============================================================================

# Автоматическая индексация при первом запуске
if is_new:
    logger.info("🆕 Новая коллекция, запускаю индексацию...")
    documents = load_all_documents()
    if documents:
        index_documents(documents)
    else:
        logger.warning("⚠️ Нет документов для индексации")

# =============================================================================
# ОСНОВНОЙ БЛОК
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("📚 RAG СИСТЕМА С OPENROUTER")
    print("="*70)
    
    # Статус API
    api_status = get_api_status()
    print("\n🔌 СТАТУС API:")
    for api_name, status in api_status.items():
        if status.get("available"):
            print(f"   ✅ {api_name}: доступен")
            if api_name == "openrouter" and status.get("model"):
                print(f"      • Модель: {status['model']}")
        else:
            print(f"   ❌ {api_name}: не настроен")
    
    # Статистика базы знаний
    stats = get_collection_stats()
    print(f"\n📊 СТАТИСТИКА БАЗЫ ЗНАНИЙ")
    print(f"   Всего чанков: {stats.get('total_chunks', 0)}")
    print(f"   Модель эмбеддингов: {stats.get('embedding_model')}")
    
    if 'sources' in stats and stats['sources']:
        print(f"\n📚 ИСТОЧНИКИ:")
        for src, count in sorted(stats['sources'].items())[:5]:
            print(f"   • {src}: {count} чанков")
        if len(stats['sources']) > 5:
            print(f"   • ... и ещё {len(stats['sources']) - 5} источников")
    
    # Тестовый RAG запрос
    if stats.get('total_chunks', 0) > 0 and rag_llm.is_available():
        print(f"\n🔍 ТЕСТОВЫЙ RAG ЗАПРОС")
        print("-" * 50)
        
        test_queries = [
            "первый закон термодинамики",
            "что такое энтропия",
            "цикл карно"
        ]
        
        for query in test_queries:
            print(f"\n📝 Запрос: '{query}'")
            result = ask_rag(query, k=2)
            
            if result["success"]:
                if result["answer"]:
                    print(f"💬 Ответ: {result['answer'][:200]}...")
                else:
                    print(f"❌ Нет ответа")
                if result.get("sources"):
                    print(f"📚 Источники: {', '.join(result['sources'])}")
            else:
                print(f"❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")
    
    print("\n✅ RAG система готова к работе!")
    print("📁 PDF файлы должны находиться в папке: ./books/")
    print(f"🤖 RAG использует модель: {rag_llm.model}")