"""
RAG модуль для работы с PDF-документами по термодинамике
"""

import os
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Конфигурация
BOOKS_DIR = Path("./books")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
K_RETRIEVAL = 5

# Доступные модели эмбеддингов
EMBEDDING_MODELS = {
    "e5-base": "intfloat/multilingual-e5-base",           # Рекомендуемая
    "bge-m3": "BAAI/bge-m3",                               # Максимальное качество
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Быстрая
    "rubert": "ai-forever/sbert_large_mt_nlu_ru",         # Специально для русского
    "distiluse": "distiluse-base-multilingual-cased-v2",   # Легкая
}

# Выбор модели (можно изменить через переменную окружения)
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "e5-base")
EMBEDDING_MODEL = EMBEDDING_MODELS.get(DEFAULT_MODEL, EMBEDDING_MODELS["e5-base"])


class ThermodynamicsKnowledgeBase:
    """Управление базой знаний из PDF-документов по термодинамике."""
    
    def __init__(self, books_dir: Path = BOOKS_DIR):
        self.books_dir = Path(books_dir)
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self._embeddings = None
        self._loaded = False
        self._stats = {"total_pages": 0, "total_chunks": 0, "files": []}
    
    def load(self, force_reload: bool = False) -> bool:
        """Загружает PDF-файлы и создает векторное хранилище."""
        if self._loaded and not force_reload:
            logger.info("База знаний уже загружена")
            return True
        
        if not self.books_dir.exists():
            logger.warning(f"Папка {self.books_dir} не найдена")
            print(f"⚠️ Папка {self.books_dir} не найдена. Создайте её и добавьте PDF-файлы.")
            return False
        
        pdf_files = list(self.books_dir.glob("**/*.pdf"))
        if not pdf_files:
            logger.warning(f"PDF-файлы не найдены в {self.books_dir}")
            print(f"⚠️ PDF-файлы не найдены в {self.books_dir}")
            return False
        
        print(f"\n📚 Загрузка PDF-документов из {self.books_dir}...")
        print(f"   Найдено файлов: {len(pdf_files)}")
        print(f"   Модель эмбеддингов: {EMBEDDING_MODEL}")
        
        # Загрузка всех PDF
        all_docs = []
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                all_docs.extend(pages)
                self._stats["total_pages"] += len(pages)
                self._stats["files"].append({"name": pdf_path.name, "pages": len(pages)})
                print(f"   • {pdf_path.name}: {len(pages)} стр.")
            except Exception as e:
                logger.error(f"Ошибка загрузки {pdf_path.name}: {e}")
        
        if not all_docs:
            logger.warning("Не удалось загрузить ни одного PDF-документа")
            return False
        
        print(f"✅ Загружено страниц: {self._stats['total_pages']}")
        
        # Разбивка на чанки
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(all_docs)
        self._stats["total_chunks"] = len(chunks)
        print(f"🔪 Создано чанков: {len(chunks)}")
        
        # Создание эмбеддингов
        print(f"⏳ Загрузка модели эмбеддингов: {EMBEDDING_MODEL}")
        print("   Это может занять несколько минут при первом запуске...")
        
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Создание векторного хранилища
        print("⏳ Создание векторного хранилища...")
        self.vectorstore = FAISS.from_documents(chunks, self._embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVAL})
        
        print(f"✅ Векторное хранилище создано: {self.vectorstore.index.ntotal} векторов")
        print(f"   Размерность эмбеддингов: {self.vectorstore.index.d}")
        
        self._loaded = True
        return True
    
    def create_rag_chain(self, llm: ChatOpenAI, system_prompt: str) -> Optional[Any]:
        """Создает RAG цепочку для ответов на вопросы."""
        if not self._loaded or self.retriever is None:
            logger.error("База знаний не загружена")
            return None
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("system", "\n\n--- МАТЕРИАЛ ИЗ PDF-ДОКУМЕНТОВ (ИСПОЛЬЗУЙ В ПЕРВУЮ ОЧЕРЕДЬ) ---\n{context}"),
            ("human", "{input}"),
        ])
        
        combine_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, combine_chain)
        
        return self.rag_chain
    
    def get_relevant_chunks(self, query: str, k: int = None) -> List[str]:
        """Возвращает релевантные фрагменты из базы знаний."""
        if not self._loaded or self.vectorstore is None:
            return []
        
        k = k or K_RETRIEVAL
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Ошибка поиска в векторном хранилище: {e}")
            return []
    
    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        Поиск с возвратом метаданных.
        
        Args:
            query: Поисковый запрос
            k: Количество фрагментов
            
        Returns:
            Список словарей с текстом и метаданными
        """
        if not self._loaded or self.vectorstore is None:
            return []
        
        k = k or K_RETRIEVAL
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": Path(doc.metadata.get("source", "unknown")).name,
                }
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Возвращает статистику базы знаний."""
        if not self._loaded or self.vectorstore is None:
            return {"loaded": False, "vectors": 0}
        
        return {
            "loaded": self._loaded,
            "vectors": self.vectorstore.index.ntotal,
            "embedding_dim": self.vectorstore.index.d,
            "embedding_model": EMBEDDING_MODEL,
            "total_pages": self._stats["total_pages"],
            "total_chunks": self._stats["total_chunks"],
            "files": self._stats["files"],
        }


def load_knowledge_base(books_dir: Path = BOOKS_DIR) -> Optional[ThermodynamicsKnowledgeBase]:
    """Быстрая загрузка базы знаний."""
    kb = ThermodynamicsKnowledgeBase(books_dir)
    if kb.load():
        return kb
    return None


# Для обратной совместимости
def get_relevant_chunks(query: str, vectorstore=None, k: int = K_RETRIEVAL) -> List[str]:
    """
    Совместимая функция для получения релевантных фрагментов.
    
    Args:
        query: Поисковый запрос
        vectorstore: Векторное хранилище (опционально)
        k: Количество фрагментов
        
    Returns:
        Список текстовых фрагментов
    """
    if vectorstore is None:
        return []
    
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        return []