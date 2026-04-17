"""
RAG модуль для работы с PDF-документами
"""

import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

BOOKS_DIR = Path("./books")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
K_RETRIEVAL = 5
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


class ThermodynamicsKnowledgeBase:
    def __init__(self, books_dir: Path = BOOKS_DIR):
        self.books_dir = Path(books_dir)
        self.vectorstore = None
        self._loaded = False
        self._stats = {"vectors": 0, "files": []}
    
    def load(self, force_reload: bool = False) -> bool:
        if self._loaded and not force_reload:
            return True
        
        if not self.books_dir.exists():
            print(f"⚠️ Папка {self.books_dir} не найдена")
            return False
        
        pdf_files = list(self.books_dir.glob("**/*.pdf"))
        txt_files = list(self.books_dir.glob("**/*.txt"))
        all_files = pdf_files + txt_files
        
        if not all_files:
            print(f"⚠️ Документы не найдены в {self.books_dir}")
            return False
        
        print(f"\n📚 Загрузка документов...")
        print(f"   Найдено: {len(all_files)} файлов")
        print(f"   Модель: {EMBEDDING_MODEL}")
        
        all_docs = []
        for file_path in all_files:
            try:
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    pages = loader.load()
                    all_docs.extend(pages)
                    print(f"   • {file_path.name}: {len(pages)} стр.")
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    doc = Document(page_content=content, metadata={"source": str(file_path)})
                    all_docs.append(doc)
                    print(f"   • {file_path.name}: текст")
                self._stats["files"].append(file_path.name)
            except Exception as e:
                print(f"   ❌ {file_path.name}: {e}")
        
        if not all_docs:
            return False
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(all_docs)
        print(f"🔪 Создано чанков: {len(chunks)}")
        
        print(f"⏳ Загрузка модели эмбеддингов...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        print("⏳ Создание векторного хранилища...")
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        self._stats["vectors"] = self.vectorstore.index.ntotal
        
        print(f"✅ Готово! {self._stats['vectors']} векторов")
        self._loaded = True
        return True
    
    def get_relevant_chunks(self, query: str, k: int = None) -> List[str]:
        if not self._loaded or self.vectorstore is None:
            return []
        
        k = k or K_RETRIEVAL
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_stats(self) -> Dict:
        return {
            "loaded": self._loaded,
            "vectors": self._stats["vectors"],
            "files": self._stats["files"],
        }