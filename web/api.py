"""
FastAPI-сервер для веб-интерфейса термодинамического бота.
"""

from __future__ import annotations

import logging
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# Добавляем корень проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Пытаемся импортировать модули бота
# ---------------------------------------------------------------------------

# Переменные с значениями по умолчанию
get_answer = None
tavily = None
knowledge_base = None
LANGFUSE_ENABLED = False
langfuse = None
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_BASE = "http://localhost:11434/v1"

# Пробуем импортировать из bot-local.py
bot_local_path = ROOT_DIR / "bot-local.py"

if bot_local_path.exists():
    try:
        with open(bot_local_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        namespace = {}
        exec(code, namespace)
        
        get_answer = namespace.get('get_answer')
        tavily = namespace.get('tavily')
        knowledge_base = namespace.get('knowledge_base')
        LANGFUSE_ENABLED = namespace.get('LANGFUSE_ENABLED', False)
        langfuse = namespace.get('langfuse')
        OLLAMA_MODEL = namespace.get('OLLAMA_MODEL', 'qwen3:4b')
        OLLAMA_BASE = namespace.get('OLLAMA_BASE', 'http://localhost:11434/v1')
        
        logger.info("✅ Успешно загружен bot-local.py")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки bot-local.py: {e}")
else:
    logger.warning(f"⚠️ Файл bot-local.py не найден в {ROOT_DIR}")

if get_answer is None:
    logger.warning("⚠️ Используется заглушка для get_answer")
    def get_answer(question, session_id=None):
        return f"🤖 Тестовый ответ на: {question}", "⚠️ Тестовый режим"

if tavily is None:
    class TavilyStub:
        def is_available(self): return False
    tavily = TavilyStub()

# ---------------------------------------------------------------------------
# Pydantic-модели
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=100)

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    source: str = "🤖 LLM"
    processing_time: float = 0.0

class ClearRequest(BaseModel):
    session_id: str

# ---------------------------------------------------------------------------
# Хранилище сессий
# ---------------------------------------------------------------------------

class SessionStore:
    def __init__(self):
        self.sessions = {}
        self.history = {}
    
    def get_or_create_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": datetime.now(),
                "message_count": 0,
                "last_active": datetime.now()
            }
            self.history[session_id] = []
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str, source: str = None):
        if session_id not in self.history:
            self.history[session_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "source": source if role == "assistant" else None
        }
        self.history[session_id].append(message)
        
        if session_id in self.sessions:
            self.sessions[session_id]["message_count"] += 1
            self.sessions[session_id]["last_active"] = datetime.now()
    
    def get_history(self, session_id: str, limit: int = 50):
        if session_id not in self.history:
            return []
        return self.history[session_id][-limit:]
    
    def clear_history(self, session_id: str):
        if session_id in self.history:
            self.history[session_id] = []
        if session_id in self.sessions:
            self.sessions[session_id]["message_count"] = 0

session_store = SessionStore()

# ---------------------------------------------------------------------------
# FastAPI приложение
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск веб-сервера")
    logger.info(f"🤖 Модель: {OLLAMA_MODEL}")
    
    try:
        if hasattr(tavily, 'is_available'):
            logger.info(f"🌐 Веб-поиск: {'доступен' if tavily.is_available() else 'недоступен'}")
    except:
        pass
    
    yield
    logger.info("👋 Остановка сервера")

app = FastAPI(title="Термодинамический Консультант", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/api/stats")
def get_stats():
    web_available = False
    if hasattr(tavily, 'is_available'):
        try:
            web_available = tavily.is_available()
        except:
            pass
    
    kb_loaded = False
    if knowledge_base and hasattr(knowledge_base, '_loaded'):
        kb_loaded = knowledge_base._loaded
    
    return {
        "knowledge_base_loaded": kb_loaded,
        "web_search_available": web_available,
        "langfuse_enabled": LANGFUSE_ENABLED,
        "ollama_model": OLLAMA_MODEL,
        "ollama_available": check_ollama(),
        "total_sessions": len(session_store.sessions)
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logger.info(f"📨 Запрос: {req.message[:50]}...")
    
    session_store.get_or_create_session(req.session_id)
    session_store.add_message(req.session_id, "user", req.message)
    
    start_time = time.time()
    
    try:
        result = get_answer(req.message, session_id=req.session_id)
        
        if isinstance(result, tuple):
            answer, source = result
        else:
            answer = result
            source = "🤖 Бот"
        
        # Обработка формул: заменяем \ на \\ для JSON
        # Но оставляем как есть для отображения
        
        processing_time = time.time() - start_time
        
        session_store.add_message(req.session_id, "assistant", answer, source)
        
        logger.info(f"✅ Ответ за {processing_time:.2f}с")
        
        return ChatResponse(
            reply=answer,
            session_id=req.session_id,
            source=source,
            processing_time=processing_time
        )
    except Exception as exc:
        logger.exception("❌ Ошибка")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/clear")
def clear_history(req: ClearRequest):
    logger.info(f"🧹 Очистка истории: {req.session_id}")
    session_store.clear_history(req.session_id)
    return {"status": "ok"}

@app.get("/api/history")
def get_history(session_id: str, limit: int = 50):
    return {
        "session_id": session_id,
        "messages": session_store.get_history(session_id, limit),
        "count": len(session_store.get_history(session_id, limit))
    }

def check_ollama() -> bool:
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE}/models", timeout=3)
        return response.status_code == 200
    except:
        return False

# ---------------------------------------------------------------------------
# HTML интерфейс с поддержкой формул
# ---------------------------------------------------------------------------

SIMPLE_HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Термодинамический Консультант</title>
    
    <!-- MathJax для рендеринга формул -->
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                packages: {'[+]': ['unicode']}
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            },
            startup: {
                pageReady: () => {
                    return MathJax.startup.defaultPageReady();
                }
            }
        };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50, #1a1a2e);
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .header h1 { font-size: 1.3rem; display: flex; align-items: center; gap: 10px; }
        .status { display: flex; gap: 10px; flex-wrap: wrap; }
        .badge {
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
        }
        .chat {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background: #f5f6fa;
        }
        .message {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { justify-content: flex-end; }
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
        }
        .message.user .message-avatar { background: #ff6b35; }
        .message.bot .message-avatar { background: #2c3e50; }
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            background: white;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .message.user .message-content {
            background: #ff6b35;
            color: white;
        }
        .message-text {
            line-height: 1.6;
            word-wrap: break-word;
        }
        .message-text p {
            margin-bottom: 10px;
        }
        .message-text p:last-child {
            margin-bottom: 0;
        }
        .message-text ul, .message-text ol {
            margin: 10px 0;
            padding-left: 25px;
        }
        .message-text li {
            margin: 5px 0;
        }
        .message-text pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 10px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .message-text code {
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .message.user .message-text code {
            background: rgba(255,255,255,0.2);
            color: white;
        }
        /* Стили для формул */
        .message-text mjx-container {
            margin: 12px 0;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 8px 0;
        }
        .message-source {
            font-size: 0.7rem;
            margin-top: 8px;
            opacity: 0.7;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e1e8ed;
            display: flex;
            gap: 10px;
        }
        textarea {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e1e8ed;
            border-radius: 25px;
            resize: none;
            font-family: inherit;
            font-size: 1rem;
            transition: all 0.3s;
        }
        textarea:focus { outline: none; border-color: #ff6b35; }
        button {
            background: #ff6b35;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s;
        }
        button:hover:not(:disabled) { background: #e55a2b; transform: translateY(-2px); }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .typing {
            color: #666;
            font-style: italic;
            padding: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .typing-dots {
            display: flex;
            gap: 4px;
        }
        .typing-dots span {
            width: 8px;
            height: 8px;
            background: #7f8c8d;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        .clear-btn {
            background: rgba(255,255,255,0.2);
            margin-left: 10px;
        }
        .clear-btn:hover { background: rgba(255,255,255,0.3); transform: none; }
        
        @media (max-width: 768px) {
            .message-content { max-width: 85%; }
            .header h1 { font-size: 1rem; }
            .badge span:last-child { display: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <span>🔥</span>
                <span>Термодинамический Консультант</span>
            </h1>
            <div class="status">
                <div class="badge" id="kb-status">📚 RAG</div>
                <div class="badge" id="web-status">🌐 Поиск</div>
                <div class="badge" id="model-status">🤖 Модель</div>
                <button class="clear-btn" onclick="clearChat()">🗑️ Очистить</button>
            </div>
        </div>
        <div class="chat" id="chat"></div>
        <div class="input-area">
            <textarea id="message" placeholder="Введите вопрос по термодинамике..." rows="2"></textarea>
            <button id="sendBtn" onclick="sendMessage()">📤 Отправить</button>
        </div>
    </div>

    <script>
        let sessionId = null;
        let isWaiting = false;
        
        // Функция для исправления LaTeX формул
        function fixLatex(content) {
            if (!content) return content;
            
            // Заменяем \ln на \ln (оставляем как есть)
            // Заменяем \frac на \frac (оставляем как есть)
            // Убираем лишние экранирования
            content = content.replace(/\\\\/g, '\\');
            
            return content;
        }
        
        // Функция для рендеринга формул
        async function renderFormulas(element) {
            if (window.MathJax) {
                try {
                    await MathJax.typesetPromise([element]);
                    console.log('Formulas rendered');
                } catch (err) {
                    console.log('MathJax error:', err);
                }
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Page loaded');
            
            sessionId = localStorage.getItem('session_id');
            if (!sessionId) {
                sessionId = crypto.randomUUID();
                localStorage.setItem('session_id', sessionId);
            }
            
            loadStats();
            setInterval(loadStats, 30000);
            
            setTimeout(async () => {
                await addMessage('bot', '👋 **Здравствуйте!** Я консультант по технической термодинамике.\n\nЗадайте мне вопрос!');
            }, 500);
            
            document.getElementById('message').focus();
        });
        
        async function sendMessage() {
            const input = document.getElementById('message');
            const message = input.value.trim();
            
            if (!message || isWaiting) return;
            
            input.value = '';
            input.style.height = 'auto';
            
            await addMessage('user', message);
            
            isWaiting = true;
            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;
            showTyping();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, session_id: sessionId })
                });
                
                if (!response.ok) {
                    throw new Error('Server error: ' + response.status);
                }
                
                const data = await response.json();
                hideTyping();
                
                let reply = data.reply;
                await addMessage('bot', reply, data.source);
                
            } catch (error) {
                console.error('Error:', error);
                hideTyping();
                await addMessage('bot', '❌ Error: ' + error.message);
            } finally {
                isWaiting = false;
                sendBtn.disabled = false;
                document.getElementById('message').focus();
            }
        }
        
        async function addMessage(role, content, source = null) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message ' + role;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = role === 'user' ? '👤' : '🤖';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            const textDiv = document.createElement('div');
            textDiv.className = 'message-text';
            
            // Обработка текста и формул
            let formattedContent = content || '';
            
            // Экранирование HTML
            formattedContent = formattedContent
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            
            // Markdown форматирование
            formattedContent = formattedContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            formattedContent = formattedContent.replace(/\*(.*?)\*/g, '<em>$1</em>');
            formattedContent = formattedContent.replace(/\n/g, '<br>');
            
            // Обработка списков
            formattedContent = formattedContent.replace(/^[•\-] (.*?)$/gm, '<li>$1</li>');
            formattedContent = formattedContent.replace(/(<li>.*?<\/li>)+/gs, function(match) {
                if (!match.includes('<ul>')) {
                    return '<ul>' + match + '</ul>';
                }
                return match;
            });
            
            textDiv.innerHTML = formattedContent;
            contentDiv.appendChild(textDiv);
            
            if (source && role === 'bot') {
                const sourceDiv = document.createElement('div');
                sourceDiv.className = 'message-source';
                sourceDiv.innerHTML = '📎 ' + source;
                contentDiv.appendChild(sourceDiv);
            }
            
            div.appendChild(avatar);
            div.appendChild(contentDiv);
            chat.appendChild(div);
            
            // Рендерим формулы
            await renderFormulas(textDiv);
            
            chat.scrollTop = chat.scrollHeight;
        }
        
        function showTyping() {
            const chat = document.getElementById('chat');
            const typingDiv = document.createElement('div');
            typingDiv.id = 'typing-indicator';
            typingDiv.className = 'typing';
            typingDiv.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div><span>🤖 Бот печатает...</span>';
            chat.appendChild(typingDiv);
            chat.scrollTop = chat.scrollHeight;
        }
        
        function hideTyping() {
            const typing = document.getElementById('typing-indicator');
            if (typing) typing.remove();
        }
        
        async function clearChat() {
            if (!confirm('Очистить историю диалога?')) return;
            try {
                await fetch('/api/clear', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                });
                document.getElementById('chat').innerHTML = '';
                await addMessage('bot', '🧹 История очищена. Задайте новый вопрос!');
            } catch(e) {
                console.error('Clear error:', e);
            }
        }
        
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                if (!response.ok) return;
                const stats = await response.json();
                
                const kbBadge = document.getElementById('kb-status');
                const webBadge = document.getElementById('web-status');
                const modelBadge = document.getElementById('model-status');
                
                if (kbBadge) kbBadge.innerHTML = stats.knowledge_base_loaded ? '📚 RAG ✅' : '📚 RAG ❌';
                if (webBadge) webBadge.innerHTML = stats.web_search_available ? '🌐 Поиск ✅' : '🌐 Поиск ❌';
                if (modelBadge) modelBadge.innerHTML = stats.ollama_available ? '🤖 ' + stats.ollama_model + ' ✅' : '🤖 Модель ❌';
            } catch(e) {
                console.log('Stats error:', e);
            }
        }
        
        // Auto-resize textarea
        const textarea = document.getElementById('message');
        if (textarea) {
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 150) + 'px';
            });
            
            textarea.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }
        
        window.sendMessage = sendMessage;
        window.clearChat = clearChat;
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    """Главная страница с чатом."""
    return HTMLResponse(content=SIMPLE_HTML)

# ---------------------------------------------------------------------------
# Статические файлы (опционально)
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")