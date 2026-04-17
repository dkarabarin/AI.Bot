"""
FastAPI-сервер для веб-интерфейса ИИ преподавателя по ТТД и ТМО
"""

from __future__ import annotations

import logging
import sys
import time
import asyncio
import json
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import AsyncGenerator

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from guardrails_light import guardrails

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Загрузка бота
# ============================================================================

get_answer = None
get_answer_stream = None
web_search = None
knowledge_base = None
OLLAMA_MODEL = "qwen3:4b"

bot_local_path = ROOT_DIR / "bot-local.py"

if bot_local_path.exists():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bot_local", bot_local_path)
        bot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot_module)
        
        get_answer = getattr(bot_module, 'get_answer', None)
        get_answer_stream = getattr(bot_module, 'get_answer_stream', None)
        web_search = getattr(bot_module, 'web_search', None)
        knowledge_base = getattr(bot_module, 'knowledge_base', None)
        OLLAMA_MODEL = getattr(bot_module, 'OLLAMA_MODEL', 'qwen3:4b')
        
        logger.info("✅ Загружен bot-local.py")
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")

if get_answer is None:
    def get_answer(question, session_id=None):
        return f"📚 **Вопрос:** {question}\n\n**Ответ:** Это тестовый режим. Для работы установите Ollama.", "⚠️ Тест"

if get_answer_stream is None:
    async def get_answer_stream(question: str, session_id: str = None) -> AsyncGenerator[str, None]:
        answer, source = get_answer(question, session_id)
        words = answer.split()
        for i in range(0, len(words), 3):
            chunk = ' '.join(words[i:i+3])
            yield json.dumps({"chunk": chunk + " ", "source": source if i == 0 else None, "done": False}) + "\n"
            await asyncio.sleep(0.03)
        yield json.dumps({"chunk": "", "source": source, "done": True}) + "\n"

# ============================================================================
# Модели
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: str
    stream: bool = True

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    source: str
    processing_time: float

class ClearRequest(BaseModel):
    session_id: str

# ============================================================================
# Хранилище
# ============================================================================

class SessionStore:
    def __init__(self):
        self.sessions = {}
        self.history = {}
    
    def get_or_create(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"created": datetime.now(), "count": 0}
            self.history[session_id] = []
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str, source: str = None):
        if session_id not in self.history:
            self.history[session_id] = []
        self.history[session_id].append({
            "role": role, "content": content, "time": datetime.now().isoformat(), "source": source
        })
        if session_id in self.sessions:
            self.sessions[session_id]["count"] += 1
    
    def clear(self, session_id: str):
        if session_id in self.history:
            self.history[session_id] = []
        if session_id in self.sessions:
            self.sessions[session_id]["count"] = 0

store = SessionStore()

# ============================================================================
# FastAPI
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск сервера")
    yield
    logger.info("👋 Остановка")

app = FastAPI(title="ИИ преподаватель по ТТД и ТМО", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API
# ============================================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

@app.get("/api/stats")
def stats():
    kb_loaded = knowledge_base and hasattr(knowledge_base, '_loaded') and knowledge_base._loaded
    web_avail = web_search and hasattr(web_search, 'is_available') and web_search.is_available()
    
    return {
        "knowledge_base_loaded": kb_loaded,
        "web_search_available": web_avail,
        "web_search_engine": web_search.get_engine_name() if web_search else "Нет",
        "ollama_model": OLLAMA_MODEL,
        "ollama_available": True,
        "total_sessions": len(store.sessions),
        "guardrails": guardrails.get_status(),
        "streaming_supported": True
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Обычный endpoint без стриминга (для тестирования и обратной совместимости)"""
    logger.info(f"Обычный запрос: {req.message[:50]}...")
    
    # Проверка безопасности
    is_safe, error_msg, details = guardrails.check(req.message, req.session_id)
    
    if not is_safe:
        logger.warning(f"Блокировка: {details.get('category')}")
        return ChatResponse(
            reply=error_msg,
            session_id=req.session_id,
            source="🛡️ Guardrails",
            processing_time=0.0
        )
    
    store.get_or_create(req.session_id)
    store.add_message(req.session_id, "user", req.message)
    
    start = time.time()
    
    try:
        result = get_answer(req.message, session_id=req.session_id)
        
        if isinstance(result, tuple):
            answer, source = result
        else:
            answer = result
            source = "🤖 Бот"
        
        elapsed = time.time() - start
        
        store.add_message(req.session_id, "assistant", answer, source)
        
        return ChatResponse(
            reply=answer,
            session_id=req.session_id,
            source=source,
            processing_time=elapsed
        )
    except Exception as e:
        logger.exception("Ошибка")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Endpoint с поддержкой стриминга"""
    logger.info(f"Stream запрос: {req.message[:50]}...")
    
    # Проверка безопасности
    is_safe, error_msg, details = guardrails.check(req.message, req.session_id)
    
    if not is_safe:
        async def error_stream():
            yield json.dumps({"chunk": error_msg, "source": "🛡️ Guardrails", "done": True}) + "\n"
        return StreamingResponse(error_stream(), media_type="application/x-ndjson")
    
    store.get_or_create(req.session_id)
    store.add_message(req.session_id, "user", req.message)
    
    async def generate():
        full_answer = ""
        source = None
        
        try:
            async for chunk_data in get_answer_stream(req.message, session_id=req.session_id):
                if isinstance(chunk_data, str):
                    try:
                        data = json.loads(chunk_data)
                        chunk = data.get("chunk", "")
                        source = data.get("source", source)
                        is_done = data.get("done", False)
                    except:
                        chunk = chunk_data
                        is_done = False
                else:
                    chunk = chunk_data.get("chunk", "")
                    source = chunk_data.get("source", source)
                    is_done = chunk_data.get("done", False)
                
                full_answer += chunk
                yield json.dumps({"chunk": chunk, "source": source, "done": is_done}) + "\n"
                
                if is_done:
                    store.add_message(req.session_id, "assistant", full_answer, source)
                    
        except Exception as e:
            logger.exception(f"Stream ошибка: {e}")
            yield json.dumps({"chunk": f"\n\n❌ Ошибка: {str(e)}", "source": "⚠️ Ошибка", "done": True}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.post("/api/clear")
def clear(req: ClearRequest):
    store.clear(req.session_id)
    return {"status": "ok"}


# ============================================================================
# HTML
# ============================================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ИИ преподаватель по ТТД и ТМО</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 90vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e, #0f3460);
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .header h1 { font-size: 1.2rem; }
        .status { display: flex; gap: 8px; flex-wrap: wrap; }
        .badge {
            background: rgba(255,255,255,0.2);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
        }
        .clear-btn {
            background: rgba(255,255,255,0.2);
            margin-left: 8px;
            cursor: pointer;
        }
        .clear-btn:hover { background: rgba(255,255,255,0.3); }
        .chat {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f0f2f5;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message {
            display: flex;
            gap: 10px;
        }
        .message.user { justify-content: flex-end; }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            flex-shrink: 0;
        }
        .message.user .avatar { background: #e94560; }
        .message.bot .avatar { background: #0f3460; }
        .bubble {
            max-width: 75%;
            padding: 12px 16px;
            border-radius: 18px;
            background: white;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .message.user .bubble {
            background: #e94560;
            color: white;
        }
        .bubble-text { line-height: 1.5; }
        .bubble-text p { margin-bottom: 8px; }
        .bubble-source {
            font-size: 0.7rem;
            margin-top: 6px;
            opacity: 0.7;
        }
        .cursor-blink {
            display: inline-block;
            width: 2px;
            height: 1.2em;
            background-color: #e94560;
            margin-left: 2px;
            animation: blink 1s step-end infinite;
            vertical-align: middle;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
        .input-area {
            padding: 15px 20px;
            background: white;
            border-top: 1px solid #e1e8ed;
            display: flex;
            gap: 10px;
        }
        textarea {
            flex: 1;
            padding: 10px 15px;
            border: 2px solid #e1e8ed;
            border-radius: 25px;
            resize: none;
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
        }
        textarea:focus { border-color: #e94560; }
        button {
            background: #e94560;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.95rem;
        }
        button:hover { background: #c73e56; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .typing {
            padding: 10px 15px;
            background: white;
            border-radius: 18px;
            width: fit-content;
            display: flex;
            gap: 8px;
            color: #666;
            font-style: italic;
        }
        .dot {
            width: 6px;
            height: 6px;
            background: #666;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
        @media (max-width: 768px) {
            .bubble { max-width: 85%; }
            .header h1 { font-size: 1rem; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🔥 ИИ преподаватель по ТТД и ТМО</h1>
        <div class="status">
            <div class="badge" id="badge-kb">📚 RAG</div>
            <div class="badge" id="badge-web">🌐 Поиск</div>
            <div class="badge" id="badge-guard">🛡️ Guardrails</div>
            <div class="badge clear-btn" id="btn-clear">🗑️ Очистить</div>
        </div>
    </div>
    <div class="chat" id="chat"></div>
    <div class="input-area">
        <textarea id="input" placeholder="Введите вопрос по термодинамике..." rows="1"></textarea>
        <button id="btn-send">📤 Отправить</button>
    </div>
</div>

<script>
(function() {
    let sessionId = localStorage.getItem("session_id");
    if (!sessionId) {
        sessionId = crypto.randomUUID();
        localStorage.setItem("session_id", sessionId);
    }
    
    let isLoading = false;
    let fullAnswer = "";
    
    const chat = document.getElementById("chat");
    const input = document.getElementById("input");
    const sendBtn = document.getElementById("btn-send");
    const clearBtn = document.getElementById("btn-clear");
    const badgeKb = document.getElementById("badge-kb");
    const badgeWeb = document.getElementById("badge-web");
    const badgeGuard = document.getElementById("badge-guard");
    
    function formatText(text) {
        if (!text) return "";
        let result = text;
        result = result.replace(/&/g, "&amp;");
        result = result.replace(/</g, "&lt;");
        result = result.replace(/>/g, "&gt;");
        result = result.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        result = result.replace(/\n/g, "<br>");
        return result;
    }
    
    function addUserMessage(content) {
        const div = document.createElement("div");
        div.className = "message user";
        
        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = "👤";
        
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        
        const textDiv = document.createElement("div");
        textDiv.className = "bubble-text";
        textDiv.innerHTML = formatText(content);
        bubble.appendChild(textDiv);
        
        div.appendChild(avatar);
        div.appendChild(bubble);
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
    }
    
    function createStreamingMessage() {
        const div = document.createElement("div");
        div.className = "message bot";
        div.id = "streaming-message";
        
        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = "🤖";
        
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        
        const textDiv = document.createElement("div");
        textDiv.className = "bubble-text";
        
        const contentSpan = document.createElement("span");
        contentSpan.id = "streaming-content";
        const cursorSpan = document.createElement("span");
        cursorSpan.className = "cursor-blink";
        
        textDiv.appendChild(contentSpan);
        textDiv.appendChild(cursorSpan);
        bubble.appendChild(textDiv);
        
        const sourceDiv = document.createElement("div");
        sourceDiv.className = "bubble-source";
        sourceDiv.id = "streaming-source";
        bubble.appendChild(sourceDiv);
        
        div.appendChild(avatar);
        div.appendChild(bubble);
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
        
        return { contentSpan, sourceDiv, cursorSpan };
    }
    
    function updateStreaming(content, source, isDone) {
        const contentSpan = document.getElementById("streaming-content");
        const sourceDiv = document.getElementById("streaming-source");
        const cursorSpan = document.querySelector(".cursor-blink");
        
        if (contentSpan) {
            contentSpan.innerHTML = formatText(content);
        }
        if (sourceDiv && source) {
            sourceDiv.textContent = source;
        }
        if (isDone && cursorSpan) {
            cursorSpan.style.display = "none";
        }
        
        if (window.MathJax && contentSpan) {
            MathJax.typesetPromise([contentSpan]).catch(console.error);
        }
        
        chat.scrollTop = chat.scrollHeight;
    }
    
    function showLoading() {
        const div = document.createElement("div");
        div.id = "loading";
        div.className = "typing";
        div.innerHTML = '<div style="display:flex;gap:4px"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div><span>🤖 Преподаватель печатает...</span>';
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
    }
    
    function hideLoading() {
        const el = document.getElementById("loading");
        if (el) el.remove();
    }
    
    async function sendMessage() {
        const message = input.value.trim();
        if (!message || isLoading) return;
        
        input.value = "";
        input.style.height = "auto";
        
        addUserMessage(message);
        
        isLoading = true;
        sendBtn.disabled = true;
        showLoading();
        
        fullAnswer = "";
        createStreamingMessage();
        
        try {
            const response = await fetch("/api/chat/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: message, session_id: sessionId, stream: true })
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";
                
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const data = JSON.parse(line);
                            if (data.chunk !== undefined) {
                                fullAnswer += data.chunk;
                                updateStreaming(fullAnswer, data.source, data.done);
                            }
                        } catch (e) {
                            console.error("Parse error:", e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Stream error:", error);
            updateStreaming(fullAnswer + "\n\n❌ Ошибка: " + error.message, "⚠️ Ошибка", true);
        } finally {
            hideLoading();
            isLoading = false;
            sendBtn.disabled = false;
            input.focus();
        }
    }
    
    async function clearChat() {
        if (!confirm("Очистить историю диалога?")) return;
        try {
            await fetch("/api/clear", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId })
            });
            chat.innerHTML = "";
            addWelcomeMessage();
        } catch (err) {
            console.error("Clear error:", err);
        }
    }
    
    async function loadStats() {
        try {
            const res = await fetch("/api/stats");
            if (!res.ok) return;
            const stats = await res.json();
            
            if (badgeKb) {
                badgeKb.innerHTML = stats.knowledge_base_loaded ? "📚 RAG ✅" : "📚 RAG ❌";
            }
            if (badgeWeb) {
                const engine = stats.web_search_engine || "Поиск";
                badgeWeb.innerHTML = stats.web_search_available ? "🌐 " + engine + " ✅" : "🌐 Поиск ❌";
            }
            if (badgeGuard) {
                badgeGuard.innerHTML = stats.guardrails?.active ? "🛡️ Guardrails ✅" : "🛡️ Guardrails ⚠️";
            }
        } catch(e) {
            console.log("Stats error:", e);
        }
    }
    
    function addWelcomeMessage() {
        const div = document.createElement("div");
        div.className = "message bot";
        
        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = "🤖";
        
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        
        const textDiv = document.createElement("div");
        textDiv.className = "bubble-text";
        textDiv.innerHTML = "👋 Здравствуйте! Я ИИ-преподаватель по <strong>технической термодинамике (ТТД)</strong> и <strong>тепломассообмену (ТМО)</strong>.<br><br>Задайте мне вопрос по:<br>• 📚 Лабораторным работам<br>• 📊 Обработке данных<br>• 📖 Теоретическому материалу<br>• 🔬 Подготовке к экзаменам";
        
        bubble.appendChild(textDiv);
        div.appendChild(avatar);
        div.appendChild(bubble);
        chat.appendChild(div);
    }
    
    sendBtn.onclick = sendMessage;
    clearBtn.onclick = clearChat;
    
    input.onkeydown = function(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };
    
    input.oninput = function() {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 120) + "px";
    };
    
    loadStats();
    setInterval(loadStats, 30000);
    addWelcomeMessage();
    input.focus();
    
    console.log("Chat initialized, sessionId:", sessionId);
})();
</script>
</body>
</html>
"""

@app.get("/")
async def root():
    return HTMLResponse(content=HTML_PAGE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)