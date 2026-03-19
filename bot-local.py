import telebot
from dotenv import load_dotenv
import os
import logging
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Optional
import time
import requests
import socket
import urllib3
import ssl
from urllib3.exceptions import InsecureRequestWarning

# Отключаем все предупреждения SSL
urllib3.disable_warnings(InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()

# Импортируем все необходимое из rag.py
from rag import (
    ask_rag,
    get_collection_stats,
    get_api_status,
    tavily,
    linear,
    config as rag_config
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

class Config:
    """Конфигурация бота"""
    
    # Telegram
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    
    # Telegram API - используем прямой IP
    TELEGRAM_API_IP = "149.154.166.110"
    TELEGRAM_API_URL = f"https://{TELEGRAM_API_IP}"
    
    # Bot Settings
    MAX_HISTORY_LENGTH = 10
    MAX_HISTORY_AGE_HOURS = 24
    MAX_HISTORY_IN_CONTEXT = 4
    
    # Search Settings
    DEFAULT_K_RESULTS = 3
    
    # Security Settings
    MAX_MESSAGE_LENGTH = 4000
    RATE_LIMIT_MESSAGES = 10
    RATE_LIMIT_WINDOW = 60
    
    # Telegram API Settings
    TELEGRAM_API_RETRIES = 5
    TELEGRAM_API_TIMEOUT = 30
    
    # SSL Settings
    SSL_VERIFY = False  # Отключаем проверку SSL
    
    # Proxy Settings
    USE_PROXY = False
    PROXY_TYPE = 'socks5'
    PROXY_HOST = '127.0.0.1'
    PROXY_PORT = 9150
    
    # Content Filter Settings
    BLOCKED_TOPICS = [
        "политика", "government", "политическая партия", "выборы", 
        "президент", "putin", "путин", "война", "санкции",
        "протест", "митинг", "оппозиция", "нацизм", "фашизм",
        "терроризм", "экстремизм", "религиозная рознь"
    ]
    
    TOXIC_PATTERNS = [
        "убей", "убить", "смерть", "ненавижу", "терпеть не могу",
        "тупой", "идиот", "дебил", "придурок", "козел",
        "сука", "блядь", "хуй", "пизда", "ебал",
    ]
    
    WARNING_MESSAGE = "⚠️ Пожалуйста, воздержитесь от использования оскорбительных выражений и обсуждения политических тем. Я здесь, чтобы помочь с учебой!"
    ERROR_MESSAGE = "❌ Произошла ошибка. Пожалуйста, попробуйте позже."

config = Config()
logger.info("🚀 Запуск бота...")

# =============================================================================
# НАСТРОЙКА SSL И TELEGRAM API
# =============================================================================

def setup_telegram_api():
    """Настраивает Telegram API для работы через прямой IP с отключенным SSL"""
    try:
        from telebot import apihelper
        
        # Создаем кастомную сессию с отключенной проверкой SSL
        session = requests.Session()
        session.verify = False
        
        # Настраиваем API helper
        apihelper.SESSION = session
        
        # Устанавливаем базовый URL на прямой IP
        apihelper.API_URL = f"{config.TELEGRAM_API_URL}/bot{config.BOT_TOKEN}/"
        
        # Настраиваем прокси если нужно
        if config.USE_PROXY:
            try:
                proxy_url = f"{config.PROXY_TYPE}://{config.PROXY_HOST}:{config.PROXY_PORT}"
                apihelper.proxy = {'https': proxy_url}
                logger.info(f"🔌 Используется прокси: {proxy_url}")
            except Exception as e:
                logger.error(f"❌ Ошибка настройки прокси: {e}")
        
        logger.info(f"✅ Telegram API настроен на {config.TELEGRAM_API_IP} (SSL проверка отключена)")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка настройки Telegram API: {e}")
        return False

def create_ssl_disabled_session():
    """Создает сессию с отключенной проверкой SSL"""
    session = requests.Session()
    session.verify = False
    
    # Настраиваем адаптер с отключенным SSL
    adapter = requests.adapters.HTTPAdapter()
    session.mount('https://', adapter)
    
    return session

# =============================================================================
# ПРОВЕРКА СЕТИ
# =============================================================================

def check_internet_connection():
    """Проверяет доступность интернета"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        logger.info("✅ Интернет доступен")
        return True
    except OSError:
        logger.error("❌ Нет подключения к интернету")
        return False

def check_telegram_api():
    """Проверяет доступность Telegram API через прямой IP с отключенным SSL"""
    try:
        session = create_ssl_disabled_session()
        response = session.get(
            f"{config.TELEGRAM_API_URL}/bot{config.BOT_TOKEN}/getMe",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                logger.info(f"✅ Telegram API доступен (бот: @{data['result']['username']})")
                return True
        logger.error(f"❌ Telegram API вернул код {response.status_code}")
        return False
    except requests.exceptions.SSLError as e:
        logger.error(f"🔒 SSL ошибка (игнорируем): {e}")
        # Пробуем без проверки SSL через другой метод
        try:
            # Используем urllib3 напрямую
            import urllib3
            http = urllib3.PoolManager(cert_reqs='CERT_NONE', assert_hostname=False)
            response = http.request('GET', f"{config.TELEGRAM_API_URL}/bot{config.BOT_TOKEN}/getMe")
            if response.status == 200:
                import json
                data = json.loads(response.data.decode('utf-8'))
                if data.get("ok"):
                    logger.info(f"✅ Telegram API доступен (urllib3, бот: @{data['result']['username']})")
                    return True
        except Exception as e2:
            logger.error(f"❌ Ошибка при альтернативной проверке: {e2}")
        return False
    except requests.exceptions.Timeout:
        logger.error("❌ Таймаут при подключении к Telegram API")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Ошибка соединения с Telegram API: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка при проверке Telegram API: {e}")
        return False

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ БОТА
# =============================================================================

if not config.BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN не найден в .env файле!")

# Настраиваем API
setup_telegram_api()

def create_bot_with_retries():
    """Создает бота с повторными попытками при ошибках"""
    
    # Проверяем интернет
    if not check_internet_connection():
        logger.error("❌ Нет интернет-соединения")
        return None
    
    # Проверяем Telegram API
    check_telegram_api()
    
    for attempt in range(config.TELEGRAM_API_RETRIES):
        try:
            # Создаем бота с кастомной сессией
            bot = telebot.TeleBot(
                config.BOT_TOKEN,
                parse_mode='Markdown'
            )
            
            # Переопределяем внутреннюю сессию бота
            bot.session = create_ssl_disabled_session()
            
            # Проверяем соединение
            me = bot.get_me()
            logger.info(f"✅ Успешное подключение как @{me.username}")
            return bot
            
        except requests.exceptions.SSLError as e:
            logger.error(f"🔒 SSL ошибка при попытке {attempt + 1}: {e}")
            if attempt < config.TELEGRAM_API_RETRIES - 1:
                wait_time = 10 * (attempt + 1)
                logger.info(f"⏳ Повторная попытка через {wait_time} секунд...")
                time.sleep(wait_time)
            else:
                raise
                
        except requests.exceptions.Timeout as e:
            logger.error(f"⏱️ Попытка {attempt + 1}/{config.TELEGRAM_API_RETRIES} - Таймаут: {e}")
            if attempt < config.TELEGRAM_API_RETRIES - 1:
                wait_time = 10 * (attempt + 1)
                logger.info(f"⏳ Повторная попытка через {wait_time} секунд...")
                time.sleep(wait_time)
            else:
                raise
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"🔌 Попытка {attempt + 1}/{config.TELEGRAM_API_RETRIES} - Ошибка соединения: {e}")
            if attempt < config.TELEGRAM_API_RETRIES - 1:
                wait_time = 15 * (attempt + 1)
                logger.info(f"⏳ Повторная попытка через {wait_time} секунд...")
                time.sleep(wait_time)
            else:
                raise
                
        except Exception as e:
            logger.error(f"❌ Попытка {attempt + 1}/{config.TELEGRAM_API_RETRIES} - Ошибка: {e}")
            if attempt < config.TELEGRAM_API_RETRIES - 1:
                wait_time = 5 * (attempt + 1)
                logger.info(f"⏳ Повторная попытка через {wait_time} секунд...")
                time.sleep(wait_time)
            else:
                raise

try:
    bot = create_bot_with_retries()
    if bot is None:
        logger.error("💥 Не удалось создать бота после всех попыток")
        exit(1)
except Exception as e:
    logger.error(f"💥 Критическая ошибка при создании бота: {e}")
    logger.info("💡 Возможные решения:")
    logger.info("   1. Установите более новую версию pyTelegramBotAPI: pip install --upgrade pyTelegramBotAPI")
    logger.info("   2. Попробуйте использовать VPN")
    logger.info("   3. Проверьте правильность BOT_TOKEN")
    logger.info("   4. Установите pysocks: pip install pysocks requests[socks]")
    exit(1)

# История диалогов
user_histories = defaultdict(list)

# Rate limiting
rate_limits = defaultdict(list)

# =============================================================================
# ФУНКЦИИ БЕЗОПАСНОСТИ
# =============================================================================

def get_user_hash(user_id: int) -> str:
    """Создает анонимизированный хеш пользователя"""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:8]

def contains_banned_topic(text: str) -> bool:
    """Проверяет наличие запрещенных тем"""
    text_lower = text.lower()
    for topic in config.BLOCKED_TOPICS:
        if topic.lower() in text_lower:
            return True
    return False

def contains_toxic_language(text: str) -> bool:
    """Проверяет наличие токсичных выражений"""
    import re
    text_lower = text.lower()
    
    for pattern in config.TOXIC_PATTERNS:
        if pattern in text_lower:
            return True
    
    words = re.findall(r'\b\w+\b', text_lower)
    for word in words:
        if len(word) > 3 and word[0] in 'ухпбз' and word[-1] in 'йяи':
            toxic_endings = ['уй', 'хуй', 'ля', 'дебил', 'идиот']
            if any(ending in word for ending in toxic_endings):
                return True
    
    return False

def check_rate_limit(user_id: int) -> bool:
    """Проверяет rate limiting"""
    now = datetime.now()
    rate_limits[user_id] = [
        t for t in rate_limits[user_id] 
        if (now - t).total_seconds() < config.RATE_LIMIT_WINDOW
    ]
    
    if len(rate_limits[user_id]) >= config.RATE_LIMIT_MESSAGES:
        return False
    
    rate_limits[user_id].append(now)
    return True

# =============================================================================
# ФУНКЦИИ УПРАВЛЕНИЯ ИСТОРИЕЙ
# =============================================================================

def cleanup_old_messages(chat_id: int):
    """Удаляет старые сообщения"""
    now = datetime.now()
    user_histories[chat_id] = [
        msg for msg in user_histories[chat_id] 
        if (now - msg['timestamp']) < timedelta(hours=config.MAX_HISTORY_AGE_HOURS)
    ]

def add_to_history(chat_id: int, user_message: str, ai_response: str):
    """Добавляет сообщение в историю"""
    cleanup_old_messages(chat_id)
    
    user_histories[chat_id].append({
        'timestamp': datetime.now(),
        'human': user_message,
        'ai': ai_response
    })
    
    if len(user_histories[chat_id]) > config.MAX_HISTORY_LENGTH:
        user_histories[chat_id] = user_histories[chat_id][-config.MAX_HISTORY_LENGTH:]

def get_recent_history(chat_id: int) -> List[Dict]:
    """Возвращает последние сообщения"""
    cleanup_old_messages(chat_id)
    return user_histories[chat_id][-config.MAX_HISTORY_IN_CONTEXT:]

# =============================================================================
# ФУНКЦИЯ ФОРМАТИРОВАНИЯ ОТВЕТА
# =============================================================================

def format_rag_response(result: Dict) -> str:
    """Форматирует ответ RAG для отправки в Telegram"""
    
    if not result.get("success", False):
        return f"❌ {result.get('error', 'Неизвестная ошибка')}"
    
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    
    if not answer:
        return "❌ Не удалось получить ответ"
    
    formatted = f"💬 **Ответ:**\n{answer}\n"
    
    if sources:
        formatted += f"\n📚 **Источники:**\n"
        for src in sources[:3]:
            formatted += f"• {src}\n"
    
    return formatted

# =============================================================================
# ОБРАБОТЧИКИ КОМАНД
# =============================================================================

@bot.message_handler(commands=['start'])
def send_welcome(message):
    user_hash = get_user_hash(message.from_user.id)
    logger.info(f"👤 Новый пользователь: {user_hash}")
    
    api_status = get_api_status()
    stats = get_collection_stats()
    
    welcome_text = (
        "👋 **Привет! Я преподаватель по технической термодинамике.**\n\n"
        "📚 **Что я могу:**\n"
        "• Отвечать на вопросы по термодинамике\n"
        "• Использовать учебные материалы из папки /books\n\n"
        "🔌 **Статус системы:**\n"
        f"• 📚 База знаний: {stats.get('total_chunks', 0)} чанков\n"
        f"• 🤖 OpenRouter: {'✅' if api_status['openrouter']['available'] else '❌'}\n\n"
        "📋 **Доступные команды:**\n"
        "/start - приветствие\n"
        "/help - помощь\n"
        "/reset - сбросить историю\n"
        "/stats - статистика\n\n"
        "💡 **Просто задай вопрос по термодинамике!**"
    )
    
    bot.reply_to(message, welcome_text, parse_mode='Markdown')

@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = (
        "📚 **ПОМОЩЬ ПО КОМАНДАМ**\n\n"
        "/start - приветствие\n"
        "/help - помощь\n"
        "/reset - сбросить историю\n"
        "/stats - статистика базы знаний\n\n"
        "💡 **Советы:**\n"
        "• Задавай конкретные вопросы по термодинамике\n"
        "• Я отвечаю только на основе учебных материалов\n"
    )
    bot.reply_to(message, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['reset'])
def reset_history(message):
    chat_id = message.chat.id
    user_histories[chat_id] = []
    bot.reply_to(message, "🔄 История диалога сброшена!")

@bot.message_handler(commands=['stats'])
def send_stats(message):
    try:
        stats = get_collection_stats()
        stats_text = (
            f"📊 **СТАТИСТИКА БАЗЫ ЗНАНИЙ**\n\n"
            f"📚 Всего чанков: {stats.get('total_chunks', 0)}\n"
            f"🤖 Модель RAG: {stats.get('rag_model', 'N/A')}\n"
        )
        bot.reply_to(message, stats_text, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"❌ Ошибка stats: {e}")
        bot.reply_to(message, "❌ Не удалось получить статистику")

# =============================================================================
# ОСНОВНОЙ ОБРАБОТЧИК СООБЩЕНИЙ
# =============================================================================

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    user_hash = get_user_hash(user_id)
    user_text = message.text
    
    if user_text.startswith('/'):
        return
    
    if len(user_text) > config.MAX_MESSAGE_LENGTH:
        bot.reply_to(message, f"⚠️ Сообщение слишком длинное (макс. {config.MAX_MESSAGE_LENGTH} символов)")
        return
    
    if not check_rate_limit(user_id):
        bot.reply_to(message, "⏱️ Слишком много сообщений. Подождите минутку.")
        return
    
    if contains_banned_topic(user_text):
        logger.warning(f"🚫 Политика от {user_hash}")
        bot.reply_to(message, config.WARNING_MESSAGE)
        return
    
    if contains_toxic_language(user_text):
        logger.warning(f"🚫 Токсичность от {user_hash}")
        bot.reply_to(message, config.WARNING_MESSAGE)
        return
    
    bot.send_chat_action(chat_id, 'typing')
    
    try:
        logger.info(f"📝 Запрос от {user_hash}: {user_text[:50]}...")
        
        result = ask_rag(user_text, k=config.DEFAULT_K_RESULTS)
        response_text = format_rag_response(result)
        
        if result.get("success") and result.get("answer"):
            add_to_history(chat_id, user_text, result["answer"])
        
        bot.reply_to(message, response_text, parse_mode='Markdown')
        logger.info(f"✅ Ответ для {user_hash} отправлен")
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки сообщения: {e}")
        bot.reply_to(message, config.ERROR_MESSAGE)

# =============================================================================
# ЗАПУСК
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 TELEGRAM БОТ ЗАПУСКАЕТСЯ...")
    print("="*60)
    
    if not check_internet_connection():
        print("❌ Нет интернет-соединения!")
        exit(1)
    
    api_status = get_api_status()
    stats = get_collection_stats()
    
    print(f"📡 Telegram API IP: {config.TELEGRAM_API_IP}")
    print(f"📚 База знаний: {stats.get('total_chunks', 0)} чанков")
    print(f"🤖 OpenRouter: {'✅' if api_status['openrouter']['available'] else '❌'}")
    print("="*60)
    print("⏳ Попытка подключения к Telegram API...")
    print("="*60 + "\n")
    
    try:
        bot.polling(none_stop=True, interval=1, timeout=30)
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен пользователем")
        print("\n👋 До свидания!")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")