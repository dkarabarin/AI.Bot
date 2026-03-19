import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rag import get_relevant_chunks, get_collection_stats
import requests
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# =============================================================================
# КОНФИГУРАЦИЯ ЛОКАЛЬНОЙ LLM
# =============================================================================

# Настройки для локальной Ollama
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_API_URL = f"{OLLAMA_BASE}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"

# Модель по умолчанию (можно изменить)
DEFAULT_MODEL = "qwen3:4b"  # или "llama3.2:3b", "mistral:7b", "gemma:2b"

# Альтернативные модели (раскомментируйте нужную)
# DEFAULT_MODEL = "llama3.2:3b"
# DEFAULT_MODEL = "mistral:7b"
# DEFAULT_MODEL = "gemma:2b"
# DEFAULT_MODEL = "phi3:mini"

# =============================================================================
# ПРОВЕРКА OLLAMA
# =============================================================================

def check_ollama():
    """Проверяет доступность Ollama и наличие моделей"""
    try:
        # Проверяем, запущен ли Ollama сервер
        response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            
            print("\n🔍 Доступные модели Ollama:")
            for m in model_names:
                print(f"  • {m}")
            
            # Проверяем, есть ли наша модель
            if DEFAULT_MODEL not in model_names:
                print(f"\n⚠️ Модель {DEFAULT_MODEL} не найдена!")
                print(f"💡 Установите её командой: ollama pull {DEFAULT_MODEL}")
                
                # Предлагаем выбрать из доступных
                if model_names:
                    print("\n📌 Доступные модели:")
                    for i, m in enumerate(model_names, 1):
                        print(f"   {i}. {m}")
                    
                    try:
                        choice = input("\nВыберите номер модели (или нажмите Enter для выхода): ").strip()
                        if choice.isdigit() and 1 <= int(choice) <= len(model_names):
                            selected_model = model_names[int(choice) - 1]
                            print(f"✅ Выбрана модель: {selected_model}")
                            return selected_model
                        else:
                            print("❌ Выход...")
                            sys.exit(1)
                    except:
                        sys.exit(1)
                else:
                    print("\n❌ Нет доступных моделей!")
                    print("💡 Установите модель: ollama pull qwen3:4b")
                    sys.exit(1)
            
            print(f"\n✅ Используется модель: {DEFAULT_MODEL}")
            return DEFAULT_MODEL
            
        else:
            logger.error(f"❌ Ошибка ответа от Ollama: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Ollama не запущена!")
        print("\n🔧 Запустите Ollama командой: ollama serve")
        print("📥 Установите Ollama с https://ollama.com")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к Ollama: {e}")
        sys.exit(1)

# Инициализация
MODEL_NAME = check_ollama()

# =============================================================================
# ЛОКАЛЬНЫЙ LLM КЛИЕНТ
# =============================================================================

class LocalLLM:
    """Клиент для локальной LLM через Ollama"""
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = OLLAMA_BASE
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Генерирует ответ на основе промпта"""
        
        # Формируем запрос
        if system_prompt:
            # Используем chat API с системным промптом
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature
                }
            }
            url = OLLAMA_CHAT_URL
        else:
            # Используем простой generate API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature
                }
            }
            url = OLLAMA_API_URL
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "message" in result:
                return result["message"]["content"]
            elif "response" in result:
                return result["response"]
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            return f"Ошибка: {e}"
    
    def chat(self, messages: list) -> str:
        """Отправляет список сообщений и получает ответ"""
        
        # Преобразуем LangChain сообщения в формат Ollama
        ollama_messages = []
        system_content = None
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_content = msg.content
            elif isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                ollama_messages.append({"role": "assistant", "content": msg.content})
        
        # Если есть системное сообщение, добавляем его в начало
        if system_content:
            ollama_messages.insert(0, {"role": "system", "content": system_content})
        
        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }
        
        try:
            response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", str(result))
            
        except Exception as e:
            logger.error(f"❌ Ошибка чата: {e}")
            return f"Ошибка: {e}"

# Инициализация локальной LLM
llm = LocalLLM(model_name=MODEL_NAME, temperature=0.7)

# =============================================================================
# ИСТОРИЯ ДИАЛОГА
# =============================================================================

conversation_history = []

# Системный промпт
SYSTEM_PROMPT = """Ты — преподаватель по технической термодинамике. Твоя задача — проверять знания студента по дисциплине.

Ты должен:
1. Опираться только на материал из базы знаний (он приведён ниже). Не придумывай факты, которых нет в материале.
2. Задавать уточняющие вопросы и проверять понимание: определения, формулы, законы, типовые задачи.
3. Корректно указывать на ошибки в ответах студента и объяснять правильный ответ по материалу.
4. Быть строгим, но вежливым. Хвалить за верные ответы, чётко указывать на неверные.
5. Отвечать кратко и по делу, на русском языке.

Ниже приведён фрагмент материала из базы знаний по теме. Используй его как эталон для проверки ответов студента."""

# =============================================================================
# ФУНКЦИИ
# =============================================================================

def clear_screen():
    """Очищает экран терминала"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Выводит заголовок программы"""
    print("\n" + "="*70)
    print("🎓 ПРЕПОДАВАТЕЛЬ ПО ТЕХНИЧЕСКОЙ ТЕРМОДИНАМИКЕ (ЛОКАЛЬНАЯ ВЕРСИЯ)")
    print("="*70)
    print(f"🤖 Модель: {MODEL_NAME} (локально через Ollama)")
    
    # Показываем статистику базы знаний
    try:
        stats = get_collection_stats()
        print(f"📚 База знаний: {stats.get('total_chunks', 0)} чанков")
    except:
        print("📚 База знаний: загружена")
    
    print("="*70)
    print("📋 Команды: /help - помощь, /clear - очистить экран, /exit - выход")
    print("="*70 + "\n")

def print_help():
    """Выводит справку"""
    help_text = """
📚 **ДОСТУПНЫЕ КОМАНДЫ**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/help     - показать эту справку
/clear    - очистить экран
/stats    - статистика базы знаний
/history  - показать историю диалога
/clear_history - очистить историю
/models   - показать доступные модели
/switch   - переключить модель
/exit     - выйти из программы

💡 **Советы:**
• Просто задавай вопрос по термодинамике
• Можно спрашивать определения, формулы, законы
• Я отвечаю только на основе учебных материалов
• Если информации нет в базе, я честно скажу об этом
"""
    print(help_text)

def print_stats():
    """Выводит статистику базы знаний"""
    try:
        stats = get_collection_stats()
        print("\n📊 СТАТИСТИКА БАЗЫ ЗНАНИЙ")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📚 Всего чанков: {stats.get('total_chunks', 0)}")
        print(f"📁 Коллекция: {stats.get('collection_name', 'N/A')}")
        
        if 'sources' in stats and stats['sources']:
            print(f"\n📚 Источники:")
            for src, count in sorted(stats['sources'].items())[:5]:
                print(f"  • {src}: {count} чанков")
            if len(stats['sources']) > 5:
                print(f"  • ... и ещё {len(stats['sources']) - 5} источников")
    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")

def print_history():
    """Выводит историю диалога"""
    if not conversation_history:
        print("\n📭 История диалога пуста")
        return
    
    print("\n📜 ИСТОРИЯ ДИАЛОГА")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for i, msg in enumerate(conversation_history[-10:], 1):
        if isinstance(msg, HumanMessage):
            print(f"\n👤 [{i}] Вы: {msg.content[:100]}..." if len(msg.content) > 100 else f"\n👤 [{i}] Вы: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"🤖 [{i}] Бот: {msg.content[:100]}..." if len(msg.content) > 100 else f"🤖 [{i}] Бот: {msg.content}")

def list_available_models():
    """Показывает доступные модели в Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("\n📦 ДОСТУПНЫЕ МОДЕЛИ:")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for i, m in enumerate(models, 1):
                name = m.get("name")
                size = m.get("size", 0) / 1e9  # в GB
                print(f"  {i}. {name} ({size:.1f} GB)")
            return [m.get("name") for m in models]
        else:
            print("❌ Не удалось получить список моделей")
            return []
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return []

def switch_model():
    """Переключает текущую модель"""
    global llm, MODEL_NAME
    
    models = list_available_models()
    if not models:
        return
    
    try:
        choice = input("\nВыберите номер модели: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            new_model = models[int(choice) - 1]
            llm = LocalLLM(model_name=new_model, temperature=0.7)
            MODEL_NAME = new_model
            print(f"✅ Модель переключена на: {new_model}")
        else:
            print("❌ Неверный выбор")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def build_system_message(user_question: str) -> tuple:
    """Собирает системное сообщение и контекст"""
    try:
        chunks = get_relevant_chunks(user_question, k=3)
        if chunks:
            context = "\n\n---\n\n".join(chunks)
        else:
            context = "(Материал по запросу не найден. Проверяй знания в рамках общей дисциплины техническая термодинамика.)"
    except Exception as e:
        logger.error(f"❌ Ошибка получения чанков: {e}")
        context = "(Ошибка доступа к базе знаний)"
    
    return SYSTEM_PROMPT, context

def process_question(question: str):
    """Обрабатывает вопрос пользователя"""
    global conversation_history
    
    print("\n⏳ Думаю...")
    
    try:
        # Получаем системный промпт и контекст
        system_prompt, context = build_system_message(question)
        
        # Формируем сообщения для чата
        messages = []
        
        # Добавляем системное сообщение
        messages.append(SystemMessage(content=system_prompt))
        
        # Добавляем контекст как отдельное сообщение
        messages.append(HumanMessage(content=f"Вот материал из базы знаний:\n\n{context}"))
        
        # Добавляем историю (последние 6 сообщений)
        for msg in conversation_history[-6:]:
            messages.append(msg)
        
        # Добавляем текущий вопрос
        messages.append(HumanMessage(content=question))
        
        # Получаем ответ от локальной LLM
        response_text = llm.chat(messages)
        
        # Обновляем историю
        conversation_history.append(HumanMessage(content=question))
        conversation_history.append(AIMessage(content=response_text))
        
        # Ограничиваем историю последними 20 сообщениями
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        
        # Выводим ответ
        print(f"\n🤖 **Ответ:**\n{response_text}\n")
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки вопроса: {e}")
        print(f"\n❌ Произошла ошибка: {e}")

# =============================================================================
# ОСНОВНОЙ ЦИКЛ
# =============================================================================

def main():
    """Главная функция программы"""
    
    clear_screen()
    print_header()
    
    print("💡 Введите ваш вопрос по термодинамике (или /help для списка команд):\n")
    
    while True:
        try:
            # Получаем ввод пользователя (без readline, просто input)
            user_input = input("👤 Вы: ").strip()
            
            # Пропускаем пустые строки
            if not user_input:
                continue
            
            # Обработка команд
            if user_input.lower() == '/exit':
                print("\n👋 До свидания!")
                break
                
            elif user_input.lower() == '/clear':
                clear_screen()
                print_header()
                continue
                
            elif user_input.lower() == '/help':
                print_help()
                continue
                
            elif user_input.lower() == '/stats':
                print_stats()
                continue
                
            elif user_input.lower() == '/history':
                print_history()
                continue
                
            elif user_input.lower() == '/clear_history':
                conversation_history.clear()
                print("\n🔄 История диалога очищена!")
                continue
                
            elif user_input.lower() == '/models':
                list_available_models()
                continue
                
            elif user_input.lower() == '/switch':
                switch_model()
                continue
            
            # Обработка вопроса
            process_question(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
            
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка: {e}")
            print(f"\n❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 До свидания!")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")