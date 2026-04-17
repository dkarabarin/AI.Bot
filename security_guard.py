"""
Усиленная система безопасности для ИИ-преподавателя по ТТД и ТМО
"""

import re
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List
from collections import defaultdict

class SecurityGuard:
    """Усиленная система безопасности"""
    
    def __init__(self):
        self.blocked_count = 0
        self.request_log = defaultdict(list)
        self.log_file = Path("security_log.json")
    
    # Расширенный список запрещенных паттернов
    BLOCK_PATTERNS = [
        # Prompt Injection (критические)
        (r"(?i)(ignore|forget|disregard).*(instructions|rules|prompts?)", "prompt_injection", 1.0),
        (r"(?i)(system|admin).*(override|reset|ignore)", "prompt_injection", 1.0),
        (r"(?i)(reveal|show|print|display).*(system.*prompt)", "system_leak", 1.0),
        (r"(?i)(jailbreak|dan mode|aim mode)", "jailbreak", 1.0),
        
        # Академическая честность
        (r"(?i)(cheat|crack|hack|steal|exploit)", "academic", 0.95),
        (r"(?i)(готовые? ответы?|списать|сдуть|срисовать)", "academic", 0.95),
        (r"(?i)(сделай за меня|напиши за меня|реши за меня)", "academic", 0.95),
        (r"(?i)(лабораторную работу за меня|курсовую за меня)", "academic", 0.95),
        (r"(?i)(обмануть преподавателя|обойти проверку)", "academic", 0.95),
        (r"(?i)(фальсифицировать данные|подделать результаты)", "academic", 0.95),
        (r"(?i)(ответы на экзамен|шпаргалка|шпоры)", "academic", 0.95),
        
        # Опасные темы
        (r"(?i)(how to hack|how to steal|how to cheat)", "dangerous", 0.95),
        (r"(?i)(взломать|украсть|обмануть)", "dangerous", 0.95),
    ]
    
    # Паттерны для PII
    PII_PATTERNS = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
        (r"\b\+?[78][\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b", "phone"),
        (r"\b\d{4}\s?\d{6}\b", "passport"),
        (r"\b\d{11}\b", "snils"),
        (r"\b\d{10}\b", "inn"),
    ]
    
    # Образовательные паттерны (разрешены)
    EDUCATIONAL_PATTERNS = [
        r"(?i)(термодинамик|тепломассообмен|энтропи|энтальпи|теплообмен)",
        r"(?i)(формула|расчет|вычисление|определение|нахождение)",
        r"(?i)(лабораторн|практикум|эксперимент|измерение)",
        r"(?i)(закон|правило|принцип|теорема)",
        r"(?i)(thermodynamics|heat transfer|entropy|enthalpy)",
        r"(?i)(nusselt|reynolds|prandtl|fourier|biot)",
        r"(?i)(первый закон|второй закон|третий закон)",
        r"(?i)(цикл карно|кпд|эффективность)",
        r"(?i)(идеальный газ|реальный газ|уравнение состояния)",
        r"(?i)(теплопроводность|конвекция|излучение|диффузия)",
        r"(?i)(изотермический|адиабатный|изобарный|изохорный)",
    ]
    
    def check(self, message: str, session_id: str) -> Tuple[bool, str]:
        """Проверка запроса на безопасность"""
        message_lower = message.lower()
        
        # Проверка на критически опасные паттерны
        for pattern, category, weight in self.BLOCK_PATTERNS:
            if re.search(pattern, message_lower):
                self.blocked_count += 1
                self._log_attack(message, category, session_id, weight)
                return False, self._get_block_message(category)
        
        # Проверка на PII
        for pattern, pii_type in self.PII_PATTERNS:
            if re.search(pattern, message):
                self.blocked_count += 1
                self._log_attack(message, f"pii_{pii_type}", session_id, 0.9)
                return False, "⚠️ Запрос содержит персональные данные. Пожалуйста, удалите их."
        
        # Проверка rate limiting
        is_rate_ok, rate_msg = self._check_rate_limit(session_id)
        if not is_rate_ok:
            return False, rate_msg
        
        return True, ""
    
    def _check_rate_limit(self, session_id: str, limit: int = 10, window: int = 60) -> Tuple[bool, str]:
        """Проверка ограничения частоты запросов"""
        now = time.time()
        requests = self.request_log[session_id]
        
        # Очищаем старые запросы
        requests[:] = [t for t in requests if now - t < window]
        
        if len(requests) >= limit:
            wait_time = window - (now - requests[0])
            return False, f"⏳ Слишком много запросов. Подождите {wait_time:.0f} секунд."
        
        requests.append(now)
        return True, ""
    
    def clean(self, text: str) -> str:
        """Очистка текста от PII"""
        for pattern, _ in self.PII_PATTERNS:
            text = re.sub(pattern, "[СКРЫТО]", text)
        return text
    
    def is_educational(self, message: str) -> bool:
        """Проверка, является ли запрос образовательным"""
        message_lower = message.lower()
        for pattern in self.EDUCATIONAL_PATTERNS:
            if re.search(pattern, message_lower):
                return True
        return False
    
    def _get_block_message(self, category: str) -> str:
        """Сообщение о блокировке"""
        messages = {
            "prompt_injection": "⛔ Запрос отклонен. Обнаружена попытка взлома системы безопасности.",
            "system_leak": "🔒 Запрос отклонен. Системная информация не разглашается.",
            "jailbreak": "🚫 Запрос отклонен. Обнаружена попытка обхода безопасности.",
            "dangerous": "⚠️ Запрос отклонен. Вопрос нарушает политику безопасности.",
            "academic": "📚 Запрос отклонен. Я помогаю учиться, но не даю готовые ответы. Задайте конкретный учебный вопрос по термодинамике.",
        }
        return messages.get(category, "❌ Запрос отклонен системой безопасности.")
    
    def _log_attack(self, message: str, category: str, session_id: str, score: float):
        """Логирование атаки"""
        log_entry = {
            "time": datetime.now().isoformat(),
            "session": session_id[:8],
            "category": category,
            "score": score,
            "message": message[:100]
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except:
            pass
    
    def get_stats(self) -> Dict:
        return {
            "blocked_count": self.blocked_count,
            "active_sessions": len(self.request_log),
            "log_file": str(self.log_file)
        }

security_guard = SecurityGuard()
security = security_guard