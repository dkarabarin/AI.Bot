"""
Усиленная система Guardrails с агрессивной PII защитой
"""

import re
import time
from collections import defaultdict
from typing import Tuple, Dict
from dataclasses import dataclass, field

@dataclass
class SecurityMetrics:
    total_requests: int = 0
    blocked_requests: int = 0
    prompt_injection_detected: int = 0
    jailbreak_detected: int = 0
    academic_violation_detected: int = 0
    pii_detected: int = 0
    rate_limiting_blocks: int = 0
    
    @property
    def security_score(self) -> float:
        if self.total_requests == 0:
            return 100.0
        return max(0, 100 - (self.blocked_requests / self.total_requests * 100))
    
    def to_dict(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": round(self.blocked_requests / max(1, self.total_requests) * 100, 2),
            "security_score": round(self.security_score, 2),
            "detections": {
                "prompt_injection": self.prompt_injection_detected,
                "jailbreak": self.jailbreak_detected,
                "academic_violation": self.academic_violation_detected,
                "pii": self.pii_detected,
                "rate_limiting": self.rate_limiting_blocks
            }
        }


class LightweightGuardrails:
    def __init__(self, rate_limit: int = 10, rate_window: int = 60):
        self.metrics = SecurityMetrics()
        self.request_log = defaultdict(list)
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        
        # ============================================================
        # АГРЕССИВНЫЕ PII ПАТТЕРНЫ
        # ============================================================
        
        self.pii_patterns = [
            # Email (упрощенный паттерн для лучшего обнаружения)
            (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'email'),
            (r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}', 'email'),
            
            # Телефоны (российские)
            (r'\+7\s*\(?\d{3}\)?\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', 'phone'),
            (r'8\s*\(?\d{3}\)?\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', 'phone'),
            (r'\+7\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}', 'phone'),
            (r'8\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}', 'phone'),
            (r'\(\d{3}\)\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', 'phone'),
            (r'\d{3}[-\s]?\d{3}[-\s]?\d{4}', 'phone'),
            
            # Паспорт РФ
            (r'\d{4}\s*\d{6}', 'passport'),
            (r'\d{2}\s*\d{2}\s*\d{6}', 'passport'),
            (r'[А-ЯA-Z]{2}\s*\d{6}', 'passport'),
            
            # СНИЛС
            (r'\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}', 'snils'),
            (r'\d{11}', 'snils'),
            
            # ИНН
            (r'\b\d{10}\b', 'inn'),
            (r'\b\d{12}\b', 'inn'),
            
            # Банковские карты
            (r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}', 'card'),
            (r'\d{16}', 'card'),
        ]
        
        # Паттерны для prompt injection
        self.prompt_patterns = [
            (r"(?i)(ignore|forget|disregard).*(instructions|rules|prompts?)", "prompt_injection"),
            (r"(?i)(system|admin).*(override|reset|ignore)", "prompt_injection"),
            (r"(?i)(reveal|show|print|display).*(system.*prompt)", "prompt_injection"),
            (r"(?i)игнорируй.*инструкц", "prompt_injection"),
            (r"(?i)забудь.*предыдущ", "prompt_injection"),
            (r"(?i)системный промпт", "prompt_injection"),
            (r"(?i)покажи.*промпт", "prompt_injection"),
        ]
        
        # Паттерны для jailbreak
        self.jailbreak_patterns = [
            (r"(?i)(jailbreak|dan mode|aim mode)", "jailbreak"),
            (r"(?i)(DAN|Do Anything Now|AIM)", "jailbreak"),
            (r"(?i)no restrictions", "jailbreak"),
            (r"(?i)без ограничений", "jailbreak"),
        ]
        
        # Паттерны для академической честности
        self.academic_patterns = [
            (r"(?i)(cheat|crack|hack|steal|exploit)", "academic"),
            (r"(?i)(готовые? ответы?|списать|сдуть)", "academic"),
            (r"(?i)(сделай за меня|напиши за меня)", "academic"),
            (r"(?i)(лабораторную за меня|курсовую за меня)", "academic"),
            (r"(?i)(ответы на экзамен)", "academic"),
        ]
        
        # Паттерны для опасных запросов
        self.dangerous_patterns = [
            (r"(?i)how to (hack|steal|cheat)", "dangerous"),
            (r"(?i)как (взломать|украсть|обмануть)", "dangerous"),
        ]
    
    def check_pii(self, message: str) -> Tuple[bool, str]:
        """Агрессивная проверка PII"""
        message_lower = message.lower()
        
        for pattern, pii_type in self.pii_patterns:
            if re.search(pattern, message):
                return False, f"🔒 Запрос содержит персональные данные ({pii_type}). Пожалуйста, удалите их."
        return True, ""
    
    def check(self, message: str, session_id: str) -> Tuple[bool, str, Dict]:
        self.metrics.total_requests += 1
        details = {"risk_score": 0.0, "category": "safe"}
        
        # 1. PII проверка (САМЫЙ ВЫСОКИЙ ПРИОРИТЕТ)
        is_pii_safe, pii_msg = self.check_pii(message)
        if not is_pii_safe:
            self.metrics.blocked_requests += 1
            self.metrics.pii_detected += 1
            details["risk_score"] = 0.95
            details["category"] = "pii"
            return False, pii_msg, details
        
        # 2. Prompt injection
        for pattern, category in self.prompt_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                self.metrics.blocked_requests += 1
                self.metrics.prompt_injection_detected += 1
                details["category"] = category
                return False, "⛔ Запрос отклонен. Обнаружена попытка инъекции инструкций.", details
        
        # 3. Jailbreak
        for pattern, category in self.jailbreak_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                self.metrics.blocked_requests += 1
                self.metrics.jailbreak_detected += 1
                details["category"] = category
                return False, "🚫 Запрос отклонен. Обнаружена попытка обхода безопасности.", details
        
        # 4. Academic integrity
        for pattern, category in self.academic_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                self.metrics.blocked_requests += 1
                self.metrics.academic_violation_detected += 1
                details["category"] = category
                return False, "📚 Запрос отклонен. Я помогаю учиться, но не даю готовые ответы.", details
        
        # 5. Dangerous
        for pattern, category in self.dangerous_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                self.metrics.blocked_requests += 1
                details["category"] = category
                return False, "⚠️ Запрос отклонен. Вопрос нарушает политику безопасности.", details
        
        # 6. Rate limiting
        now = time.time()
        requests = self.request_log[session_id]
        requests[:] = [t for t in requests if now - t < self.rate_window]
        if len(requests) >= self.rate_limit:
            wait_time = self.rate_window - (now - requests[0])
            self.metrics.blocked_requests += 1
            self.metrics.rate_limiting_blocks += 1
            details["category"] = "rate_limit"
            return False, f"⏳ Слишком много запросов. Подождите {wait_time:.0f} секунд.", details
        requests.append(now)
        
        return True, "", details
    
    def check_response(self, prompt: str, response: str) -> Tuple[bool, str]:
        unsafe_indicators = [
            r"(?i)как взломать",
            r"(?i)how to hack",
            r"(?i)готовые ответы",
        ]
        for indicator in unsafe_indicators:
            if re.search(indicator, response, re.IGNORECASE):
                return False, "Ответ был отклонен системой безопасности."
        
        # Проверяем ответ на PII
        is_safe, _ = self.check_pii(response)
        if not is_safe:
            return False, "Ответ содержал персональные данные и был отклонен."
        
        return True, response
    
    def sanitize_pii(self, text: str) -> str:
        for pattern, pii_type in self.pii_patterns:
            text = re.sub(pattern, f"[{pii_type.upper()}_СКРЫТО]", text)
        return text
    
    def get_metrics(self) -> Dict:
        return self.metrics.to_dict()
    
    def get_status(self) -> Dict:
        return {
            "active": True,
            "version": "lightweight",
            "rate_limit": self.rate_limit,
            "pii_patterns_count": len(self.pii_patterns),
            "total_patterns": len(self.prompt_patterns) + len(self.jailbreak_patterns) + 
                             len(self.academic_patterns) + len(self.dangerous_patterns) + 
                             len(self.pii_patterns),
            "metrics": self.get_metrics()
        }


# Глобальный экземпляр
guardrails = LightweightGuardrails()