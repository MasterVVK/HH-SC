import openai
import httpx
from httpx_socks import SyncProxyTransport
from config import OPENAI_API_KEY, PROXY_URL
import logging

logger = logging.getLogger(__name__)


class OpenAIService:
    """Класс для взаимодействия с OpenAI API через SOCKS5-прокси."""

    def __init__(self, timeout: int = 120):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY отсутствует. Проверьте файл config.py.")
        if not PROXY_URL:
            raise ValueError("PROXY_URL отсутствует. Проверьте файл config.py.")

        # Настройка прокси
        self.transport = SyncProxyTransport.from_url(PROXY_URL)
        self.client = httpx.Client(transport=self.transport, timeout=httpx.Timeout(timeout))

        # Устанавливаем API ключ для OpenAI
        openai.api_key = OPENAI_API_KEY

    def chat_completion(self, model: str, messages: list, max_tokens: int = 1000, temperature: float = 0.7):
        """
        Отправка запроса на OpenAI API для генерации текста.
        :param model: Модель для использования (например, "gpt-4").
        :param messages: Сообщения в формате [{"role": "system", "content": "..."}].
        :param max_tokens: Максимальное количество токенов в ответе.
        :param temperature: Температура генерации.
        :return: Ответ от модели.
        """
        try:
            logger.info(f"Запрос к OpenAI Chat Completion API с параметрами: {messages}")
            response = self.client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Ошибка при подключении к OpenAI API: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Ошибка статуса HTTP: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Неизвестная ошибка: {e}")
            raise
