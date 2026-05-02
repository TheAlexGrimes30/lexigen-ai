import re
from abc import abstractmethod, ABC

import requests


class BaseLLMClient(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class BasePromptBuilder(ABC):

    @abstractmethod
    def build(self, query: str, context: str) -> str:
        raise NotImplementedError


class BaseContextCleaner(ABC):

    @abstractmethod
    def clean_context(self, text: str) -> str:
        raise NotImplementedError


class BaseGenerator(ABC):

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        raise NotImplementedError

class ContextCleaner(BaseContextCleaner):

    def clean_context(self, text: str) -> str:
        text = re.sub(r"#+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

class OllamaClient(BaseLLMClient):

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout: int = 120,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

        self.options = {
            "temperature": 0.0,
            "top_p": 0.8,
            "top_k": 40,
            "repeat_penalty": 1.15,
            "num_predict": 200,
            "num_ctx": 4096,
            "num_thread": 8,
        }

        self.stop_tokens = ["\n\n\n", "Контекст:", "Вопрос:"]

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "options": self.options,
                },
                timeout=self.timeout,
            )

            response.raise_for_status()

            answer = response.json()["message"]["content"].strip()

            for token in self.stop_tokens:
                if token in answer:
                    answer = answer.split(token)[0]

            answer = re.sub(r"<.*?>", "", answer)
            answer = re.sub(r"\n{3,}", "\n\n", answer)

            return answer.strip()

        except Exception as e:
            return f"LLM error: {e}"
