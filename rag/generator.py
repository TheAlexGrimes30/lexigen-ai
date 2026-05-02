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

class QwenClient(BaseLLMClient):

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

class Generator(BaseGenerator):

    def __init__(
        self,
        llm: BaseLLMClient,
        prompt_builder: BasePromptBuilder,
        cleaner: BaseContextCleaner
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.cleaner = cleaner

    def generate(self, query: str, context: str) -> str:
        context = self.cleaner.clean_context(context)

        if not context:
            return "Ответ:\nНет данных в контексте\n\nИсточник:\n-"

        prompt = self.prompt_builder.build(query, context)
        raw = self.llm.generate(prompt)

        return self._postprocess(raw)

    def _postprocess(self, text: str) -> str:

        text = re.sub(r"КОНЕЦ_ОТВЕТА.*", "", text, flags=re.DOTALL)

        if text.count("Ответ:") > 1:
            text = "Ответ:" + text.split("Ответ:")[1]

        text = re.sub(r"Вот ответ:?", "", text, flags=re.IGNORECASE)

        text = re.split(r"(Источник:)", text, maxsplit=1)

        if len(text) >= 3:
            text = text[0] + text[1] + text[2]

        text = re.sub(r"\n{3,}", "\n\n", text)

        if "Источник:" not in text:
            text += "\n\nИсточник:\n-"

        return text.strip()
