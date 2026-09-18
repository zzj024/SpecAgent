"""llm.py：LLM 客户端——OpenAI 兼容协议 httpx 直连，不引 SDK。

为什么 httpx 裸调而不是 openai SDK：只需要 /chat/completions 一个端点 +
json_object 一种输出格式，SDK 连带引入十几个传递依赖（守则：每个依赖
必须能说出不可替代的理由——这里反面成立，裸调就是不可替代项）。
DeepSeek / Qwen / 任何 OpenAI 兼容服务只改 SPECAGENT_LLM_* 环境变量。

可靠性（L4）：指数退避重试 LLM_MAX_RETRY 次；调用方（nodes）再接熔断器。
available=False（未配 KEY）时调用方走 rule 轨，本类不抛错。
"""
import json
import time

import httpx

from core.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MAX_RETRY, LLM_MODEL, LLM_TIMEOUT_S,
)


class LLMUnavailable(Exception):
    """未配置 KEY 或重试耗尽——调用方据此降级到 rule 轨或标 degraded。"""


class LLMClient:
    def __init__(
        self,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat_json(self, system: str, user: str, max_retry: int = LLM_MAX_RETRY) -> dict:
        """对话 → 强制 JSON 对象输出。失败重试（指数退避），耗尽抛 LLMUnavailable。"""
        if not self.available:
            raise LLMUnavailable("未配置 DEEPSEEK_API_KEY")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,           # 审查判定要稳不要创作
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_err: Exception | None = None
        for attempt in range(1, max_retry + 1):
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, headers=headers,
                    timeout=LLM_TIMEOUT_S,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(content)
            except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < max_retry:
                    time.sleep(2 ** (attempt - 1))   # 1s, 2s, 4s
        raise LLMUnavailable(f"LLM 重试 {max_retry} 次耗尽: {last_err}")


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
