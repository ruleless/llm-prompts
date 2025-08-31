#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI风格API客户端
支持多轮对话、模型列表获取和流式输出
"""

import logging
import json
from typing import Dict, Iterator, List, Optional

import requests

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 300  # 默认超时时间（秒）
STREAM_PREFIX = "data: "
STREAM_END_MARKER = "[DONE]"


class OpenAIClient:
    """OpenAI风格API客户端"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        system_prompt: Optional[str] = None,
    ) -> None:
        """初始化OpenAI API客户端。

        Args:
            api_key: API密钥
            base_url: API基础URL，默认为OpenAI官方API
            system_prompt: 系统提示词，用于设置AI助手的角色和行为
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []

        # 如果有系统提示词，添加到对话历史开头
        if self.system_prompt:
            self.conversation_history.append(
                {"role": "system", "content": self.system_prompt}
            )

    def list_models(self) -> List[Dict[str, str]]:
        """获取可用模型列表。

        Returns:
            模型列表，每个模型包含id和相关信息
        """
        url = f"{self.base_url}/models"

        try:
            response = requests.get(url, headers=self.headers, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"获取模型列表失败: {e}")
            return []

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
    ) -> Optional[Dict]:
        """发送聊天完成请求。

        Args:
            messages: 消息列表
            model: 使用的模型
            temperature: 温度参数
            max_tokens: 最大令牌数
            stream: 是否使用流式输出

        Returns:
            响应数据或None
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=stream,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            return response.json()
        except Exception as e:
            error = f"""request error: {e}
request headers: {response.request.headers}
request body: {response.request.body}
"""
            logging.error(error)

    def _handle_stream_response(self, response: requests.Response) -> Iterator[Dict]:
        """处理流式响应。

        Args:
            response: 响应对象

        Yields:
            流式响应数据块
        """
        if response is None:
            return

        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode("utf-8")
            if not decoded_line.startswith(STREAM_PREFIX):
                continue

            data_str = decoded_line[len(STREAM_PREFIX) :]  # 移除 "data: " 前缀
            if self._is_stream_end(data_str):
                break

            data = self._parse_stream_data(data_str)
            if data:
                yield data

    def _is_stream_end(self, data_str: str) -> bool:
        """检查是否到达流式响应的结束。

        Args:
            data_str: 流式数据字符串

        Returns:
            是否到达结束标记
        """
        return data_str.strip() == STREAM_END_MARKER

    def _parse_stream_data(self, data_str: str) -> Optional[Dict]:
        """解析流式响应数据。

        Args:
            data_str: 流式数据字符串

        Returns:
            解析后的数据或None
        """
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            return None

    def add_message(self, role: str, content: str) -> None:
        """添加消息到对话历史。

        Args:
            role: 消息角色 (system, user, assistant)
            content: 消息内容
        """
        self.conversation_history.append({"role": role, "content": content})

    def set_system_prompt(self, system_prompt: str) -> None:
        """设置系统提示词。

        Args:
            system_prompt: 系统提示词内容
        """
        self.system_prompt = system_prompt

        # 如果对话历史中已有系统消息，更新它
        if (
            self.conversation_history
            and self.conversation_history[0]["role"] == "system"
        ):
            self.conversation_history[0]["content"] = system_prompt
        else:
            # 否则在开头添加系统消息
            self.conversation_history.insert(
                0, {"role": "system", "content": system_prompt}
            )

    def get_system_prompt(self) -> Optional[str]:
        """获取当前系统提示词。

        Returns:
            系统提示词内容，如果未设置则返回None
        """
        return self.system_prompt

    def clear_history(self, keep_system_prompt: bool = True) -> None:
        """清空对话历史。

        Args:
            keep_system_prompt: 是否保留系统提示词，默认为True
        """
        if keep_system_prompt and self.system_prompt:
            self.conversation_history = [
                {"role": "system", "content": self.system_prompt}
            ]
        else:
            self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史。

        Returns:
            对话历史列表
        """
        return self.conversation_history.copy()


def print_models(models: List[Dict[str, str]]) -> None:
    """Print model list.

    Args:
        models: List of models
    """
    if not models:
        print("No models available")
        return

    print("\nAvailable models:")
    print("-" * 50)
    for i, model in enumerate(models, 1):
        model_id = model.get("id", "Unknown")
        owned_by = model.get("owned_by", "Unknown")
        print(f"{i:2d}. {model_id:<30} (Owner: {owned_by})")
    print("-" * 50)


def stream_chat(client: OpenAIClient, user_input: str, model: str) -> None:
    """Stream chat.

    Args:
        client: OpenAI client instance
        user_input: User input
        model: Model to use
    """
    client.add_message("user", user_input)

    print("From Assistant:")
    print("-" * 30)

    full_response = ""
    response = client.chat_completion(
        messages=client.get_history(), model=model, stream=True
    )

    if response is None:
        print("Failed to get response from API")
        return

    for chunk in response:
        if not chunk or "choices" not in chunk:
            continue

        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")

        if content:
            print(content, end="", flush=True)
            full_response += content

    print("\n" + "-" * 30)

    if full_response:
        client.add_message("assistant", full_response)


def normal_chat(client: OpenAIClient, user_input: str, model: str) -> None:
    """Normal chat.

    Args:
        client: OpenAI client instance
        user_input: User input
        model: Model to use
    """
    client.add_message("user", user_input)

    response = client.chat_completion(
        messages=client.get_history(), model=model, stream=False
    )

    if response and "choices" in response:
        assistant_message = response["choices"][0]["message"]["content"]
        print(f"\nAssistant response:\n{assistant_message}")
        client.add_message("assistant", assistant_message)
    else:
        print("Failed to get response")
