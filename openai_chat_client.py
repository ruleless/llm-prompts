#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI风格API客户端
支持多轮对话、模型列表获取和流式输出
"""

import argparse
import json
import os
import sys
from typing import Dict, Iterator, List, Optional

import requests

# 定义模块导出的公共接口
__all__ = ['OpenAIClient', 'print_models', 'stream_chat']

# 常量定义
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
STREAM_PREFIX = "data: "
STREAM_END_MARKER = "[DONE]"


class OpenAIClient:
    """OpenAI风格API客户端"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        system_prompt: Optional[str] = None
    ) -> None:
        """初始化OpenAI API客户端。

        Args:
            api_key: API密钥
            base_url: API基础URL，默认为OpenAI官方API
            system_prompt: 系统提示词，用于设置AI助手的角色和行为
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []

        # 如果有系统提示词，添加到对话历史开头
        if self.system_prompt:
            self.conversation_history.append({"role": "system", "content": self.system_prompt})

    def list_models(self) -> List[Dict[str, str]]:
        """获取可用模型列表。

        Returns:
            模型列表，每个模型包含id和相关信息
        """
        url = f"{self.base_url}/models"

        try:
            response = requests.get(url, headers=self.headers)
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
        stream: bool = False
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
            "stream": stream
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload, stream=stream)
            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None

    def _handle_stream_response(self, response: requests.Response) -> Iterator[Dict]:
        """处理流式响应。

        Args:
            response: 响应对象

        Yields:
            流式响应数据块
        """
        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode('utf-8')
            if not decoded_line.startswith(STREAM_PREFIX):
                continue

            data_str = decoded_line[len(STREAM_PREFIX):]  # 移除 "data: " 前缀
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
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = system_prompt
        else:
            # 否则在开头添加系统消息
            self.conversation_history.insert(0, {"role": "system", "content": system_prompt})

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
            self.conversation_history = [{"role": "system", "content": self.system_prompt}]
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

    print("\nFrom Assistant:")
    print("-" * 30)

    full_response = ""
    for chunk in client.chat_completion(
        messages=client.get_history(),
        model=model,
        stream=True
    ):
        if chunk and "choices" in chunk:
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
        messages=client.get_history(),
        model=model,
        stream=False
    )

    if response and "choices" in response:
        assistant_message = response["choices"][0]["message"]["content"]
        print(f"\nAssistant response:\n{assistant_message}")
        client.add_message("assistant", assistant_message)
    else:
        print("Failed to get response")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="OpenAI-style API Client")
    parser.add_argument(
        "--api-key",
        help="API key",
        default=os.getenv("OPENAI_API_KEY")
    )
    parser.add_argument(
        "--base-url",
        help="API base URL",
        default=DEFAULT_BASE_URL
    )
    parser.add_argument(
        "--model",
        help="Model to use",
        default=DEFAULT_MODEL
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable stream output"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--system-prompt",
        help="System prompt for setting AI assistant role and behavior"
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: Please provide API key (via --api-key parameter or OPENAI_API_KEY environment variable)")
        sys.exit(1)

    client = OpenAIClient(args.api_key, args.base_url, args.system_prompt)

    if args.list_models:
        models = client.list_models()
        if models:
            print_models(models)
        else:
            print("Unable to get model list")
        return

    print(f"OpenAI-style API Client")
    print(f"API URL: {args.base_url}")
    print(f"Using model: {args.model}")
    print(f"Stream output: {'Enabled' if args.stream else 'Disabled'}")
    print("Type 'quit' to exit, 'clear' to clear history, 'models' to list models")
    print("Type 'system <prompt>' to set system prompt, 'show_system' to show current system prompt")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("Goodbye!")
                break
            elif user_input.lower() in ['clear', '清空']:
                client.clear_history()
                print("Conversation history cleared")
                continue
            elif user_input.lower() in ['models', '模型']:
                models = client.list_models()
                if models:
                    print_models(models)
                else:
                    print("Unable to get model list")
                continue
            elif user_input.lower().startswith('system '):
                # Set system prompt
                system_prompt = user_input[7:].strip()
                if system_prompt:
                    client.set_system_prompt(system_prompt)
                    print(f"System prompt set: {system_prompt}")
                else:
                    print("Please provide a system prompt after 'system'")
                continue
            elif user_input.lower() in ['show_system', '显示系统提示词']:
                current_system = client.get_system_prompt()
                if current_system:
                    print(f"Current system prompt: {current_system}")
                else:
                    print("No system prompt set")
                continue
            elif not user_input:
                continue

            if args.stream:
                stream_chat(client, user_input, args.model)
            else:
                normal_chat(client, user_input, args.model)

        except KeyboardInterrupt:
            print("\n\nProgram interrupted")
            break
        except Exception as e:
            print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
