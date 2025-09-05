#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI风格API客户端
支持多轮对话、模型列表获取和流式输出
"""

import logging
import json
from typing import Dict, List, Optional, Union

import requests
from chat_client import ChatClient

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT = 300  # 默认超时时间（秒）
STREAM_PREFIX = "data: "
STREAM_END_MARKER = "[DONE]"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"


class OpenAIClient(ChatClient):
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
        super().__init__(api_key, base_url, system_prompt)

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
        max_tokens: int = -1,
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
            "stream": stream,
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens

        response = None
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
        except requests.exceptions.RequestException as e:
            error = f"""request error: {e}"""
            if response is not None and hasattr(response, 'request'):
                error += f"\nrequest headers: {response.request.headers}"
                error += f"\nrequest body: {response.request.body}"
            logging.error(error)
            return None
        except (json.JSONDecodeError, ValueError) as e:
            error = f"json decode error: {e}"
            if response is not None:
                error += f"response text: {response.text[:500]}..."
            logging.error(error)
            return None


    def embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str = DEFAULT_EMBEDDING_MODEL,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
    ) -> Optional[Dict]:
        """获取文本的向量嵌入。

        Args:
            input_text: 输入文本，可以是单个字符串或字符串列表
            model: 使用的嵌入模型，默认为 text-embedding-ada-002
            encoding_format: 编码格式，默认为 "float"
            dimensions: 输出维度，某些模型支持自定义维度

        Returns:
            嵌入向量数据或None
        """
        url = f"{self.base_url}/embeddings"

        payload = {
            "model": model,
            "input": input_text,
            "encoding_format": encoding_format,
        }

        if dimensions is not None:
            payload["dimensions"] = dimensions

        response = None
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error = f"""embeddings request error: {e}"""
            if response is not None and hasattr(response, 'request'):
                error += f"\nrequest headers: {response.request.headers}"
                error += f"\nrequest body: {response.request.body}"
            logging.error(error)
            return None
        except (json.JSONDecodeError, ValueError) as e:
            error = f"embeddings json decode error: {e}"
            if response is not None:
                error += f"response text: {response.text[:500]}..."
            logging.error(error)
            return None
