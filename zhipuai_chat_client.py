#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Dict, Iterator, List, Optional, Union

from zai import ZhipuAiClient
from chat_client import ChatClient

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_MODEL = "glm-4"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_EMBEDDING_MODEL = "embedding-2"


class ZhipuAiChatClient(ChatClient):
    """智谱AI聊天客户端"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """初始化智谱AI聊天客户端。

        Args:
            api_key: API密钥
            base_url: API基础URL，如果为None则使用默认URL
            system_prompt: 系统提示词，用于设置AI助手的角色和行为
        """
        # 初始化基类，使用空字符串作为base_url，因为我们会使用ZhipuAiClient的默认URL
        super().__init__(api_key, "", system_prompt)

        # 创建ZhipuAiClient实例
        self.client = ZhipuAiClient(api_key=api_key, base_url=base_url)

    def list_models(self) -> List[Dict[str, str]]:
        """获取可用模型列表。

        Returns:
            模型列表，每个模型包含id和相关信息
        """
        try:
            # 智谱AI的模型列表API
            response = self.client.get("/models", cast_type=dict)
            if response and "data" in response:
                return response["data"]
            return []
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = -1,
        stream: bool = False,
    ) -> Optional[Union[Dict, Iterator[Dict]]]:
        """发送聊天完成请求。

        Args:
            messages: 消息列表
            model: 使用的模型
            temperature: 温度参数
            max_tokens: 最大令牌数
            stream: 是否使用流式输出

        Returns:
            响应数据或流式响应迭代器
        """
        try:
            # 准备参数
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
            }

            # 如果指定了最大令牌数，则添加到参数中
            if max_tokens > 0:
                kwargs["max_tokens"] = max_tokens

            # 调用智谱AI的聊天完成API
            response = self.client.chat.completions.create(**kwargs)

            if stream:
                # 处理流式响应
                return self._handle_zhipu_stream_response(response)
            else:
                # 处理普通响应
                return self._convert_zhipu_response(response)
        except Exception as e:
            logger.error(f"聊天完成请求失败: {e}")
            return None

    def _convert_zhipu_response(self, response) -> Dict:
        """转换智谱AI的响应为标准格式。

        Args:
            response: 智谱AI的响应

        Returns:
            标准格式的响应数据
        """
        if not response:
            return {}

        # 将智谱AI的响应转换为与OpenAI兼容的格式
        return {
            "id": getattr(response, "id", ""),
            "object": getattr(response, "object", "chat.completion"),
            "created": getattr(response, "created", 0),
            "model": getattr(response, "model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": getattr(response.choices[0].message, "content", ""),
                    },
                    "finish_reason": getattr(response.choices[0], "finish_reason", "stop"),
                }
            ],
            "usage": getattr(response, "usage", {}),
        }

    def _handle_zhipu_stream_response(self, response) -> Iterator[Dict]:
        """处理智谱AI的流式响应。

        Args:
            response: 智谱AI的流式响应

        Yields:
            流式响应数据块（原始JSON数据格式）
        """
        if not response:
            return

        try:
            for chunk in response:
                # 检查chunk是否有效
                if not chunk or not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                choice = chunk.choices[0]

                # 检查choice是否有效
                if not hasattr(choice, "delta"):
                    continue

                delta = choice.delta

                # 获取delta内容，如果content为空且role为空，则跳过此块
                content = getattr(delta, "content", "")
                role = getattr(delta, "role", "")

                # 如果既没有内容也没有角色信息，跳过这个块
                if not content and not role:
                    # 检查是否有finish_reason，如果有则仍然需要yield
                    finish_reason = getattr(choice, "finish_reason", None)
                    if not finish_reason:
                        continue

                # 构造与父类一致的原始数据格式
                data = {
                    "id": getattr(chunk, "id", ""),
                    "object": getattr(chunk, "object", "chat.completion.chunk"),
                    "created": getattr(chunk, "created", 0),
                    "model": getattr(chunk, "model", ""),
                    "choices": [
                        {
                            "index": getattr(choice, "index", 0),
                            "delta": {
                                "role": role,
                                "content": content,
                            },
                            "finish_reason": getattr(choice, "finish_reason", None),
                        }
                    ],
                }

                # 如果有usage信息，也添加到数据中
                if hasattr(chunk, "usage") and chunk.usage:
                    data["usage"] = chunk.usage

                yield data

                # 如果是最后一个块（有finish_reason），记录日志
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason:
                    logger.debug(f"流式响应结束，finish_reason: {finish_reason}")

        except Exception as e:
            logger.error(f"处理流式响应失败: {e}")
            # 与父类保持一致，只记录日志不重新抛出异常
            return

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
            model: 使用的嵌入模型
            encoding_format: 编码格式，默认为 "float"
            dimensions: 输出维度，某些模型支持自定义维度

        Returns:
            嵌入向量数据或None
        """
        try:
            # 准备参数
            kwargs = {
                "input": input_text,
                "model": model,
                "encoding_format": encoding_format,
            }

            # 如果指定了维度，则添加到参数中
            if dimensions is not None:
                kwargs["dimensions"] = dimensions

            # 调用智谱AI的嵌入API
            response = self.client.embeddings.create(**kwargs)

            # 转换响应为标准格式
            return self._convert_embeddings_response(response)
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            return None

    def _convert_embeddings_response(self, response) -> Dict:
        """转换智谱AI的嵌入响应为标准格式。

        Args:
            response: 智谱AI的嵌入响应

        Returns:
            标准格式的嵌入向量数据
        """
        if not response:
            return {}

        # 将智谱AI的嵌入响应转换为与OpenAI兼容的格式
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": data.embedding,
                    "index": i,
                }
                for i, data in enumerate(response.data)
            ],
            "model": response.model,
            "usage": getattr(response, "usage", {}),
        }
