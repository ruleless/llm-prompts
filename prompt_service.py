#!/usr/bin/env python3

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai_chat_client import OpenAIClient, stream_chat

# 配置日志记录
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TemplateConfig:
    """模板配置类，用于组织模板相关的属性。"""

    template_path: Optional[str] = None
    template_vars: Dict[str, str] = None

    def __post_init__(self):
        if self.template_vars is None:
            self.template_vars = {}


@dataclass
class APIConfig:
    """API配置类，用于组织API相关的属性。"""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"


class PromptService:
    """提示词服务类，用于管理和处理系统提示词模板。

    该类提供了加载模板文件、替换模板变量、更新模板变量等功能，
    并与 OpenAI 客户端集成以提供聊天服务。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        **kwargs,
    ) -> None:
        """初始化 PromptService 实例。

        Args:
            api_key: OpenAI API 密钥
            base_url: API 基础 URL，默认为 OpenAI 官方地址
            model: 使用的模型名称，默认为 gpt-3.5-turbo
            **kwargs: 可选关键字参数
                system_template_path: 系统提示词模板文件路径，可选
                system_template_vars: 模板变量字典，可选
                user_template_path: 用户提示词模板文件路径，可选
                user_template_vars: 用户提示词模板变量字典，可选
        """
        # 创建API配置对象
        self._api_config = APIConfig(api_key=api_key, base_url=base_url, model=model)

        # 创建模板配置对象
        self._system_template_config = TemplateConfig(
            template_path=kwargs.get("system_template_path"),
            template_vars=kwargs.get("system_template_vars", {}),
        )
        self._user_template_config = TemplateConfig(
            template_path=kwargs.get("user_template_path"),
            template_vars=kwargs.get("user_template_vars", {}),
        )

        system_prompt = self._load_template_file(self._system_template_config)
        self._client = OpenAIClient(
            api_key=self._api_config.api_key,
            base_url=self._api_config.base_url,
            system_prompt=system_prompt,
        )

        # 预加载用户提示词模板
        self._user_template = self._load_template_file(self._user_template_config)

    def _load_template_file(self, template_config: TemplateConfig) -> Optional[str]:
        """通用的模板文件加载和处理方法。

        Args:
            template_config: 模板配置对象
            template_type: 模板类型，用于错误消息（"system" 或 "user"）

        Returns:
            处理后的模板字符串，如果没有设置模板路径则返回 None

        Raises:
            FileNotFoundError: 当模板文件不存在时
            RuntimeError: 当读取模板文件失败时
        """
        if not template_config.template_path:
            return None

        try:
            with open(template_config.template_path, "r", encoding="utf-8") as file:
                template = file.read().strip()
                return self._replace_template_variables(
                    template, template_config.template_vars
                )
        except FileNotFoundError:
            error_msg = (
                f"prompt template file not found: {template_config.template_path}"
            )
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Error occurred while reading template file: {e}"
            logger.error(error_msg)
            raise

    def _replace_template_variables(
        self, template: str, template_vars: Dict = None
    ) -> str:
        """替换模板中的变量占位符。

        Args:
            template: 包含变量占位符的模板字符串
            template_vars: 模板变量字典

        Returns:
            替换变量后的模板字符串

        变量占位符格式为 {{variable_name}}，将被替换为对应的值。
        """
        if template_vars is None:
            return template

        result = template
        for key, value in template_vars.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result

    def get_system_prompt(self) -> Optional[str]:
        """获取当前系统提示词。

        Returns:
            当前系统提示词字符串，如果没有设置则返回 None
        """
        return self._client.get_system_prompt()

    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self._client.get_history()

    def stream_chat(self, text: str) -> None:
        """启动流式聊天对话。

        Args:
            text: 要发送的文本消息
        """
        # 如果有用户提示词模板，则使用模板处理文本
        if self._user_template:
            # 将用户输入作为 {{user_input}} 变量处理
            user_template_vars = self._user_template_config.template_vars.copy()
            user_template_vars["user_input"] = text
            processed_text = self._replace_template_variables(
                self._user_template, user_template_vars
            )
        else:
            processed_text = text

        stream_chat(self._client, processed_text, self._api_config.model)


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please provide API key(via OPENAI_API_KEY environment variable)")
        sys.exit(1)

    base_url = "https://api.deepseek.com"
    model = "deepseek-chat"

    prompt_svc = PromptService(
        api_key=api_key,
        base_url=base_url,
        model=model,
        system_template_path="system-prompts/translation.txt",
        system_template_vars={
            "to": "Chinese",
        },
    )

    # print(prompt_svc.get_system_prompt())
    prompt_svc.stream_chat(text="你好，世界！")
    prompt_svc.stream_chat(text="你好吗")


if __name__ == "__main__":
    main()
