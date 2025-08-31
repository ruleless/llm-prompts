#!/usr/bin/env python3

import os
import sys
from typing import Dict, Optional

from openai_chat_client import OpenAIClient, stream_chat


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
                template_path: 系统提示词模板文件路径，可选
                template_vars: 模板变量字典，可选
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.template_path = kwargs.get("template_path")
        self.template_vars = kwargs.get("template_vars", {})

        system_prompt = self._load_and_process_template()
        self._client = OpenAIClient(
            api_key=api_key, base_url=base_url, system_prompt=system_prompt
        )

    def _load_and_process_template(self) -> Optional[str]:
        """加载并处理模板文件。

        Returns:
            处理后的系统提示词字符串，如果没有设置模板路径则返回 None

        Raises:
            SystemExit: 当模板文件不存在或读取失败时退出程序
        """
        if not self.template_path:
            return None

        try:
            with open(self.template_path, "r", encoding="utf-8") as file:
                template = file.read().strip()
                return self._replace_template_variables(template)
        except FileNotFoundError:
            print(f"Error: System prompt template file not found {self.template_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error occurred while reading template file: {e}")
            sys.exit(1)

    def _replace_template_variables(self, template: str) -> str:
        """替换模板中的变量占位符。

        Args:
            template: 包含变量占位符的模板字符串

        Returns:
            替换变量后的模板字符串

        变量占位符格式为 {{variable_name}}，将被替换为对应的值。
        """
        result = template
        for key, value in self.template_vars.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result

    def update_template_vars(self, new_vars: Dict[str, str]) -> None:
        """更新模板变量并重新加载系统提示词。

        Args:
            new_vars: 新的模板变量字典，键为变量名，值为替换内容

        如果设置了模板路径，此方法会重新加载并处理模板文件，
        然后更新客户端的系统提示词。
        """
        self.template_vars.update(new_vars)
        if self.template_path:
            system_prompt = self._load_and_process_template()
            self._client.set_system_prompt(system_prompt)

    def get_system_prompt(self) -> Optional[str]:
        """获取当前系统提示词。

        Returns:
            当前系统提示词字符串，如果没有设置则返回 None
        """
        return self._client.get_system_prompt()

    def stream_chat(self, text: str) -> None:
        """启动流式聊天对话。

        Args:
            text: 要发送的文本消息
        """
        stream_chat(self._client, text, self.model)


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: Please provide API key " "(via OPENAI_API_KEY environment variable)"
        )
        sys.exit(1)

    base_url = "https://api.deepseek.com"
    model = "deepseek-chat"

    prompt_svc = PromptService(
        api_key=api_key,
        base_url=base_url,
        model=model,
        template_path="system-prompts/translation.txt",
        template_vars={
            "to": "Chinese",
        },
    )

    # print(prompt_svc.get_system_prompt())
    prompt_svc.stream_chat(text="你好，世界！")
    prompt_svc.stream_chat(text="你好吗")


if __name__ == "__main__":
    main()
