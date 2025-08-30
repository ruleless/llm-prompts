#!/usr/bin/env python3
"""
翻译程序

使用系统提示词模板和OpenAI客户端完成翻译任务。
"""

import os
import sys

from openai_chat_client import OpenAIClient, stream_chat


__all__ = ["TranslationService"]


class TranslationService:

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        from_lang: str = "中文",
        to_lang: str = "英文"
    ) -> None:
        system_prompt = self._load_system_prompt(from_lang, to_lang)
        self._client = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt
        )

    def _load_system_prompt(
        self,
        from_lang: str,
        to_lang: str,
        template_path: str = "system-prompts/translation.txt"
    ) -> str:
        try:
            with open(template_path, 'r', encoding='utf-8') as file:
                template = file.read().strip()
                # 直接替换模板变量并返回
                system_prompt = template.replace("{{from}}", from_lang)
                system_prompt = system_prompt.replace("{{to}}", to_lang)
                return system_prompt
        except FileNotFoundError:
            print(
                f"Error: System prompt template file not found {template_path}"
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"Error occurred while reading template file: {e}"
            )
            sys.exit(1)

    def translate(
        self,
        text: str,
        model: str = "gpt-3.5-turbo"
    ) -> None:
        stream_chat(self._client, text, model)


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: Please provide API key "
            "(via OPENAI_API_KEY environment variable)"
        )
        sys.exit(1)

    base_url = "https://api.deepseek.com"
    model = "deepseek-chat"

    zh_2_en = TranslationService(api_key, base_url, "中文", "英文")
    zh_2_en.translate(text="你好，世界！", model=model)
    zh_2_en.translate(text="你好吗", model=model)

    text_2 = """Translate to Chinese:
Hello, world!
How are you?
"""
    zh_2_en = TranslationService(api_key, base_url, "English", "Chinese")
    zh_2_en.translate(text=text_2, model=model)


if __name__ == "__main__":
    main()
