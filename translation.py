import os
import sys

from prompt_service import PromptService


class TranslationService:

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        from_lang: str = "中文",
        to_lang: str = "英文"
    ) -> None:
        template_vars = {
            "from": from_lang,
            "to": to_lang
        }

        self._prompt_service = PromptService(
            api_key=api_key,
            base_url=base_url,
            template_path="system-prompts/translation.txt",
            template_vars=template_vars
        )

    def translate(
        self,
        text: str,
        model: str = "gpt-3.5-turbo"
    ) -> None:
        self._prompt_service.stream_chat(text, model)


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

    zh_to_en = TranslationService(api_key, base_url, "中文", "英文")
    zh_to_en.translate(text="你好，世界！", model=model)
    zh_to_en.translate(text="你好吗", model=model)

    en_to_zh = TranslationService(api_key, base_url, "中文", "英文")
    text_2 = """Translate to Chinese:
Hello, world!
How are you?
"""
    en_to_zh.update_languages("English", "Chinese")
    en_to_zh.translate(text=text_2, model=model)


if __name__ == "__main__":
    main()
