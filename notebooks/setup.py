import os
import sys

rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(rootdir)

from dotenv import load_dotenv
from prompt_service import PromptService

load_dotenv()
translation_prompt_file = os.path.join(rootdir, "system-prompts/translation.txt")


def validate_api_config():
    """验证API配置并返回配置信息"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please provide API key(via OPENAI_API_KEY environment variable)")
        sys.exit(1)

    base_url = "https://api.deepseek.com"
    model = "deepseek-chat"

    return api_key, base_url, model


class ServiceFactory:
    def __init__(self):
        self._zh_to_en_service = None
        self._en_to_zh_service = None
        self._zh_to_en_multi_service = None
        self._prompt_opt_service = None

    @property
    def zh_to_en(self):
        if self._zh_to_en_service is None:
            api_key, base_url, model = validate_api_config()
            self._zh_to_en_service = PromptService(
                api_key,
                base_url,
                model,
                system_template_path=translation_prompt_file,
                system_template_vars={"to": "English"},
            )
        return self._zh_to_en_service

    @property
    def en_to_zh(self):
        if self._en_to_zh_service is None:
            api_key, base_url, model = validate_api_config()
            self._en_to_zh_service = PromptService(
                api_key,
                base_url,
                model,
                system_template_path=translation_prompt_file,
                system_template_vars={"to": "Chinese"},
            )
        return self._en_to_zh_service

    @property
    def zh_to_en_multi(self):
        if self._zh_to_en_multi_service is None:
            api_key, base_url, model = validate_api_config()
            self._zh_to_en_multi_service = PromptService(
                api_key,
                base_url,
                model,
                system_template_path=os.path.join(
                    rootdir, "system-prompts/zh_to_en.txt"
                ),
            )
        return self._zh_to_en_multi_service

    @property
    def prompt_opt(self):
        if self._prompt_opt_service is None:
            api_key, base_url, model = validate_api_config()
            self._prompt_opt_service = PromptService(
                api_key,
                base_url,
                model,
                system_template_path=os.path.join(
                    rootdir, "system-prompts/prompt_opt.txt"
                ),
            )
        return self._prompt_opt_service


services = ServiceFactory()
