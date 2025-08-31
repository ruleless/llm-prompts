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
        self._services = {
            "zh_to_en": None,
            "en_to_zh": None,
            "zh_to_en_multi": None,
            "sysprompt_opt": None,
            "manus_ai": None,
            "text_opt": None,
            "user_prompt_opt": None,
        }
        self._service_configs = {
            "zh_to_en": {
                "system_template_path": translation_prompt_file,
                "system_template_vars": {"to": "English"},
            },
            "en_to_zh": {
                "system_template_path": translation_prompt_file,
                "system_template_vars": {"to": "Chinese"},
            },
            "zh_to_en_multi": {
                "system_template_path": os.path.join(
                    rootdir, "system-prompts/zh_to_en.txt"
                ),
            },
            "sysprompt_opt": {
                "system_template_path": os.path.join(
                    rootdir, "system-prompts/sysprompt_opt.txt"
                ),
                "user_template_path": os.path.join(
                    rootdir, "user-prompts/sysprompt_opt.txt"
                ),
            },
            "manus_ai": {
                "system_template_path": os.path.join(
                    rootdir, "system-prompts/manus_ai.txt"
                ),
            },
            "text_opt": {
                "system_template_path": os.path.join(
                    rootdir, "system-prompts/text_opt.txt"
                ),
                "user_template_path": os.path.join(
                    rootdir, "user-prompts/text_opt.txt"
                ),
            },
            "user_prompt_opt": {
                "system_template_path": os.path.join(
                    rootdir, "system-prompts/user_prompt_opt.txt"
                ),
                "user_template_path": os.path.join(
                    rootdir, "user-prompts/user_prompt_opt.txt"
                ),
            },
        }

    def _ensure_service(self, service_name: str):
        """创建指定名称的服务实例"""
        if self._services[service_name] is None:
            api_key, base_url, model = validate_api_config()
            config = self._service_configs[service_name]
            self._services[service_name] = PromptService(
                api_key, base_url, model, **config
            )
        return self._services[service_name]

    @property
    def zh_to_en(self):
        return self._ensure_service("zh_to_en")

    @property
    def en_to_zh(self):
        return self._ensure_service("en_to_zh")

    @property
    def zh_to_en_multi(self):
        return self._ensure_service("zh_to_en_multi")

    @property
    def sysprompt_opt(self):
        return self._ensure_service("sysprompt_opt")

    @property
    def manus_ai(self):
        return self._ensure_service("manus_ai")

    @property
    def text_opt(self):
        return self._ensure_service("text_opt")

    @property
    def user_prompt_opt(self):
        return self._ensure_service("user_prompt_opt")


services = ServiceFactory()
