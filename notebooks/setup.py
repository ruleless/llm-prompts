import os
import sys

rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(rootdir)

from dotenv import load_dotenv
from chat_client import ChatClient
from prompt_service import PromptService

load_dotenv()
translation_prompt_file = os.path.join(rootdir, "system-prompts/translation.txt")


class ServiceFactory:
    def __init__(self):
        self._services = {
            "pure_chat": None,
            "zh_to_en": None,
            "en_to_zh": None,
            "zh_to_en_multi": None,
            "en_zh_to_multi": None,
            "sysprompt_opt": None,
            "manus_ai": None,
            "text_opt": None,
            "user_prompt_opt": None,
            "programming": None,
        }
        self._service_configs = {
            "pure_chat": {},
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
            "en_zh_to_multi": {
                "system_template_path": os.path.join(
                    rootdir, "system-prompts/en_to_zh.txt"
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
            "programming": {
                "system_template_path": os.path.join(
                    rootdir, "system-prompts/programming.txt"
                ),
            },
        }

    def _ensure_service(self, service_name: str):
        """创建指定名称的服务实例"""
        if self._services[service_name] is None:
            config = self._service_configs[service_name]

            # 创建 OpenAI 客户端实例
            client: ChatClient = None
            model: str = "glm-4.5"
            if os.getenv("ZHIPUAI_API_KEY"):
                from zhipuai_chat_client import ZhipuAiChatClient
                api_key = os.getenv("ZHIPUAI_API_KEY")
                # base_url = "https://open.bigmodel.cn/api/paas/v4/"
                model = "glm-4.5"
                client = ZhipuAiChatClient(
                    api_key=api_key,
                    # base_url=base_url,
                )
            elif os.getenv("DEEPSEEK_API_KEY"):
                base_url = "https://api.deepseek.com"
                model = "deepseek-chat"
                from openai_chat_client import OpenAIClient
                client = OpenAIClient(
                    api_key=api_key,
                    base_url=base_url,
                )
            else:
                print("Error: API key not provided")
                sys.exit(1)

            # 使用客户端实例创建 PromptService
            self._services[service_name] = PromptService(
                client=client,
                model=model,
                **config
            )
        return self._services[service_name]

    @property
    def pure_chat(self):
        return self._ensure_service("pure_chat")

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
    def en_zh_to_multi(self):
        return self._ensure_service("en_zh_to_multi")

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

    @property
    def programming(self):
        return self._ensure_service("programming")


services = ServiceFactory()
