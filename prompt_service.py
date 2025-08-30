#!/usr/bin/env python3

import os
import sys
from typing import Dict, Optional

from openai_chat_client import OpenAIClient, stream_chat


class PromptService:

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        template_path: Optional[str] = None,
        template_vars: Optional[Dict[str, str]] = None
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.template_path = template_path
        self.template_vars = template_vars or {}

        system_prompt = self._load_and_process_template()
        self._client = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt
        )

    def _load_and_process_template(self) -> Optional[str]:
        if not self.template_path:
            return None

        try:
            with open(self.template_path, 'r', encoding='utf-8') as file:
                template = file.read().strip()
                return self._replace_template_variables(template)
        except FileNotFoundError:
            print(
                f"Error: System prompt template file not found {self.template_path}"
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"Error occurred while reading template file: {e}"
            )
            sys.exit(1)

    def _replace_template_variables(self, template: str) -> str:
        result = template
        for key, value in self.template_vars.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result

    def update_template_vars(self, new_vars: Dict[str, str]) -> None:
        self.template_vars.update(new_vars)
        if self.template_path:
            system_prompt = self._load_and_process_template()
            self._client.set_system_prompt(system_prompt)

    def stream_chat(
        self,
        text: str,
        model: str = "gpt-3.5-turbo"
    ) -> None:
        stream_chat(self._client, text, model)
