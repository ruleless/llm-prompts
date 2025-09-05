#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from zhipuai_chat_client import ZhipuAiChatClient
from chat_client import print_models, stream_chat
from dotenv import load_dotenv

load_dotenv()


def print_available_models(client: ZhipuAiChatClient):
    """打印可用模型列表"""
    print("=== Listing available models ===")
    models = client.list_models()
    if models:
        print_models(models)
    else:
        print("Unable to get model list")


def start_conversation(client: ZhipuAiChatClient):
    """开始对话"""
    print("\n=== Starting conversation ===")

    client.clear_history()

    print("\nUser: Hello, how are you?")
    stream_chat(client, "Hello, how are you?", "glm-4.5")

    print("\nUser: What can you help me with?")
    stream_chat(client, "What can you help me with?", "glm-4.5")

    print("\n=== Conversation completed ===")


def main():
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        print("Error: Please set ZHIPUAI_API_KEY environment variable")
        return

    client = ZhipuAiChatClient(api_key)

    # 调用打印模型函数
    # print_available_models(client)

    # 调用对话函数
    start_conversation(client)


if __name__ == "__main__":
    main()
