#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from openai_chat_client import OpenAIClient, print_models, stream_chat

def print_available_models(client):
    """打印可用模型列表"""
    print("=== Listing available models ===")
    models = client.list_models()
    if models:
        print_models(models)
    else:
        print("Unable to get model list")

def start_conversation(client):
    """开始对话"""
    print("\n=== Starting conversation ===")

    client.clear_history()

    print("\nUser: Hello, how are you?")
    stream_chat(client, "Hello, how are you?", "deepseek-chat")

    print("\nUser: What can you help me with?")
    stream_chat(client, "What can you help me with?", "deepseek-chat")

    print("\n=== Conversation completed ===")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    client = OpenAIClient(api_key, base_url="https://api.deepseek.com")

    # 调用打印模型函数
    # print_available_models(client)

    # 调用对话函数
    start_conversation(client)

if __name__ == "__main__":
    main()
