# -*- coding: utf-8 -*-
import os
import requests
import json
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# 从环境变量中获取 API 地址和密钥
RAGFLOW_API_BASE = os.getenv("RAGFLOW_API_BASE")
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")

# 检查环境变量是否已设置
if not RAGFLOW_API_BASE or not RAGFLOW_API_KEY:
    print("错误：请在您的 .env 文件中设置 RAGFLOW_API_BASE 和 RAGFLOW_API_KEY。")
    exit(1)

# API 端点
# 移除 RAGFLOW_API_BASE 末尾可能存在的斜杠，以避免 URL 中出现双斜杠
url = f"{RAGFLOW_API_BASE.rstrip('/')}/api/v1/retrieval"

# 请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RAGFLOW_API_KEY}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

# 请求数据
data = {
    "question": "sai的故事",
    "dataset_ids": ["1f9f9e4aad7c11f0aa2efaceb254e4de"]
}

print(f"正在向地址发送请求: {url}")
print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

try:
    # 发送 POST 请求，并设置 30 秒超时
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()  # 如果状态码不是 2xx，则抛出异常

    # 打印成功的响应
    print("\n响应内容:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

except requests.exceptions.RequestException as e:
    print(f"\n请求出错: {e}")
    if e.response is not None:
        print(f"状态码: {e.response.status_code}")
        print(f"响应体: {e.response.text}")
