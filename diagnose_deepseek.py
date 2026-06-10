from __future__ import annotations

import json
import traceback
import urllib.error
import urllib.request

from config import settings


def main() -> None:
    if not settings.deepseek_api_key:
        print("DEEPSEEK_API_KEY 未配置。")
        return

    print(f"Base URL: {settings.deepseek_base_url}")
    print(f"Model: {settings.deepseek_model}")
    print("API Key: 已读取")

    print("\n[1/2] OpenAI SDK 测试")
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=20,
        )
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": "只回复 OK"}],
            temperature=0,
        )
        print("DeepSeek 调用成功：", response.choices[0].message.content)
    except Exception as exc:
        print(f"DeepSeek 调用失败：{type(exc).__name__}: {exc}")
        cause = getattr(exc, "__cause__", None)
        if cause:
            print(f"底层原因：{type(cause).__name__}: {cause}")

    print("\n[2/2] 原始 HTTP 测试")
    try:
        payload = json.dumps(
            {
                "model": settings.deepseek_model,
                "messages": [{"role": "user", "content": "只回复 OK"}],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            settings.deepseek_base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={
                "Authorization": "Bearer " + settings.deepseek_api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            print("原始 HTTP 调用成功：", response.read().decode("utf-8")[:500])
    except urllib.error.HTTPError as exc:
        print(f"原始 HTTP 调用失败：HTTP {exc.code}")
        print(exc.read().decode("utf-8", errors="replace")[:1000])
    except Exception as exc:
        print(f"原始 HTTP 调用失败：{type(exc).__name__}: {exc}")

    print("\n如果原始 HTTP 成功但 OpenAI SDK 失败，通常是 httpx/OpenAI SDK 与当前 Python 环境兼容问题。")
    print("如果两者都失败，请优先检查 API Key、模型名、DeepSeek 账号权限和网络。")


if __name__ == "__main__":
    main()
