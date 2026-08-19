import requests
import json
import time

def verify_llm_api(base_url, api_key, model_name, timeout=30):
    """
    验证 LLM API 是否可用
    :param base_url: 你的 API 地址（完整路径或基础地址）
    :param api_key: 你的 API Key
    :param model_name: 模型名称
    """
    
    # -------- 智能拼接 URL（如果你只填了基础地址，自动补全路径） --------
    if not base_url.endswith("/chat/completions"):
        # 如果用户填的是 https://api.openai.com/v1，就补上 /chat/completions
        if base_url.endswith("/v1") or base_url.endswith("/"):
            chat_url = base_url + "chat/completions"
        else:
            # 假设用户填了完整路径，直接使用
            chat_url = base_url
    else:
        chat_url = base_url

    # -------- 构造请求头 --------
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"   # 标准 OpenAI 鉴权
    }

    # -------- 构造最简单的对话请求 --------
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "请只回复 '连接成功' 这四个字，不要输出其他任何内容。"}
        ],
        "max_tokens": 20,
        "temperature": 0
    }

    print(f"🔍 正在测试: {chat_url}")
    print(f"🤖 模型: {model_name}")
    print("-" * 40)

    try:
        start_time = time.time()
        response = requests.post(chat_url, headers=headers, json=payload, timeout=timeout)
        elapsed = time.time() - start_time

        # -------- 1. 检查 HTTP 状态码 --------
        if response.status_code != 200:
            print(f"❌ HTTP 错误: {response.status_code}")
            # 尝试解析错误信息（OpenAI 标准错误格式）
            try:
                error_detail = response.json()
                print(f"📛 服务端返回: {json.dumps(error_detail, ensure_ascii=False, indent=2)}")
            except:
                print(f"📛 返回内容: {response.text[:300]}")
            
            # 常见错误排查提示
            if response.status_code == 401:
                print("💡 提示: 401 表示 API Key 无效或已过期，请检查 Key 是否正确")
            elif response.status_code == 404:
                print("💡 提示: 404 表示 URL 路径错误，请检查是否漏掉了 /v1/chat/completions")
            elif response.status_code == 429:
                print("💡 提示: 429 表示请求过频或余额不足")
            return False

        # -------- 2. 解析 JSON 并提取回复 --------
        result = response.json()
        
        # 检查是否有标准 OpenAI 结构
        if "choices" not in result or len(result["choices"]) == 0:
            print(f"❌ 返回结构异常，缺少 'choices': {result}")
            return False

        reply_content = result["choices"][0]["message"]["content"]
        
        # -------- 3. 验证内容（判断是否真的调通了模型） --------
        print(f"✅ HTTP 请求成功 (耗时: {elapsed:.2f}s)")
        print(f"💬 模型回复: {reply_content}")
        
        if "连接成功" in reply_content:
            print("🎉 验证完全通过！API 可用，模型正常响应。")
        else:
            print("⚠️ 接口通了，但模型没有按要求回复，可能是模型理解偏差，但 API 基本可用。")
        
        # 顺便打印一下用量（如果返回了）
        if "usage" in result:
            usage = result["usage"]
            print(f"📊 Token 用量: prompt={usage.get('prompt_tokens',0)}, completion={usage.get('completion_tokens',0)}")
        
        return True

    except requests.exceptions.Timeout:
        print(f"❌ 请求超时 (>{timeout}秒)，请检查网络或增大 timeout 参数")
    except requests.exceptions.ConnectionError:
        print("❌ 网络连接失败，请检查 URL 是否正确（注意 http/https）")
    except json.JSONDecodeError:
        print(f"❌ 返回内容不是合法 JSON: {response.text[:200]}")
    except Exception as e:
        print(f"❌ 未知异常: {e}")
    
    return False


# ==================== 👇 在这里填写你的三要素 ====================

API_URL = "https://api.deepseek.com/v1"   # 替换成你的 URL
API_KEY = "sk-0d80b12eb8af4a48ba83a373b52554f3"  # 替换成你的 Key
MODEL_NAME = "deepseek-v4-flash"                             # 替换成你的模型名

# ============================================================

if __name__ == "__main__":
    verify_llm_api(API_URL, API_KEY, MODEL_NAME)