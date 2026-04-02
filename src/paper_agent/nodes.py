import os
import re
import base64
import requests
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

from paper_agent.state import PaperState

model = init_chat_model(
    model="google_genai:gemini-2.5-pro-preview-03-25",
    temperature=1.0,
)


def parser_node(state: PaperState):
    url = state.get("arxiv_url", "")
    print(f"🔍 [1/5] 解析 Arxiv 链接: {url} ...")

    match = re.search(r'(\d{4}\.\d{4,5})', url)
    if not match:
        print("⚠️ 未找到有效的 Arxiv ID。")
        return {"pdf_path": "未找到有效的 Arxiv ID"}

    arxiv_id = match.group(1)
    print(f"📄 提取到 Arxiv ID: {arxiv_id}，准备获取 PDF 全文...")

    try:
        output_dir = os.path.join("outputs", arxiv_id)
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f"{arxiv_id}.pdf")

        if os.path.exists(pdf_path):
            print(f"✅ 发现已下载的论文 PDF，跳过下载: {pdf_path}")
            return {"pdf_path": pdf_path}

        print("⬇️ 正在下载 PDF 全文...")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()

        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ 成功下载论文 PDF，保存至: {pdf_path}")
        return {"pdf_path": pdf_path}
    except Exception as e:
        print(f"❌ 下载论文出现错误: {e}")
        return {"pdf_path": f"下载错误: {e}"}


def researcher_node(state: PaperState):
    print("🔬 [2/5] 研究员正在提炼核心内容...")

    pdf_path = state.get("pdf_path", "")
    if not pdf_path or "错误" in pdf_path or "未找到" in pdf_path or not os.path.exists(pdf_path):
        return {"extracted_info": "无法提取内容：未获取到有效的论文 PDF。"}

    prompt = """
请作为一名专业的AI语音领域研究员，阅读这篇论文的 PDF，并提炼出以下三个核心部分：
1. 痛点：该论文试图解决的现有技术难题是什么？
2. 创新点：该论文提出了哪些核心创新方法或架构？
3. 实验结果：其主要物理实验结论和性能提升表现如何？

请严格按照以下格式输出：
痛点：...
创新点：...
实验结果：...
"""

    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:application/pdf;base64,{pdf_base64}"
                    }
                }
            ]
        )

        response = model.invoke([message])

        token_usage = state.get("token_usage", {})
        usage = getattr(response, "usage_metadata", {}) or getattr(response, "response_metadata", {}).get("token_usage", {})
        if usage:
            token_usage["Researcher"] = {
                "input_tokens": usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

        if isinstance(response.content, list):
            result = "\n".join([block.get("text") for block in response.content if isinstance(block, dict) and block.get("text")])
        else:
            result = response.content

        return {"extracted_info": result, "token_usage": token_usage}
    except Exception as e:
        print(f"❌ 研究员提炼核心内容失败: {e}")
        return {"extracted_info": f"请求大模型时发生错误: {e}"}


# def planner_node(state: PaperState):
#     print("📋 [3/5] 内容主编正在策划大纲...")
#     return {"outline": "1. 引入\n2. 痛点\n3. 解决方案\n4. 总结"}

# def writer_node(state: PaperState):
#     print("✍️ [4/5] 爆款撰稿人撰写初稿...")
#     return {"draft": "这篇最新的TTS论文彻底惊艳了我们..."}

# def reviewer_node(state: PaperState):
#     print("🧐 [5/5] 主编审阅润色...")
#     return {"final_article": "这篇最新的TTS论文彻底惊艳了我们...[Pass]"}

# def review_decision(state: PaperState):
#     if "Pass" in state.get("final_article", ""):
#         print("✅ 审阅通过！准备发布。")
#         return "end"
#     else:
#         print("❌ 审阅未通过，打回重写。")
#         return "rewrite"
