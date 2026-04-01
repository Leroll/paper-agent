from dotenv import load_dotenv
load_dotenv()

import os
import re
import base64
import requests
from datetime import datetime
from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# ================================
# 1. 定义状态 (State)
# ================================
model = init_chat_model(
    model="google_genai:gemini-3.1-pro-preview",
    temperature=1.0, 
)



# ================================
# 1. 定义状态 (State)
# ================================
class PaperState(TypedDict):
    arxiv_url: str        # 输入的 arxiv 链接
    pdf_path: str         # 下载的 pdf 文件路径
    extracted_info: str   # 研究员提取的核心要点
    outline: str          # 主编策划的大纲
    draft: str            # 撰稿人写的初稿
    final_article: str    # 终稿
    revision_count: int   # 修改或打回次数

# ================================
# 2. 定义节点 (Nodes)
# ================================
def parser_node(state: PaperState):
    url = state.get("arxiv_url", "")
    print(f"🔍 [1/5] 解析 Arxiv 链接: {url} ...")
    
    # 提取 Arxiv ID
    match = re.search(r'(\d{4}\.\d{4,5})', url)
    if not match:
        print("⚠️ 未找到有效的 Arxiv ID。")
        return {"pdf_path": "未找到有效的 Arxiv ID"}
        
    arxiv_id = match.group(1)
    print(f"📄 提取到 Arxiv ID: {arxiv_id}，正在下载 PDF 全文...")
    
    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        output_dir = os.path.join("outputs", arxiv_id)
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
        
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
        
        # 兼容文本或包含块的返回格式
        if isinstance(response.content, list):
            result = "\n".join([block.get("text") for block in response.content if isinstance(block, dict) and block.get("text")])
        else:
            result = response.content
            
        return {"extracted_info": result}
    except Exception as e:
        print(f"❌ 研究员提炼核心内容失败: {e}")
        return {"extracted_info": f"请求大模型时发生错误: {e}"}

# def planner_node(state: PaperState):
#     print("📋 [3/5] 内容主编正在策划大纲...")
#     # TODO: 定义大纲逻辑
#     return {"outline": "1. 引入\n2. 痛点\n3. 解决方案\n4. 总结"}

# def writer_node(state: PaperState):
#     print("✍️ [4/5] 爆款撰稿人撰写初稿...")
#     # TODO: 扩写生成公众号初稿
#     return {"draft": "这篇最新的TTS论文彻底惊艳了我们..."}

# def reviewer_node(state: PaperState):
#     print("🧐 [5/5] 主编审阅润色...")
#     # TODO: AI打分和润色
#     return {"final_article": "这篇最新的TTS论文彻底惊艳了我们...[Pass]"}

# def review_decision(state: PaperState):
#     # TODO: 判断是否需要重写 (依据关键词或者打分)
#     # 模拟检查
#     if "Pass" in state.get("final_article", ""):
#         print("✅ 审阅通过！准备发布。")
#         return "end"
#     else:
#         print("❌ 审阅未通过，打回重写。")
#         return "rewrite"

# ================================
# 3. 编排工作流 (Graph)
# ================================
workflow = StateGraph(PaperState)

workflow.add_node("Parser", parser_node)
workflow.add_node("Researcher", researcher_node)
# workflow.add_node("Planner", planner_node)
# workflow.add_node("Writer", writer_node)
# workflow.add_node("Reviewer", reviewer_node)

# 定义流转顺序
workflow.add_edge("Parser", "Researcher")
# workflow.add_edge("Researcher", "Planner")
# workflow.add_edge("Planner", "Writer")
# workflow.add_edge("Writer", "Reviewer")

# 定义条件分支
# workflow.add_conditional_edges(
#     "Reviewer", 
#     review_decision, 
#     {
#         "end": END, 
#         "rewrite": "Writer"
#     }
# )

# 设定入口
workflow.set_entry_point("Parser")

# 编译成可执行的图
app = workflow.compile()

# ================================
# 4. 执行测试
# ================================
if __name__ == "__main__":
    print("====== 启动论文分析 Agent... ======")
    test_arxiv_url = "https://arxiv.org/abs/2301.02111" # 测试用 URL: VALL-E
    initial_state = {
        "arxiv_url": test_arxiv_url,
        "pdf_path": "",
        "extracted_info": "",
        "outline": "",
        "draft": "",
        "final_article": "",
        "revision_count": 0
    }
    
    # 执行图
    result = app.invoke(initial_state)
    print("\n====== 最终产出摘要: ======")
    extracted_info = result.get("extracted_info", "")
    print(extracted_info)
    
    # 保存结果到 outputs 文件夹
    match = re.search(r'(\d{4}\.\d{4,5})', test_arxiv_url)
    if match and extracted_info:
        arxiv_id = match.group(1)
        output_dir = os.path.join("outputs", arxiv_id)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{arxiv_id}_{timestamp}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(extracted_info)
        print(f"\n✅ 结果已成功保存到: {output_path}")
