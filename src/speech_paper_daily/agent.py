import os
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import ArxivLoader

from dotenv import load_dotenv
load_dotenv()

# ================================
# 1. 定义状态 (State)
# ================================
class PaperState(TypedDict):
    arxiv_url: str        # 输入的 arxiv 链接
    paper_text: str       # 原始论文文本 (后续实现 PDF 解析填入)
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
        return {"paper_text": "未找到有效的 Arxiv ID"}
        
    arxiv_id = match.group(1)
    print(f"📄 提取到 Arxiv ID: {arxiv_id}，正在下载并解析PDF大纲与全文...")
    
    # 使用 ArxivLoader 加载论文完整文本
    try:
        loader = ArxivLoader(query=arxiv_id, load_max_docs=1, load_all_available_meta=True)
        docs = loader.load()
        
        if not docs:
            print("⚠️ 未获取到论文内容。")
            return {"paper_text": "未获取到论文内容"}
        
        # 将文本存入状态
        paper_text = docs[0].page_content
        print(f"✅ 成功获取论文文本，长度: {len(paper_text)} 字符")
        return {"paper_text": paper_text}
    except Exception as e:
        print(f"❌ 解析论文出现错误: {e}")
        return {"paper_text": f"解析错误: {e}"}

def researcher_node(state: PaperState):
    print("🔬 [2/5] 研究员正在提炼核心内容...")
    # TODO: 调用大模型总结重点
    return {"extracted_info": "痛点：xxx\n创新点：xxx\n实验结果：xxx"}

def planner_node(state: PaperState):
    print("📋 [3/5] 内容主编正在策划大纲...")
    # TODO: 定义大纲逻辑
    return {"outline": "1. 引入\n2. 痛点\n3. 解决方案\n4. 总结"}

def writer_node(state: PaperState):
    print("✍️ [4/5] 爆款撰稿人撰写初稿...")
    # TODO: 扩写生成公众号初稿
    return {"draft": "这篇最新的TTS论文彻底惊艳了我们..."}

def reviewer_node(state: PaperState):
    print("🧐 [5/5] 主编审阅润色...")
    # TODO: AI打分和润色
    return {"final_article": "这篇最新的TTS论文彻底惊艳了我们...[Pass]"}

def review_decision(state: PaperState):
    # TODO: 判断是否需要重写 (依据关键词或者打分)
    # 模拟检查
    if "Pass" in state.get("final_article", ""):
        print("✅ 审阅通过！准备发布。")
        return "end"
    else:
        print("❌ 审阅未通过，打回重写。")
        return "rewrite"

# ================================
# 3. 编排工作流 (Graph)
# ================================
workflow = StateGraph(PaperState)

workflow.add_node("Parser", parser_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Planner", planner_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("Reviewer", reviewer_node)

# 定义流转顺序
workflow.add_edge("Parser", "Researcher")
workflow.add_edge("Researcher", "Planner")
workflow.add_edge("Planner", "Writer")
workflow.add_edge("Writer", "Reviewer")

# 定义条件分支
workflow.add_conditional_edges(
    "Reviewer", 
    review_decision, 
    {
        "end": END, 
        "rewrite": "Writer"
    }
)

# 设定入口
workflow.set_entry_point("Parser")

# 编译成可执行的图
app = workflow.compile()

# ================================
# 4. 执行测试
# ================================
if __name__ == "__main__":
    print("🚀 启动论文分析 Agent...")
    initial_state = {
        "arxiv_url": "https://arxiv.org/abs/2301.02111", # 测试用 URL: VALL-E
        "paper_text": "",
        "extracted_info": "",
        "outline": "",
        "draft": "",
        "final_article": "",
        "revision_count": 0
    }
    
    # 执行图
    result = app.invoke(initial_state)
    print("\n🎉 最终产出摘要:")
    print(result.get("final_article")[:50] + "...")
