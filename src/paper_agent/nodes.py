import os
import re
import urllib.parse
import requests
import arxiv
from google import genai
from google.genai import types as genai_types
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

from paper_agent.state import PaperState

_genai_client = genai.Client()

model = init_chat_model(
    model="google_genai:gemini-3.1-pro-preview",
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
        else:
            print("⬇️ 正在下载 PDF 全文...")
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()

            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✅ 成功下载论文 PDF，保存至: {pdf_path}")

        # 上传到 Gemini Files API（上传一次，所有后续 node 通过 file_uri 引用，避免重复 base64 传输）
        print("☁️ 正在上传 PDF 到 Gemini Files API...")
        uploaded_file = _genai_client.files.upload(
            file=pdf_path,
            config=genai_types.UploadFileConfig(mime_type="application/pdf"),
        )
        print(f"✅ 上传成功，file_uri: {uploaded_file.uri}")

        return {"arxiv_id": arxiv_id, "pdf_path": pdf_path, "file_uri": uploaded_file.uri}
    except Exception as e:
        print(f"❌ 解析论文出现错误: {e}")
        return {"pdf_path": f"解析错误: {e}", "arxiv_id": arxiv_id, "file_uri": ""}


def info_node(state: PaperState):
    """通过 arxiv 包获取论文元数据，构建实时引用量 badge，不消耗任何 LLM token。"""
    print("📋 [2/5] 获取论文基本信息...")

    arxiv_id = state.get("arxiv_id", "")
    if not arxiv_id:
        print("⚠️ 未找到 arxiv_id，跳过 info_node。")
        return {"paper_info": {}}

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(arxiv.Client().results(search))

        title = result.title
        authors = ", ".join(a.name for a in result.authors)
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

        # shields.io 动态 badge：每次渲染 md 文件时实时从 Semantic Scholar 拉取最新引用量
        ss_api = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}?fields=citationCount"
        badge_url = (
            "https://img.shields.io/badge/dynamic/json"
            f"?url={urllib.parse.quote(ss_api, safe='')}"
            "&query=%24.citationCount"
            "&label=Citations"
            "&color=blue"
        )
        citation_badge = f"![Citations]({badge_url})"

        paper_info = {
            "title": title,
            "authors": authors,
            "arxiv_url": arxiv_url,
            "citation_badge": citation_badge,
        }
        print(f"✅ 论文信息获取成功: {title}")
        return {"paper_info": paper_info}
    except Exception as e:
        print(f"❌ 获取论文信息失败: {e}")
        return {"paper_info": {}}


def researcher_node(state: PaperState):
    print("🔬 [3/5] 研究员正在提炼核心内容...")

    file_uri = state.get("file_uri", "")
    pdf_path = state.get("pdf_path", "")
    if not file_uri or not pdf_path or "错误" in pdf_path or "未找到" in pdf_path:
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
        # 直接用 google-genai SDK 引用 Files API URI，避免 LangChain 将其当普通 URL 下载（会 403）
        response = _genai_client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[
                genai_types.Content(parts=[
                    genai_types.Part.from_uri(file_uri=file_uri, mime_type="application/pdf"),
                    genai_types.Part.from_text(text=prompt),
                ])
            ],
            config=genai_types.GenerateContentConfig(temperature=1.0),
        )

        token_usage = state.get("token_usage", {})
        usage = response.usage_metadata
        if usage:
            token_usage["Researcher"] = {
                "input_tokens": usage.prompt_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0,
                "total_tokens": usage.total_token_count or 0,
            }

        return {"extracted_info": response.text, "token_usage": token_usage}
    except Exception as e:
        print(f"❌ 研究员提炼核心内容失败: {e}")
        return {"extracted_info": f"请求大模型时发生错误: {e}"}


def _build_token_summary(token_usage: dict) -> str:
    summary = "\n\n---\n## 📊 Token 消耗统计\n\n"
    if not token_usage:
        return summary + "暂无 Token 消耗数据。\n"

    summary += "| 节点 | 输入 Tokens | 输出 Tokens | 总计 Tokens |\n"
    summary += "|---|---|---|---|\n"
    total_in = total_out = total = 0
    for node, usage in token_usage.items():
        in_t = usage.get("input_tokens", 0)
        out_t = usage.get("output_tokens", 0)
        tot_t = usage.get("total_tokens", 0) or in_t + out_t
        total_in += in_t
        total_out += out_t
        total += tot_t
        summary += f"| {node} | {in_t} | {out_t} | {tot_t} |\n"
    summary += f"| **总计** | **{total_in}** | **{total_out}** | **{total}** |\n"
    return summary


def writer_node(state: PaperState):
    print("✍️ [4/5] 撰稿人正在撰写完整解读文章...")

    extracted_info = state.get("extracted_info", "")
    paper_info = state.get("paper_info", {})

    if not extracted_info:
        return {"final_article": f"无法生成文章：核心内容提炼失败。\n\n详情：{extracted_info}"}

    title = paper_info.get("title", "未知论文标题")
    authors = paper_info.get("authors", "")
    arxiv_url = paper_info.get("arxiv_url", "")
    citation_badge = paper_info.get("citation_badge", "")

    prompt = f"""你是一位擅长深入浅出讲解AI论文的技术写作者。请基于以下论文信息，撰写一篇完整的中文解读文章。

论文英文标题：{title}

论文核心要点（由研究员提炼）：
{extracted_info}

---

**写作要求：**
1. 文章第一行必须是 `# ` 开头的一句吸引人的中文标题（不得照搬英文标题翻译，要体现论文亮点）
2. 标题之后直接进入正文，**不要写任何作者、链接、引用数等元数据**（这部分由程序自动插入标题后）
3. 文章结构建议（可灵活调整）：
   - 一句话总结（用一段话点明这篇论文最值得关注的地方）
   - 背景与痛点（现有方法有什么问题，这篇论文要解决什么）
   - 核心创新点（方法/架构/训练策略有什么新颖之处，用类比或具体例子帮助读者理解）
   - 实验结论（关键指标提升了多少，和哪些 baseline 对比，结论是什么）
   - 总结与展望（这项工作的意义，未来方向）
4. 风格：深入浅出，讲清楚技术原理，不堆砌术语，适当用类比和具体例子帮助理解，不刻意追求标题党
5. 语言：中文，Markdown 格式，可以使用加粗、表格、代码块等丰富排版
"""

    try:
        file_uri = state.get("file_uri", "")
        contents = []
        if file_uri:
            contents.append(
                genai_types.Content(parts=[
                    genai_types.Part.from_uri(file_uri=file_uri, mime_type="application/pdf"),
                    genai_types.Part.from_text(text=prompt),
                ])
            )
        else:
            contents.append(
                genai_types.Content(parts=[genai_types.Part.from_text(text=prompt)])
            )

        response = _genai_client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=contents,
            config=genai_types.GenerateContentConfig(temperature=1.0),
        )

        token_usage = dict(state.get("token_usage", {}))
        usage = response.usage_metadata
        if usage:
            token_usage["Writer"] = {
                "input_tokens": usage.prompt_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0,
                "total_tokens": usage.total_token_count or 0,
            }

        llm_output = response.text.strip()

        # 拆分 H1 标题与正文
        lines = llm_output.splitlines()
        if lines and lines[0].startswith("# "):
            h1 = lines[0]
            article_body = "\n".join(lines[1:]).lstrip("\n")
        else:
            h1 = f"# {title}"
            article_body = llm_output

        # 程序拼接元数据 blockquote
        metadata_lines = [f"> **论文标题**：{title}"]
        if authors:
            metadata_lines.append(f">\n> **作者**：{authors}")
        if arxiv_url:
            badge_str = f"  {citation_badge}" if citation_badge else ""
            metadata_lines.append(f">\n> **论文链接**：{arxiv_url}{badge_str}")
        metadata_block = "\n".join(metadata_lines)

        token_table = _build_token_summary(token_usage)

        final_article = f"{h1}\n\n{metadata_block}\n\n---\n\n{article_body}{token_table}"

        print("✅ 文章撰写完成。")
        return {"final_article": final_article, "token_usage": token_usage}

    except Exception as e:
        print(f"❌ 撰稿人生成文章失败: {e}")
        return {"final_article": f"生成文章时发生错误: {e}"}