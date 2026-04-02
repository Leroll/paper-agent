from dotenv import load_dotenv
load_dotenv()

import os
import re
from datetime import datetime

from paper_agent.graph import app


def build_token_summary(token_usage: dict) -> str:
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


def save_result(arxiv_url: str, content: str) -> None:
    match = re.search(r'(\d{4}\.\d{4,5})', arxiv_url)
    if not match:
        return
    arxiv_id = match.group(1)
    output_dir = os.path.join("outputs", arxiv_id)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{arxiv_id}_{timestamp}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ 结果已成功保存到: {output_path}")


if __name__ == "__main__":
    test_arxiv_url = "https://arxiv.org/abs/2411.09943"

    print("====== 启动论文分析 Agent... ======")
    initial_state = {
        "arxiv_url": test_arxiv_url,
        "pdf_path": "",
        "extracted_info": "",
        "outline": "",
        "draft": "",
        "final_article": "",
        "revision_count": 0,
        "token_usage": {}
    }

    result = app.invoke(initial_state)

    print("\n====== 最终产出摘要: ======")
    extracted_info = result.get("extracted_info", "")
    print(extracted_info)

    token_summary = build_token_summary(result.get("token_usage", {}))
    print(token_summary)

    if extracted_info:
        save_result(test_arxiv_url, extracted_info + token_summary)
