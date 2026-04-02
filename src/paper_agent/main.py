from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
from datetime import datetime

from paper_agent.graph import app


def save_result(arxiv_url: str, result: dict) -> None:
    match = re.search(r'(\d{4}\.\d{4,5})', arxiv_url)
    if not match:
        return
    arxiv_id = match.group(1)
    output_dir = os.path.join("outputs", arxiv_id)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存最终文章
    final_article = result.get("final_article", "")
    if final_article:
        md_path = os.path.join(output_dir, f"{arxiv_id}_{timestamp}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_article)
        print(f"✅ 文章已保存至: {md_path}")

    # 保存 state JSON（排除过期的 file_uri）
    state_dump = {k: v for k, v in result.items() if k != "file_uri"}
    json_path = os.path.join(output_dir, f"{arxiv_id}_{timestamp}_state.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(state_dump, f, ensure_ascii=False, indent=2)
    print(f"✅ State 已保存至: {json_path}")


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
    final_article = result.get("final_article", "")
    # 打印前 800 字预览
    preview = final_article[:800] + ("..." if len(final_article) > 800 else "")
    print(preview)

    if result:
        save_result(test_arxiv_url, result)
