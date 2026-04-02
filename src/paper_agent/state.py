from typing import TypedDict


class PaperState(TypedDict):
    arxiv_url: str        # 输入的 arxiv 链接
    pdf_path: str         # 下载的 pdf 文件路径
    extracted_info: str   # 研究员提取的核心要点
    outline: str          # 主编策划的大纲
    draft: str            # 撰稿人写的初稿
    final_article: str    # 终稿
    revision_count: int   # 修改或打回次数
    token_usage: dict     # Token 消耗统计
