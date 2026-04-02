from typing import TypedDict


class PaperState(TypedDict):
    arxiv_url: str        # 输入的 arxiv 链接
    arxiv_id: str         # 从 URL 提取的 arxiv ID（如 2301.02111）
    pdf_path: str         # 下载的 pdf 文件路径
    file_uri: str         # Gemini Files API 上传后的 URI（上传一次，后续 node 复用）
    paper_info: dict      # 论文基本信息：标题、作者、arxiv 地址、实时引用量 badge
    extracted_info: str   # 研究员提取的核心要点
    outline: str          # 主编策划的大纲
    draft: str            # 撰稿人写的初稿
    final_article: str    # 终稿
    revision_count: int   # 修改或打回次数
    token_usage: dict     # Token 消耗统计
