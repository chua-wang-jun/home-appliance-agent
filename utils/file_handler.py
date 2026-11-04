import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from docx import Document as WordDocument

def get_file_md5_hex(filepath: str) -> str | None:
    if not os.path.isfile(filepath):
        print(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj=hashlib.md5()
    chunk_size = 4096
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            md5_hex=md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"文件计算{filepath}md5失败：{str(e)}")
        return None

# 返回文件夹内指定后缀的文件列表
def listdir_with_allowed_type(path: str, allowed_type: tuple[str, ...]) -> list[str]:
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return files

    # 同时兼容 txt、.txt 和 .TXT，避免因配置格式或大小写漏掉文件
    allowed_suffixes = tuple(
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in allowed_type
    )
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path) and f.lower().endswith(allowed_suffixes):
            files.append(file_path)

    return files


def pdf_loader(filepath:str,passwd=None)->list[Document]:
    return PyPDFLoader(filepath, passwd).load()

def txt_loader(filepath:str)->list[Document]:
    # 知识库文档统一使用 UTF-8，避免 Windows 下被误判为 cp1252
    return TextLoader(filepath, encoding="utf-8").load()


def docx_loader(filepath: str) -> list[Document]:
    """读取 DOCX 的段落和表格，转换为统一的文本 Document。"""
    word_doc = WordDocument(filepath)
    parts = [paragraph.text.strip() for paragraph in word_doc.paragraphs if paragraph.text.strip()]
    for table in word_doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    text = "\n".join(parts).strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": filepath})]
