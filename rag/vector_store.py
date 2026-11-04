import sys
import warnings

from pathlib import Path

# 直接运行本文件时，将项目根目录加入模块搜索路径
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factlory import embeddings_model

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
import os
from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
    docx_loader,
)
from utils.logger_handler import logger

class VectorStoreService:
    def __init__(self):
        # 相对路径按项目根目录解析，避免启动目录不同导致数据库位置变化
        persist_directory = get_abs_path(chroma_conf["persist_directory"])
        os.makedirs(persist_directory, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            persist_directory=persist_directory,
            embedding_function=embeddings_model,
        )
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_source_chunk_counts(self) -> dict[str, int]:
        """统计每个源文件当前在 Chroma 中的切分块数量。"""
        try:
            result = self.vector_store._collection.get(include=["metadatas"])
        except Exception:
            logger.warning("[知识库]读取切分块统计失败", exc_info=True)
            return {}

        counts: dict[str, int] = {}
        for metadata in result.get("metadatas", []) or []:
            if not isinstance(metadata, dict):
                continue
            source = metadata.get("source")
            if source:
                normalized = str(Path(str(source)).resolve())
                counts[normalized] = counts.get(normalized, 0) + 1
        return counts

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": chroma_conf["k"],
                "score_threshold": chroma_conf["score_threshold"],
            },
        )

    def _check_md5_hex(self, md5_for_check: str) -> bool:
        md5_path = get_abs_path(chroma_conf["md5_hex_store"])
        if not os.path.exists(md5_path):
            return False
        with open(md5_path, "r", encoding="utf-8") as f:
            return any(md5_for_check == line.strip() for line in f)

    def _save_md5_hex(self, md5_for_check: str) -> None:
        md5_path = get_abs_path(chroma_conf["md5_hex_store"])
        os.makedirs(os.path.dirname(md5_path), exist_ok=True)
        with open(md5_path, "a", encoding="utf-8") as f:
            f.write(md5_for_check + "\n")

    def rebuild(self) -> int:
        """清空当前集合和 MD5 记录，然后重新加载全部知识库文档。"""
        md5_path = get_abs_path(chroma_conf["md5_hex_store"])
        if os.path.exists(md5_path):
            os.remove(md5_path)
        self.vector_store.reset_collection()
        return self.load_document()

    @staticmethod
    def _get_file_documents(read_path: str) -> list[Document]:
        # 根据文件后缀选择对应的文档加载器
        if read_path.lower().endswith(".txt"):
            return txt_loader(read_path)
        if read_path.lower().endswith(".pdf"):
            return pdf_loader(read_path)
        if read_path.lower().endswith(".docx"):
            return docx_loader(read_path)
        return []

    def load_document(self) -> int:
        """加载知识库目录中的文档，返回本次成功入库的文件数。"""
        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )
        if not allowed_files_path:
            logger.warning("[加载知识库]未找到可加载的文档")
            return 0

        loaded_count = 0
        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                logger.warning(f"[加载知识库]{path}无法计算MD5，已跳过")
                continue
            if self._check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库中")
                continue

            try:
                documents = self._get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内容没有有效文本")
                    continue

                split_document = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本")
                    continue
                self.vector_store.add_documents(split_document)
                self._save_md5_hex(md5_hex)
                loaded_count += 1
                logger.info(f"[加载知识库]{path}内容成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
        return loaded_count

if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res=retriever.invoke("反渗透")
    for doc in res:
        print(doc.page_content)
        print("="*60)
