from pathlib import Path
import threading
import os
"""
总结服务：将用户提问和参考资料一起发给大模型，让他总结
"""
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factlory import chat_model
from langchain_core.documents import Document

def print_prompt(prompt):
    """仅在显式开启调试时打印完整提示词，避免泄露内部指令。"""
    if os.getenv("DEBUG_RAG_PROMPT", "0") == "1":
        print("=" * 60)
        print(prompt.to_string())
        print("=" * 60)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever=self.vector_store.get_retriever()
        self.prompt_text=load_rag_prompts()
        self.prompt_template=PromptTemplate.from_template(self.prompt_text)
        self.model=chat_model
        self.chain=self._init_chain()
        self._sources: list[str] = []
        self._sources_lock = threading.Lock()

    def reset_sources(self) -> None:
        with self._sources_lock:
            self._sources = []

    def get_sources(self) -> list[str]:
        with self._sources_lock:
            return list(self._sources)

    def _record_sources(self, documents: list[Document]) -> None:
        names = []
        for doc in documents:
            source = doc.metadata.get("source") if doc.metadata else None
            if source:
                name = Path(str(source)).name
                if name and name not in names:
                    names.append(name)
        if not names:
            return
        with self._sources_lock:
            for name in names:
                if name not in self._sources:
                    self._sources.append(name)

    def _init_chain(self):
        chain=self.prompt_template | print_prompt | self.model |StrOutputParser()
        return chain

    def retriever_docs(self,query:str)->list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self,query:str)->str:
        context_docs=self.retriever_docs(query)
        self._record_sources(context_docs)
        context=" "
        counter=0
        for doc in context_docs:
            counter+=1
            context+=f"[参考资料{counter}]：{doc.page_content}| 参考元数据：{doc.metadata}\n"
        return self.chain.invoke(
            {
                "input":query,
                "context":context
            }
        )


if __name__ == '__main__':
    rag=RagSummarizeService()
    print(rag.rag_summarize("净水器的使用说明"))
