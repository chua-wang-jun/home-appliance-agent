import os
from abc import ABC, abstractmethod
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from utils.config_handler import rag_conf

load_dotenv()

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model_name=rag_conf["chat_model_name"],
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            timeout=60,
            max_retries=3,
        )

class EmbeddingsFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        # 当前兼容接口只接受字符串，需关闭默认的 token 分片提交
        return OpenAIEmbeddings(
            model=rag_conf["enbedding_model_name"],
            check_embedding_ctx_length=False,
            # 当前接口单次最多接收 25 条文本
            chunk_size=25,
        )


chat_model = ChatModelFactory().generator()
embeddings_model = EmbeddingsFactory().generator()
