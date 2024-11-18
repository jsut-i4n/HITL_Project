from agentuniverse.agent.action.knowledge.embedding.dashscope_embedding import DashscopeEmbedding
from langchain_community.embeddings import OllamaEmbeddings
from agentuniverse.agent.action.knowledge.knowledge import Knowledge
from agentuniverse.agent.action.knowledge.reader.file.pdf_reader import PdfReader
from agentuniverse.agent.action.knowledge.store.chroma_store import ChromaStore
from agentuniverse.agent.action.knowledge.store.document import Document
from langchain.text_splitter import TokenTextSplitter
from pathlib import Path

from typing import Any

import toml


def get_from_toml(config_file: str, section: str, key: str) -> Any | None:
    try:
        with open(config_file, 'r', encoding='utf-8') as file:
            config = toml.load(file)
        section_data = config.get(section)
        if section_data is not None:
            return section_data.get(key)
        return None
    except FileNotFoundError:
        print(f"配置文件 {config_file} 未找到")
        return None
    except toml.TomlDecodeError:
        print(f"无法解析配置文件 {config_file}")
        return None


SPLITTER = TokenTextSplitter(chunk_size=600, chunk_overlap=100)


class TheDreamOfRedMansions(Knowledge):
    """The demo knowledge."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = get_from_toml('../../config/custom_key.toml', 'KEY_LIST', 'DASHSCOPE_API_KEY')
        self.store = ChromaStore(
            collection_name="dream_of_red_mansions",
            persist_path="../../DB/dream_of_red_mansions.db",
            embedding_model=DashscopeEmbedding(
                embedding_model_name='text-embedding-v2',
                dashscope_api_key=api_key
            ),
            dimensions=1536)
        self.reader = PdfReader()
        # Initialize the knowledge
        # self.insert_knowledge()

    def insert_knowledge(self, **kwargs) -> None:
        """
        Load pdf and save into vector database.
        """
        the_dream_of_red_mansions = self.reader.load_data(Path("../../resources/红楼梦.pdf"))
        lc_doc_list = SPLITTER.split_documents(Document.as_langchain_list(
            the_dream_of_red_mansions
        ))
        self.store.insert_documents(Document.from_langchain_list(lc_doc_list))


