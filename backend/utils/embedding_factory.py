import os

import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from utils.embedding_config import EmbeddingConfig, EmbeddingProvider


dotenv.load_dotenv()


class EmbeddingFactory:
    @staticmethod
    def create_embedding_function(config: EmbeddingConfig):
        if config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )

        if config.provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbeddings(
                model_name=config.model_name,
            )

        raise ValueError(f"Unsupported embedding provider: {config.provider}")
