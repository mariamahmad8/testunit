"""Test embedding model integration."""

from typing import Type

from langchain_goodmem.embeddings import GoodmemEmbeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests


class TestParrotLinkEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[GoodmemEmbeddings]:
        return GoodmemEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
