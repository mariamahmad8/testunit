"""Test Goodmem embeddings."""

from typing import Type

from langchain_goodmem.embeddings import GoodmemEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[GoodmemEmbeddings]:
        return GoodmemEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
