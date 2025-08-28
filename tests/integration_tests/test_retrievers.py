from typing import Type

from langchain_goodmem.retrievers import GoodmemRetriever
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)


class TestGoodmemRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[GoodmemRetriever]:
        """Get an empty vectorstore for unit tests."""
        return GoodmemRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "example query"
