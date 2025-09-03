import os
from typing import Generator

import pytest
from langchain_goodmem.vectorstores import GoodmemVectorStore
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests


class TestGoodmemVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for integration tests."""
        space_id = os.environ.get("GOODMEM_SPACE_ID")
        api_key = os.environ.get("GOODMEM_API_KEY")
        
        if not space_id or not api_key:
            pytest.skip("GOODMEM_SPACE_ID and GOODMEM_API_KEY environment variables required")
        
        store = GoodmemVectorStore(
            space_id=space_id,
            api_key=api_key,
            embedding=self.get_embeddings()
        )
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            pass
 