"""Goodmem vector stores."""

from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from goodmem_client.api import SpacesApi, MemoriesApi
from goodmem_client.configuration import Configuration
from goodmem_client.api_client import ApiClient
from goodmem_client.models import (
    MemoryCreationRequest,
    BatchMemoryCreationRequest,
    RetrieveMemoryRequest,
)
from goodmem_client.rest import ApiException

logger = logging.getLogger(__name__)

VST = TypeVar("VST", bound=VectorStore)


class GoodmemVectorStore(VectorStore):
    # TODO: Replace all TODOs in docstring.
    """Goodmem vector store integration.

    # TODO: Replace with relevant packages, env vars.
    Setup:
        Install ``langchain-goodmem`` and set environment variable ``GOODMEM_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-goodmem
            export GOODMEM_API_KEY="your-api-key"

    # TODO: Populate with relevant params.
    Key init args — indexing params:
        collection_name: str
            Name of the collection.
        embedding_function: Embeddings
            Embedding function to use.

    # TODO: Populate with relevant params.
    Key init args — client params:
        client: Optional[Client]
            Client to use.
        connection_args: Optional[dict]
            Connection arguments.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain_goodmem.vectorstores import GoodmemVectorStore
            from langchain_openai import OpenAIEmbeddings

            vector_store = GoodmemVectorStore(
                collection_name="foo",
                embedding_function=OpenAIEmbeddings(),
                connection_args={"uri": "./foo.db"},
                # other params...
            )

    # TODO: Populate with relevant variables.
    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    # TODO: Populate with relevant variables.
    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    # TODO: Fill out with relevant variables and example output.
    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter={"bar": "baz"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Async:
        .. code-block:: python

            # add documents
            # await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            # await vector_store.adelete(ids=["3"])

            # search
            # results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: python

            # TODO: Example output

    """  # noqa: E501

    def __init__(
        self, 
        space_id: str,
        api_key: str,
        embedding: Optional[Embeddings] = None,
        host: str = "https://api.goodmem.com"
    ) -> None:
        """Initialize with Goodmem configuration.

        Args:
            space_id: The Goodmem space ID to store memories in.
            api_key: Your Goodmem API key.
            embedding: Optional embedding function for LangChain compatibility. 
                      Goodmem handles embeddings at the space level, so this is typically not needed.
            host: The Goodmem API host URL.
        """
        self.embedding = embedding
        self._space_id = space_id
        
        # Configure Goodmem client
        self._configuration = Configuration(
            host=host,
            api_key={"ApiKeyAuth": api_key}
        )
        self._api_client = ApiClient(self._configuration)
        self._memories_api = MemoriesApi(self._api_client)
        self._spaces_api = SpacesApi(self._api_client)



    

    @classmethod
    def from_texts(
        cls: Type[GoodmemVectorStore],
        texts: List[str],
        space_id: str, 
        api_key: str,
        embedding: Optional[Embeddings] = None, 
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> GoodmemVectorStore:
        store = cls(
            space_id=space_id, 
            api_key=api_key, 
            embedding=embedding,
        )
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    # optional: add custom async implementations
    # @classmethod
    # async def afrom_texts(
    #     cls: Type[VST],
    #     texts: List[str],
    #     embedding: Embeddings,
    #     metadatas: Optional[List[dict]] = None,
    #     **kwargs: Any,
    # ) -> VST:
    #     return await asyncio.get_running_loop().run_in_executor(
    #         None, partial(cls.from_texts, **kwargs), texts, embedding, metadatas
    #     )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding


    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if not texts:
             return []
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length")
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids length must match texts length")
        metas   = metadatas or [{} for _ in texts]
        ext_ids = ids or [None] * len(texts)
        requests = []
        for t, m, ext in zip(texts, metas, ext_ids):
            req = MemoryCreationRequest(
                space_id=self._space_id,             
                original_content=t,
                content_type="text/plain",
                metadata=m,
                external_id=ext,                
            )
            requests.append(req)

        if len(requests) == 1: #if they are only adding a single memory
            created = self._memories_api.create_memory(requests[0])
            return [str(created.memory_id)]
        else:
            batch_request = BatchMemoryCreationRequest(requests=requests)
            batch_response = self._memories_api.batch_create_memory(batch_request)
            return [str(result.memory_id) for result in batch_response.results]

    

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the store."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

    # optional: add custom async implementations
    # async def aadd_documents(
    #     self,
    #     documents: List[Document],
    #     ids: Optional[List[str]] = None,
    #     **kwargs: Any,
    # ) -> List[str]:
    #     raise NotImplementedError

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete memories by their IDs."""
        if not ids:
            return
            
        for memory_id in ids:
            try:
                self._memories_api.delete_memory(memory_id)
            except ApiException as e:
                logger.warning(f"Failed to delete memory {memory_id}: {e.reason}")
            except Exception as e:
                logger.error(f"Unexpected error deleting memory {memory_id}: {e}")

    # optional: add custom async implementations
    # async def adelete(
    #     self, ids: Optional[List[str]] = None, **kwargs: Any
    # ) -> None:
    #     raise NotImplementedError

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        documents = []
        for memory_id in ids: 
            try: 
                memory = self._memories_api.get_memory(memory_id)
                if memory:
                  content = memory.original_content.decode('utf-8') if memory.original_content else ""
                  documents.append(Document(
                      page_content=content,
                      metadata=memory.metadata or {}
                ))
            except Exception: 
                pass
        return documents

    # optional: add custom async implementations
    # async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
    #     raise NotImplementedError


    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        retrieve_request = RetrieveMemoryRequest(
            message=query,
            space_keys=[{"space_id": self._space_id}], #shoudl be passed in through constructor/init
            requested_size=k,
            fetch_memory=True,
            fetch_memory_content=True
        )
        
        documents = []
        for event in self._memories_api.retrieve_memory(retrieve_request):
            if hasattr(event, 'retrieved_item') and event.retrieved_item:
                item = event.retrieved_item
                if item.memory:
                    content = item.memory.original_content.decode('utf-8') if item.memory.original_content else ""
                    documents.append(Document(
                        page_content=content,
                        metadata=item.memory.metadata or {}
                    ))
                elif item.chunk:
                    documents.append(Document(
                        page_content=item.chunk.chunk_text,
                        metadata=item.chunk.metadata or {}
                    ))
        
        return documents[:k]

    # optional: add custom async implementations
    # async def asimilarity_search(
    #     self, query: str, k: int = 4, **kwargs: Any
    # ) -> List[Document]:
    #     # This is a temporary workaround to make the similarity search
    #     # asynchronous. The proper solution is to make the similarity search
    #     # asynchronous in the vector store implementations.
    #     func = partial(self.similarity_search, query, k=k, **kwargs)
    #     return await asyncio.get_event_loop().run_in_executor(None, func)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        retrieve_request = RetrieveMemoryRequest(
            message=query,
            space_keys=[{"space_id": self._space_id}],
            requested_size=k,
            fetch_memory=True,
            fetch_memory_content=True
        )
        
        documents_with_scores = []
        for event in self._memories_api.retrieve_memory(retrieve_request):
            if hasattr(event, 'retrieved_item') and event.retrieved_item:
                item = event.retrieved_item
                if item.memory:
                    content = item.memory.original_content.decode('utf-8') if item.memory.original_content else ""
                    score = item.memory.relevance_score if hasattr(item.memory, 'relevance_score') else 0.0
                    documents_with_scores.append((
                        Document(page_content=content, metadata=item.memory.metadata or {}),
                        score
                    ))
                elif item.chunk:
                    score = item.chunk.relevance_score  # This has the similarity score from Goodmem
                    documents_with_scores.append((
                        Document(page_content=item.chunk.chunk_text, metadata=item.chunk.metadata or {}),
                        score
                    ))
        
        return documents_with_scores[:k]

    # optional: add custom async implementations
    # async def asimilarity_search_with_score(
    #     self, *args: Any, **kwargs: Any
    # ) -> List[Tuple[Document, float]]:
    #     # This is a temporary workaround to make the similarity search
    #     # asynchronous. The proper solution is to make the similarity search
    #     # asynchronous in the vector store implementations.
    #     func = partial(self.similarity_search_with_score, *args, **kwargs)
    #     return await asyncio.get_event_loop().run_in_executor(None, func)

    ### ADDITIONAL OPTIONAL SEARCH METHODS BELOW ###

    # def similarity_Be(
    #     self, embedding: List[float], k: int = 4, **kwargs: Any
    # ) -> List[Document]:
    #     raise NotImplementedError

    # optional: add custom async implementations
    # async def asimilarity_search_by_vector(
    #     self, embedding: List[float], k: int = 4, **kwargs: Any
    # ) -> List[Document]:
    #     # This is a temporary workaround to make the similarity search
    #     # asynchronous. The proper solution is to make the similarity search
    #     # asynchronous in the vector store implementations.
    #     func = partial(self.similarity_search_by_vector, embedding, k=k, **kwargs)
    #     return await asyncio.get_event_loop().run_in_executor(None, func)

    # def max_marginal_relevance_search(
    #     self,
    #     query: str,
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     raise NotImplementedError

    # optional: add custom async implementations
    # async def amax_marginal_relevance_search(
    #     self,
    #     query: str,
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     # This is a temporary workaround to make the similarity search
    #     # asynchronous. The proper solution is to make the similarity search
    #     # asynchronous in the vector store implementations.
    #     func = partial(
    #         self.max_marginal_relevance_search,
    #         query,
    #         k=k,
    #         fetch_k=fetch_k,
    #         lambda_mult=lambda_mult,
    #         **kwargs,
    #     )
    #     return await asyncio.get_event_loop().run_in_executor(None, func)

    # def max_marginal_relevance_search_by_vector(
    #     self,
    #     embedding: List[float],
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     raise NotImplementedError

    # optional: add custom async implementations
    # async def amax_marginal_relevance_search_by_vector(
    #     self,
    #     embedding: List[float],
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     raise NotImplementedError
