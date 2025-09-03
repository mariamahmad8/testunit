import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from langchain_goodmem.vectorstores import GoodmemVectorStore

document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
]


def test_add_documents(): 
    with patch('langchain_goodmem.vectorstores.MemoriesApi') as mock_memories_api, \
           patch('langchain_goodmem.vectorstores.SpacesApi'), \
           patch('langchain_goodmem.vectorstores.ApiClient'), \
           patch('langchain_goodmem.vectorstores.Configuration'):
        
        vector_store = GoodmemVectorStore(
            space_id="test-space",
            api_key="test-key"
        )
        
        mock_memories_instance = mock_memories_api.return_value
        mock_batch_response = Mock()
        mock_batch_response.results = [
            Mock(memory_id="1"),
            Mock(memory_id="2"), 
            Mock(memory_id="3")
        ]
        mock_memories_instance.batch_create_memory.return_value = mock_batch_response
        
        result = vector_store.add_documents(documents=documents, ids=["1", "2", "3"])
        assert result == ["1", "2", "3"]

def test_delete(): 
    with patch('langchain_goodmem.vectorstores.MemoriesApi') as mock_memories_api, \
         patch('langchain_goodmem.vectorstores.SpacesApi'), \
         patch('langchain_goodmem.vectorstores.ApiClient'), \
         patch('langchain_goodmem.vectorstores.Configuration'):
        
        vector_store = GoodmemVectorStore(
            space_id="test-space",
            api_key="test-key"
        )
        

        mock_memories_instance = mock_memories_api.return_value
        result = vector_store.delete(ids=["1", "2", "3"])
        
        assert result is None

        assert mock_memories_instance.delete_memory.call_count == 3
        mock_memories_instance.delete_memory.assert_any_call("1")
        mock_memories_instance.delete_memory.assert_any_call("2")
        mock_memories_instance.delete_memory.assert_any_call("3")


def test_add_texts():
    """Test adding texts to the vector store.""" 
    with patch('langchain_goodmem.vectorstores.MemoriesApi') as mock_memories_api, \
         patch('langchain_goodmem.vectorstores.SpacesApi'), \
         patch('langchain_goodmem.vectorstores.ApiClient'), \
         patch('langchain_goodmem.vectorstores.Configuration'):
        
        vector_store = GoodmemVectorStore(
            space_id="test-space",
            api_key="test-key"
        )
        
        mock_memories_instance = mock_memories_api.return_value
        mock_memory_response = Mock()
        mock_memory_response.memory_id = "memory-123"
        mock_memories_instance.create_memory.return_value = mock_memory_response

        result = vector_store.add_texts(texts=["Hello world"])

        assert result == ["memory-123"]
        assert mock_memories_instance.create_memory.call_count == 1


def test_similarity_search():
    """Test similarity search returns Document objects."""
    with patch('langchain_goodmem.vectorstores.MemoriesApi') as mock_memories_api, \
         patch('langchain_goodmem.vectorstores.SpacesApi'), \
         patch('langchain_goodmem.vectorstores.ApiClient'), \
         patch('langchain_goodmem.vectorstores.Configuration'):
        
        vector_store = GoodmemVectorStore(
            space_id="test-space", 
            api_key="test-key"
        )
        
        mock_memories_instance = mock_memories_api.return_value
        mock_event = Mock()
        mock_event.retrieved_item = Mock()
        mock_event.retrieved_item.memory = Mock()
        mock_event.retrieved_item.memory.original_content = b"Test content"
        mock_event.retrieved_item.memory.metadata = {"source": "test"}
        mock_event.retrieved_item.chunk = None
        
        mock_memories_instance.retrieve_memory.return_value = [mock_event]
    
        result = vector_store.similarity_search("test query", k=1)
    
        assert len(result) == 1
        assert result[0].page_content == "Test content"
        assert result[0].metadata == {"source": "test"}


def test_get_by_ids():
    with patch('langchain_goodmem.vectorstores.MemoriesApi') as mock_memories_api, \
         patch('langchain_goodmem.vectorstores.SpacesApi'), \
         patch('langchain_goodmem.vectorstores.ApiClient'), \
         patch('langchain_goodmem.vectorstores.Configuration'):
        
        vector_store = GoodmemVectorStore(
            space_id="test-space",
            api_key="test-key"
        )
        

        mock_memories_instance = mock_memories_api.return_value
        mock_memory = Mock()
        mock_memory.original_content = b"Retrieved content"
        mock_memory.metadata = {"id": "memory-1"}
        mock_memories_instance.get_memory.return_value = mock_memory

        result = vector_store.get_by_ids(["memory-1"])
        assert len(result) == 1
        assert result[0].page_content == "Retrieved content"
        assert result[0].metadata == {"id": "memory-1"}
    
        mock_memories_instance.get_memory.assert_called_with("memory-1")
