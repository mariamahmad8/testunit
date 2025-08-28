from importlib import metadata

from langchain_goodmem.chat_models import ChatGoodmem
from langchain_goodmem.document_loaders import GoodmemLoader
from langchain_goodmem.embeddings import GoodmemEmbeddings
from langchain_goodmem.retrievers import GoodmemRetriever
from langchain_goodmem.toolkits import GoodmemToolkit
from langchain_goodmem.tools import GoodmemTool
from langchain_goodmem.vectorstores import GoodmemVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatGoodmem",
    "GoodmemVectorStore",
    "GoodmemEmbeddings",
    "GoodmemLoader",
    "GoodmemRetriever",
    "GoodmemToolkit",
    "GoodmemTool",
    "__version__",
]
