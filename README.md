# langchain-goodmem

This package contains the LangChain integration with Goodmem

## Installation

```bash
pip install -U langchain-goodmem
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatGoodmem` class exposes chat models from Goodmem.

```python
from langchain_goodmem import ChatGoodmem

llm = ChatGoodmem()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`GoodmemEmbeddings` class exposes embeddings from Goodmem.

```python
from langchain_goodmem import GoodmemEmbeddings

embeddings = GoodmemEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`GoodmemLLM` class exposes LLMs from Goodmem.

```python
from langchain_goodmem import GoodmemLLM

llm = GoodmemLLM()
llm.invoke("The meaning of life is")
```
