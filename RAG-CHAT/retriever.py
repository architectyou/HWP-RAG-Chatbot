# retreiver
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from typing import Optional

def retriever_loader(sparse_retreiver : Optional[str], dense_retreiver) -> str : 
    