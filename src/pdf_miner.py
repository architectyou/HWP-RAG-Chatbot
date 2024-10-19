from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

import pdb

from dotenv import load_dotenv
load_dotenv()

def show_metadata(docs):
    if docs:
        print("[metadata]")
        print(list(docs[0].metadata.keys()))
        print("\n[examples]")
        max_key_length = max(len(k) for k in docs[0].metadata.keys())
        for k, v in docs[0].metadata.items():
            print(f"{k:<{max_key_length}} : {v}")


def pdf_loader(file_path) :
    # file_path = "개인정보보호법.pdf"
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # print(docs[0].page_content[:300])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
    splits = text_splitter.split_documents(docs)
    pdb.set_trace()
    
    print(show_metadata(docs))
    # pdb.set_trace()

    return splits

def pdf_semantic_chunker(file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    
    documents = []
    for doc in docs : 
        chunks = text_splitter.split_text(doc.page_content)
        documents.append(chunks)
        
    pdb.set_trace()
    
    print(len(chunks))
    return chunks

if __name__ == "__main__" :
    file_path = "./../test.pdf"
    splits = pdf_loader(file_path)
    print(len(splits))