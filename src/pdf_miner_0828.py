import pdb
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# html parser
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from html_split import extract_articles

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from bs4 import BeautifulSoup
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 150)
    
    splits = []
    for doc in docs : 
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks : 
            splits.append(chunk)
            
    print(show_metadata(docs))
    # pdb.set_trace()
    return splits

def pdf_html_loader(file_path) :
    loader = PDFMinerPDFasHTMLLoader(file_path)
    doc = loader.load()
    pdb.set_trace()
    
    soup = BeautifulSoup(doc[0].page_content)
    splits = extract_articles(soup, "개인정보보호법시행령", "개정 날짜")
    
    pdb.set_trace()

    return splits

def pdf_semantic_chunker(file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    
    documents = []
    for doc in docs : 
        chunks = text_splitter.split_text(doc.page_content)
        documents.append(chunks)
    
    print(len(chunks))
    return chunks

if __name__ == "__main__" :
    # file_path = "./../test.pdf"
    law_1 = "/home/ubuntu/rag_chatbot/개인정보 보호법 시행령(대통령령)(제34309호)(20240315).pdf"
    law_2 = "/home/ubuntu/rag_chatbot/개인정보 보호법(법률)(제19234호)(20240315).pdf"
    law_3 = "/home/ubuntu/rag_chatbot/표준 개인정보 보호지침(개인정보보호위원회고시)(제2024-1호)(20240104).pdf"
    splits = pdf_html_loader(law_1)
    print(len(splits))