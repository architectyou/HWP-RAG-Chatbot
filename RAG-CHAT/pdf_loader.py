from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

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
    
    print(show_metadata(docs))
    # pdb.set_trace()

    return splits

if __name__ == "__main__" :
    file_path = "./../개인정보보호법.pdf"
    splits = pdf_loader(file_path)
    print(len(splits))