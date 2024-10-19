from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

embedding_model = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3",
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': False},
)

def vectorize_and_store(pdf_meta, persist_directory="./chroma_db"):
    # 임베딩 모델 초기화
    embeddings = embedding_model
    
    # 문서 리스트 생성
    documents = [
        Document(
            page_content=page["content"],
            metadata={
                "file_name": page["file_name"],
                "page_num": page["page_num"],
                "total_pages": page["total_pages"]
            }
        ) for page in pdf_meta
    ]

    # Chroma DB 생성 및 문서 저장
    db = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=persist_directory
    )

    # 변경사항 저장
    db.persist()

    return db