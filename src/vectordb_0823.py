from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document


dimension_1024 = "upskyy/e5-large-korean"
dimension_768 = "bespin-global/klue-sroberta-base-continue-learning-by-mnr"

embedding_model =  HuggingFaceEmbeddings(
    model_name = dimension_1024,
    model_kwargs = {'device' : 'cuda'},
    encode_kwargs = {'normalize_embeddings' : True},
)

def vectorize_and_store(splitted_docs, persist_directory="./vector_db"):
    
    embeddings = embedding_model
    # Chroma DB 생성 및 문서 저장
    db = Chroma.from_documents(
        splitted_docs,
        embeddings,
        persist_directory=persist_directory, 
        collection_name = "vector_db"
    )

    db.add_documents(splitted_docs)
    # 변경사항 저장
    db.persist()
    return db
