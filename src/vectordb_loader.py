from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.http import models
import torch

#qdrant client
qdrant_client = QdrantClient(url = "http://localhost:6333")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dimension_1024 = "upskyy/e5-large-korean"
dimension_768 = "bespin-global/klue-sroberta-base-continue-learning-by-mnr"

embedding_model =  HuggingFaceEmbeddings(
    model_name = dimension_1024,
    model_kwargs = {'device' : device},
    encode_kwargs = {'normalize_embeddings' : True},
)

model = SentenceTransformer(dimension_768)

#chromadb
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

#qdrant
def create_collection(collection_name) :
    collections = qdrant_client.get_collections()
    if any(collection.name == collection_name for collection in collections.collections) :
        return
    else : 
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size = 768, distance=Distance.COSINE)
        )
        
def qdrant_vectorize_and_store(collection_name, documents):
    collections = qdrant_client.get_collections()
    if any(collection.name == collection_name for collection in collections.collections) :

        points = []
        for i, doc in enumerate(documents):
            vector = model.encode(doc.page_content)
            meta = doc.metadata
            point = models.PointStruct(
                id = i,
                vector = vector.tolist(),
                payload={
                    "article_number" : meta['article_number'],
                    "article_name" : meta['article_name'],
                    "article_year" : meta["article_year"],
                    "content" : doc.page_content
                }
            )
            points.append(point)
            
        qdrant_client.upsert(
            collection_name=collection_name,
            points = points
        )
        return f"finished {len(points)} documents into collection '{collection_name}'"
    
    else : 
        print(f"Error : The {collection_name} is not Exists.")
        
        
        
if __name__ == "__main__" : 
    # data upsert
    # docs = [Document(metadata={'article_number': '제34조', 'article_name': '개인정보보호법시행령', 'article_year': '개정 날짜'}, page_content='제34조의2(개인정보 보호 인증의 기준ㆍ방법ㆍ절차 등) ① 보호위원회는 제30조제1항 각 호의 사항을 고려하여 개인 정보 보호의 관리적ㆍ기술적ㆍ물리적 보호대책의 수립 등을 포함한 법 제32조의2제1항에 따른 인증의 기준을 정하 여 고시한다. <개정 2017. 7. 26., 2020. 8. 4., 2023. 9. 12.> ② 법 제32조의2제1항 따라 개인정보 보호의 인증을 받으려는 자(이하 이 조 및 제34조의3에서 “신청인”이라 한다 )는 다음 각 호의 사항이 포함된 개인정보 보호 인증신청서(전자문서로 된 신청서를 포함한다)를 제34조의6에 따른 개인정보 보호 인증 전문기관(이하 “인증기관”이라 한다)에 제출하여야 한다.<개정 2024. 3. 12.> 1. 인증 대상 개인정보 처리시스템의 목록 2. 개인정보 관리체계를 수립ㆍ운영하는 방법과 절차 3. 개인정보 관리체계 및 보호대책 구현과 관련되는 문서 목록 ③ 인증기관은 제2항에 따른 인증신청서를 받은 경우에는 신청인과 인증의 범위 및 일정 등에 관하여 협의하여야 한다. ④ 법 제32조의2제1항에 따른 개인정보 보호 인증심사는 제34조의8에 따른 개인정보 보호 인증심사원이 서면심사 또는 현장심사의 방법으로 실시한다. ⑤ 인증기관은 제4항에 따른 인증심사의 결과를 심의하기 위하여 정보보호에 관한 학식과 경험이 풍부한 사람을 위 원으로 하는 인증위원회를 설치ㆍ운영하여야 한다. ⑥ 제1항부터 제5항까지에서 규정한 사항 외에 인증신청, 인증심사, 인증위원회의 설치ㆍ운영 및 인증서의 발급 등 개인정보 보호 인증에 필요한 세부사항은 보호위원회가 정하여 고시한다.<개정 2017. 7. 26., 2020. 8. 4.> [본조신설 2016. 7. 22.]'), Document(metadata={'article_number': '제34조', 'article_name': '개인정보보호법시행령', 'article_year': '개정 날짜'}, page_content='제34조의3(개인정보 보호 인증의 수수료) ① 신청인은 인증기관에 개인정보 보호 인증 심사에 소요되는 수수료를 납부 하여야 한다. ② 보호위원회는 개인정보 보호 인증 심사에 투입되는 인증 심사원의 수 및 인증심사에 필요한 일수 등을 고려하여 제1항에 따른 수수료 산정을 위한 구체적인 기준을 정하여 고시한다.<개정 2017. 7. 26., 2020. 8. 4.> [본조신설 2016. 7. 22.]'), Document(metadata={'article_number': '제34조', 'article_name': '개인정보보호법시행령', 'article_year': '개정 날짜'}, page_content='제34조의4(인증취소) ① 인증기관은 법 제32조의2제3항에 따라 개인정보 보호 인증을 취소하려는 경우에는 제34조의 2제5항에 따른 인증위원회의 심의ㆍ의결을 거쳐야 한다. ② 보호위원회 또는 인증기관은 법 제32조의2제3항에 따라 인증을 취소한 경우에는 그 사실을 당사자에게 통보하 고, 관보 또는 인증기관의 홈페이지에 공고하거나 게시해야 한다.<개정 2017. 7. 26., 2020. 8. 4.> [본조신설 2016. 7. 22.]'), Document(metadata={'article_number': '제34조', 'article_name': '개인정보보호법시행령', 'article_year': '개정 날짜'}, page_content='제34조의5(인증의 사후관리) ① 법 제32조의2제4항에 따른 사후관리 심사는 서면심사 또는 현장심사의 방법으로 실시 한다. ② 인증기관은 제1항에 따른 사후관리를 실시한 결과 법 제32조의2제3항 각 호의 사유를 발견한 경우에는 제34조 의2제5항에 따른 인증위원회의 심의를 거쳐 그 결과를 보호위원회에 제출해야 한다.<개정 2017. 7. 26., 2020. 8. 4.> [본조신설 2016. 7. 22.]'), Document(metadata={'article_number': '제34조', 'article_name': '개인정보보호법시행령', 'article_year': '개정 날짜'}, page_content='제34조의6(개인정보 보호 인증 전문기관) ① 법 제32조의2제5항에서 “대통령령으로 정하는 전문기관”이란 다음 각 호 의 기관을 말한다. <개정 2016. 9. 29., 2017. 7. 26., 2020. 8. 4.> 1. 한국인터넷진흥원 2. 다음 각 목의 요건을 모두 충족하는 법인, 단체 또는 기관 중에서 보호위원회가 지정ㆍ고시하는 법인, 단체 또는 기관 가. 제34조의8에 따른 개인정보 보호 인증심사원 5명 이상을 보유할 것 법제처 26 국가법령정보센터 Page 27 개인정보 보호법 시행령 나. 보호위원회가 실시하는 업무수행 요건ㆍ능력 심사에서 적합하다고 인정받을 것 ② 제1항제2호에 해당하는 법인, 단체 또는 기관의 지정과 그 지정의 취소에 필요한 세부기준 등은 보호위원회가 정하여 고시한다.<개정 2017. 7. 26., 2020. 8. 4.> [본조신설 2016. 7. 22.]'),
    # Document(metadata={'article_number': '제4조', 'article_name': '개인정보보호법시행령', 'article_year': '개정 날짜'}, page_content='제4조의2(영리업무의 금지) 법 제7조제1항에 따른 개인정보 보호위원회(이하 “보호위원회”라 한다)의 위원은 법 제7조 의6제1항에 따라 영리를 목적으로 다음 각 호의 어느 하나에 해당하는 업무에 종사해서는 안 된다. 1. 법 제7조의9제1항에 따라 보호위원회가 심의ㆍ의결하는 사항과 관련된 업무 2. 법 제40조제1항에 따른 개인정보 분쟁조정위원회(이하 “분쟁조정위원회”라 한다)가 조정하는 사항과 관련된 업 무 [본조신설 2020. 8. 4.]'), Document(metadata={'article_number': '제5조', 'article_name': '개인정보보호법시행령', 'article_year': '개정 날짜'}, page_content='제5조(전문위원회) ① 보호위원회는 법 제7조의9제1항에 따른 심의ㆍ의결 사항에 대하여 사전에 전문적으로 검토하기 위하여 보호위원회에 다음 각 호의 분야별 전문위원회(이하 “전문위원회”라 한다)를 둔다. <개정 2020. 8. 4., 2023. 9. 12.> 1. 개인정보의 국외 이전 분야 2. 그 밖에 보호위원회가 필요하다고 인정하는 분야 ② 제1항에 따라 전문위원회를 두는 경우 각 전문위원회는 위원장 1명을 포함한 20명 이내의 위원으로 성별을 고려 하여 구성하되, 전문위원회 위원은 다음 각 호의 사람 중에서 보호위원회 위원장이 임명하거나 위촉하고, 전문위원 회 위원장은 보호위원회 위원장이 전문위원회 위원 중에서 지명한다.<개정 2016. 7. 22., 2020. 8. 4., 2023. 9. 12.> 1. 보호위원회 위원 2. 개인정보 보호 관련 업무를 담당하는 중앙행정기관의 관계 공무원 3. 개인정보 보호에 관한 전문지식과 경험이 풍부한 사람 4. 개인정보 보호와 관련된 단체 또는 사업자단체에 속하거나 그 단체의 추천을 받은 사람 ③ 제1항 및 제2항에서 규정한 사항 외에 전문위원회의 구성 및 운영 등에 필요한 사항은 보호위원회의 의결을 거 쳐 보호위원회 위원장이 정한다.<신설 2023. 9. 12.>')]
    # create_collection("test")
    # qdrant_vectorize_and_store("test", docs)
    
    #query search
    hits = qdrant_client.query_points(
        collection_name="test",
        query=model.encode("인증기관에서 수수료 산정은 어떻게 해?").tolist(),
        limit=3,
    ).points

    for hit in hits:
        print(hit.payload, "score:", hit.score)