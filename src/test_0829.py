import os
import pdb
import shutil
from typing import List, Tuple
from langchain.prompts import ChatPromptTemplate, PromptTemplate
import chainlit as cl
from llm_loader import LLM
from pdf_miner_0828 import pdf_loader
from vectordb_0823 import vectorize_and_store, embedding_model
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_teddynote.retrievers import KiwiBM25Retriever
import logging
import tempfile
import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_model = LLM()
llm = llm_model.load_llm()

persist_directory = tempfile.mkdtemp()

@cl.on_chat_start
async def start():

    cl.user_session.set("vector_db", None)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("ensemble_retriever", None)
    
    base_dir = "./server/"
    temp_dir = os.path.join(base_dir,"temp_" + cl.user_session.get("id"))
    os.makedirs(temp_dir, exist_ok=True)
    cl.user_session.set("temp_dir", temp_dir)
    
    await cl.Message(content="안녕하세요 채팅봇입니다. 궁금하신 부분을 물어보세요!").send()

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        formatted_doc = {
            "page_content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown source"),
            "page": doc.metadata.get("page", "Unknown page")
        }
        formatted_docs.append(formatted_doc)
    return formatted_docs

async def process_pdf(file_path: str):
    msg = cl.Message(content=f"`{os.path.basename(file_path)}`파일을 읽고 있습니다. 잠시만 기다려주세요...")
    await msg.send()

    # retrieval
    docs_metas = pdf_loader(file_path)
    temp_dir = cl.user_session.get("temp_dir")
    vector_db = vectorize_and_store(docs_metas, persist_directory=temp_dir)
    
    chroma_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3, 
            "lambda_mult": 0.4, 
            "fetch_k": 20}
    )
    
    bm25_retriever = KiwiBM25Retriever.from_documents(docs_metas)
    bm25_retriever.k = 4
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.4, 0.6]
    )
    
    # print(f"debugging vector_store : {vector_db}")
    cl.user_session.set("vector_db", vector_db)
    cl.user_session.set("ensemble_retriever", ensemble_retriever)
    # elements = [
    #   cl.Pdf(name="pdf1", display="inline", path=file_path)
    # ]
    msg.content = f"Processing `{os.path.basename(file_path)}` done. You can now ask questions about the PDF!"
    # msg.elements = elements
    await msg.update()

async def get_response(query: str, context: str, chat_history: List[Tuple[str, str]]) -> str:
    prompt_template = """
당신은 사용자 질문에 논리적으로 답변하는 AI 어시스턴트입니다. 다음 지침을 따라 답변을 작성하세요:

1. 한국어로 답변하세요.
2. 제공된 컨텍스트와 대화 기록만 사용하세요.
3. 정보가 부족하면 "주어진 정보로는 답변할 수 없습니다"라고 하세요.
4. 답변 과정:
   a) [질문 분석]: 질문의 핵심을 파악합니다.
   b) [정보 검색]: 관련 정보를 찾습니다.
   c) [추론]: 논리적 추론을 합니다.
   d) [결론]: 최종 답변을 제시합니다.
5. 추측이 필요하면 "이는 추측입니다"라고 표시하세요.
6. 친절하고 공손하게 답변하세요.
7. 모르는 내용은 솔직히 모른다고 하세요.
8. 참고한 컨텍스트의 'page' 정보를 답변에 포함하세요.

질문: {question}

컨텍스트: {context}

대화 기록: {chat_history}

위 정보를 바탕으로 답변을 작성하세요.
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    formatted_prompt = PROMPT.format(
        context=context,
        question=query,
        chat_history="\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in chat_history]),
    )
    
    # pdb.set_trace()
    
    logger.info(f"User query : {query}")
    logger.info(f"context : {context}")
    logger.info(f"chat_history : {chat_history}")
    logger.info(f"chat_tern : {len(chat_history)}")
    answer = await cl.make_async(llm.invoke)(formatted_prompt)
    logger.info(f"llm answer : {answer.content}")
    
    return answer

@cl.on_message
async def on_message(message: cl.Message):
    
    vector_db = cl.user_session.get("vector_db")
    ensemble_retriever = cl.user_session.get("ensemble_retriever")
    
    if message.elements:
        for element in message.elements:
            if element.mime == "application/pdf":
                try : 
                    await process_pdf(element.path)
                except PSEOF :
                    await cl.Message(content = "PDF 파일을 처리하는 중 오류가 발생했습니다. 파일이 손상되었거나 불완전할 수 있습니다.").send()

                vector_db = cl.user_session.get("vector_db")
                ensemble_retriever = cl.user_session.get("ensemble_retriever")
                
    if ensemble_retriever is not None:
        relevant_docs = ensemble_retriever.get_relevant_documents(message.content)
        context = format_docs(relevant_docs)
        
    elif vector_db is not None:
        relevant_docs = vector_db.similarity_search(message.content, k=3)
        context = format_docs(relevant_docs)
        
    else :
        context = ""

    
    chat_history = cl.user_session.get("chat_history")
    
    # chat_history limit
    if len(chat_history) > 5 : 
        chat_history = chat_history[-5:]

    answer = await get_response(message.content, context, chat_history)
    logger.info(f"answer : {answer}")
    chat_history.append((f"Human : {message.content}, Assistant :{answer.content}"))
    cl.user_session.set("chat_history", chat_history)
    
    print(cl.user_session.get("chat_history"))
    print(cl.user_session.get("vector_db"))
    print(cl.user_session.get("ensemble_retriever"))
    
    ai_message = cl.Message(content=answer.content)
    await ai_message.send()

@cl.on_chat_end
async def on_chat_end():
    
    temp_dir = cl.user_session.get("temp_dir")
    
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Temporary directory {temp_dir} has been deleted.")
    # if os.path.exists(persist_directory):
    #     shutil.rmtree(persist_directory)
    #     print(f"ChromaDB at {persist_directory} has been deleted.")
    cl.user_session.set("vector_db", None)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("ensemble_retriever", None)
    cl.user_session.set("temp_dir", None)