import os
import pdb
import shutil
from typing import List, Tuple
from PyPDF2 import PdfReader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableConfig
import chainlit as cl
from llm_loader import LLM
from pdf_loader import pdf_loader
from vectordb_loader import vectorize_and_store, embedding_model
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_model = LLM()
llm = llm_model.load_llm()

persist_directory = tempfile.mkdtemp()

# def get_port_number():
    # return os.environ.get("SERVER_PORT", '9000')

@cl.on_chat_start
async def start():
    cl.user_session.set("vector_db", None)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("ensemble_retriever", None)
    
    # 임시 디렉토리
    # port = get_port_number()
    # base_dir = os.path.join(os.getcwd(), port)
    base_dir = "./server/"
    temp_dir = os.path.join(base_dir,"temp_" + cl.user_session.get("id"))
    os.makedirs(temp_dir, exist_ok=True)
    cl.user_session.set("temp_dir", temp_dir)
    
    await cl.Message(content="안녕하세요 채팅봇입니다. 궁금하신 부분을 물어보세요!").send()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

async def process_pdf(file_path: str):
    msg = cl.Message(content=f"`{os.path.basename(file_path)}`파일을 읽고 있습니다. 잠시만 기다려주세요...")
    await msg.send()

    docs_metas = pdf_loader(file_path)
    temp_dir = cl.user_session.get("temp_dir")
    vector_db = vectorize_and_store(docs_metas, persist_directory=temp_dir)
    
    chroma_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.4, "fetch_k": 20}
    )
    bm25_retriever = BM25Retriever.from_documents(docs_metas)
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    
    # print(f"debugging vector_store : {vector_db}")
    cl.user_session.set("vector_db", vector_db)
    cl.user_session.set("ensemble_retriever", ensemble_retriever)
    # print(f"session db :", cl.user_session.get("vectordb"))

    msg.content = f"Processing `{os.path.basename(file_path)}` done. You can now ask questions about the PDF!"
    await msg.update()

async def get_response(query: str, context: str, chat_history: List[Tuple[str, str]]) -> str:
    prompt_template = """
        당신의 역할은 사용자의 질문에 논리적이고 단계적으로 답하는 것입니다.
        제공된 *컨텍스트*와 *대화 기록*을 바탕으로 사용자의 *질문*에 답변하세요.

        #지침:
        1. 반드시 한국어로 답변하세요.
        2. 주어진 컨텍스트와 대화 기록에 있는 정보만을 사용하여 답변하세요.
        3. 답변 과정을 다음 단계로 나누어 진행하세요:
        a) 질문 분석: 사용자의 질문을 명확히 이해하고 핵심을 파악합니다.
        b) 정보 검색: 컨텍스트와 대화 기록에서 관련 정보를 찾습니다.
        c) 추론: 찾은 정보를 바탕으로 논리적 추론을 진행합니다.
        d) 결론 도출: 추론을 바탕으로 최종 답변을 형성합니다.
        4. 각 단계를 명확히 구분하여 표시하세요. (예: [질문 분석], [정보 검색] 등)
        5. 컨텍스트에 관련 정보가 없다면 "주어진 정보로는 이 질문에 답변할 수 없습니다."라고 대답하세요.
        6. 답변은 논리적이고 단계적으로 구성하되, 최종적으로는 간결하고 정확해야 합니다.
        7. 추측이 필요한 경우, 그 이유를 명확히 설명하고 "이는 추측입니다"라고 표시하세요.
        8. 질문과 관련 없는 내용으로 답변하지 마세요.
        9. 친절하고 공손하게 답변하세요.
        10. 모르는 내용은 솔직히 모른다고 답변하세요.

        #질문:
        {question}

        #컨텍스트:
        {context}

        #대화 기록:
        {chat_history}

        #답변:
        [질문 분석]
        (질문의 핵심을 파악하고 어떤 정보가 필요한지 분석합니다.)

        [정보 검색]
        (컨텍스트와 대화 기록에서 관련 정보를 찾아 정리합니다.)

        [추론]
        (찾은 정보를 바탕으로 논리적 추론을 진행합니다.)

        [결론]
        (추론을 바탕으로 최종 답변을 제시합니다.)
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
    
    logger.info(f"User query : {query}")
    logger.info(f"context : {context}")
    logger.info(f"chat_history : {chat_history}")
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
                await process_pdf(element.path)
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