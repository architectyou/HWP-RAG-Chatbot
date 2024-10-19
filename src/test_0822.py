import os, shutil
from PyPDF2 import PdfReader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableConfig
import chainlit as cl

# llm 
from llm_loader import LLM
# rag
from pdf_miner import pdf_loader
from vectordb_modified import vectorize_and_store, embedding_model
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

# retreiver
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# validation
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_model = LLM()
llm = llm_model.load_llm()

# temp directory
import tempfile
persist_directory = tempfile.mkdtemp()



@cl.on_chat_start
async def start():
    # session test
    # session_id = cl.user_session.get("id")
    # persist_directory = f""
    cl.user_session.set("vector_db", None)
    cl.user_session.set("chat_history", [])
    await cl.Message(
        content=f"안녕하세요 채팅봇입니다. 궁금하신 부분을 물어보세요!"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    vector_db = cl.user_session.get("vector_db")
    chat_history = cl.user_session.get("chat_history")
    
    if message.elements:
        for element in message.elements:
            if element.mime == "application/pdf":
                file_path = element.path
                msg = cl.Message(content=f"`{element.name}`파일을 읽고 있습니다. 잠시만 기다려주세요...")
                await msg.send()

                # pdf to docs & embedding
                docs_metas = pdf_loader(file_path)
                vector_db = vectorize_and_store(docs_metas)
                
                cl.user_session.set("vector_db", vector_db)

                msg.content = f"Processing `{element.name}` done. You can now ask questions about the PDF! \n\n 파일을 읽고 있습니다. 잠시만 기다려주세요..."
                await msg.update()
    
    # if vector_db is not None:
        # PDF가 업로드된 경우 RAG 체인 사용
                chroma_retriever = vector_db.as_retriever(
                    search_type="mmr",
                    search_kwargs = {"k" : 3, "lambda_mult" : 0.4, "fetch_k" : 20}
                    # search_type="similarity_score_threshold",
                    # search_kwargs = {"score_threshold" : 0.5}
                    )
                bm25_retriever = BM25Retriever.from_documents(docs_metas)
                
                ensemble_retriever = EnsembleRetriever(
                    retrievers = [chroma_retriever, bm25_retriever],
                    weights = [0.7, 0.3]
                )
                
                prompt_template = """
                당신의 역할은 사용자의 질문에 답하는 것입니다.
                제공하는 *컨텍스트* 내용으로 답을 해야 합니다.
                사용자의 *질문*에 대해 답변하세요.
                만약 질문에 대한 답이 문맥에 있는 경우 '모릅니다'를 대답하세요. 문맥에 없는 답변을 지어내지 마세요.

                #지침:
                1. 반드시 한국어로 답변하세요.
                2. 주어진 컨텍스트에 있는 정보만을 사용하여 답변하세요.
                3. 컨텍스트에 관련 정보가 없다면 "주어진 정보로는 이 질문에 답변할 수 없습니다."라고 대답하세요.
                4. 답변은 간결하고 정확하게 유지하세요.
                5. 추측하거나 컨텍스트 외의 정보를 사용하지 마세요.
                6. 질문과 관련 없는 내용으로 답변하지 마세요.
                7. 친절하고 공손하게 답변합니다.
                8. 모르는 내용은 모른다고 답변합니다.

                #질문:
                {question}

                #컨텍스트:
                {context}

                #답변:
                """

                PROMPT = PromptTemplate(
                    template = prompt_template, input_variables = ["context", "question"]
                )
                
                # debugging
                relevant_docs = ensemble_retriever.get_relevant_documents(message.content)
                print(f"got docs : {relevant_docs}")
            
                # docs 내용 debug
                print(f"len docs : {len(relevant_docs)}")
                for i, doc in enumerate(relevant_docs, 1):
                    page = doc.metadata['page']
                    content = doc.page_content
                    print(f"Document {i}:")
                    print(f"Page: {page}")
                    print(f"Content:\n{content}")
                    print("\n" + "-"*50 + "\n")
                
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                context = format_docs(relevant_docs)

                formatted_prompt = PROMPT.format(
                    context = context,
                    question = message.content,
                    chat_history = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in chat_history]),
                )
                
                logger.info(f"User query : {message.content}")
                answer = await cl.make_async(llm.invoke)(formatted_prompt)
                logger.info(f"llm answer : {answer}")
        # pdb.set_trace()
    else:
        # PDF가 업로드되지 않은 경우 일반 대화 응답
        chat_history_text = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in chat_history])
        chat_prompt = PromptTemplate(
            template="""당신은 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 간결하고 명확하게 답변해주세요.
            기존의f 대화 기록을 참고하여 답변하되, 새로운 정보나 관점도 제시할 수 있습니다.

            #대화 기록:
            {chat_history}

            #질문:
            {question}
            #답변:
            """, 
            input_variables = ["chat_history","question"]
        )
        context =""
        formatted_prompt = chat_prompt.format(chat_history=chat_history_text, question=message.content)
        answer = await cl.make_async(llm.invoke)(formatted_prompt)
        # answer = answer.content

    chat_history.append((message.content, answer))
    cl.user_session.set("chat_history", chat_history)
    
    
    ai_message = cl.Message(content=answer.content)
    await ai_message.send()
    
@cl.on_chat_end
async def on_chat_end():
    # ChromaDB의 persist_directory 경로
    persist_directory = "./chroma_db"
    
    # vector_db 객체 가져오기
    vector_db = cl.user_session.get("vector_db")
    
    # persist_directory가 존재하면 삭제
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"ChromaDB at {persist_directory} has been deleted.")
    
    # 세션에서 vector_db 제거
    cl.user_session.set("vector_db", None)