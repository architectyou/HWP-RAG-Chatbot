import os, shutil
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableConfig
import chainlit as cl

# rag
from pdf_preprocesser import content_extractor
from vectordb import vectorize_and_store, embedding_model
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

# debug
from langchain.globals import set_verbose
set_verbose(True)
# Ollama 모델 초기화
llm = ChatOllama(model="VirnectX-Llama3-Korean-8B-V2-Q4_K_M:latest")

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
                docs_metas = content_extractor(file_path)
                vector_db = vectorize_and_store(docs_metas)
                
                cl.user_session.set("vector_db", vector_db)

                msg.content = f"Processing `{element.name}` done. You can now ask questions about the PDF! \n\n 파일을 읽고 있습니다. 잠시만 기다려주세요..."
                await msg.update()
    
    if vector_db is not None:
        # PDF가 업로드된 경우 RAG 체인 사용
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs = {"k" : 3}
            )
        
        prompt_template = """
        당신은 질문-답변 작업을 위한 도우미입니다. 주어진 컨텍스트를 바탕으로 질문에 답변해야 합니다.

        컨텍스트:
        {context}

        질문: {question}

        # 지침
        1. 주어진 컨텍스트에 있는 정보만을 사용하여 답변하세요.
        2. 컨텍스트에 관련 정보가 없다면 "주어진 정보로는 이 질문에 답변할 수 없습니다."라고 대답하세요.
        3. 답변은 간결하고 정확하게 유지하세요.
        4. 추측하거나 컨텍스트 외의 정보를 사용하지 마세요.
        5. 답변 시 참고한 출처를 반드시 표기하세요. 출처는 아래 형식을 사용하세요.

        답변:

        출처:
        {sources}
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question", "sources"]
        )
        
        relevant_docs = retriever.get_relevant_documents(message.content)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def format_sources(docs):
            sources = []
            for doc in docs:
                source = f"- {doc.metadata['file_name']}, 페이지 {doc.metadata['page_num']}/{doc.metadata['total_pages']}"
                sources.append(source)
            return "\n".join(sources)

        context = format_docs(relevant_docs)
        sources = format_sources(relevant_docs)

        formatted_prompt = PROMPT.format(
            context=context,
            question=message.content,
            sources=sources,
            chat_history = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in chat_history]),
        )
        
        answer = await cl.make_async(llm.invoke)(formatted_prompt)
        # pdb.set_trace()
    else:
        # PDF가 업로드되지 않은 경우 일반 대화 응답
        chat_history_text = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in chat_history])
        chat_prompt = PromptTemplate(
            template="""당신은 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 간결하고 명확하게 답변해주세요.
            이전 대화 기록을 참고하여 답변하되, 새로운 정보나 관점도 제시할 수 있습니다.

            대화 기록:
            {chat_history}

            Human: {question}
            AI:"""
        )
        formatted_prompt = chat_prompt.format(chat_history=chat_history_text, question=message.content)
        answer = await cl.make_async(llm.invoke)(formatted_prompt)

    chat_history.append((message.content, answer.content))
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