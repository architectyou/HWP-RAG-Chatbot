# RAG Chatbot

## 사용방법
- 레포지토리 복사
```bash
git clone "현재 repository URL"
```

- 가상환경 셋팅 및 라이브러리 설치
: 가상환경이 설치되어 있다고 가정
```bash
pip install -r requirements.txt
```

- 로컬 환경에서 실행
```bash
chainlit run main.py
```

<br/>

## 파일 구성
- main.py -> python main 로직을 실행시키는 함수. 위 명령어 통해 실행
- vectordb.py -> 파일 embedding 로직 (chroma db 이용한 vectordb 생성)
- pdf_preprocesser.py -> pdf 전처리 로직 (다양한 파일 유형 업로드를 시행한다면 이같은 전처리 로직이 파일별로 필요)

<br/>

## 수정 사항
- 0822
    - 현재 llama3.1:70b 로 로컬 로드되어 있음 -> 노트북 사양으로 불가능
    - 10B 이하 모델로 변경해야 함 -> main.py 에서 line20 model 이름 수정
    - /src 이하 폴더에서 retriever, chunking 관련 조합 시도 중

- 0823
    - 파일 유무에 따른 답변 로직 다시 짜야함
    - retriever이용해 docs는 잘 가져오는 것 확인
    - prompt + 로직 수정

- 0826(수정사항)
    - prompt (CoT) 기법 적용. llm 답변시 user query에 대해 분석 / 정보 검색 / 추론 / 결론 을 내는 과정으로 답변하도록 수정하였음.
    - 파일 유무에 따른 답변 로직 과정 수정 (prompt 한번에 묶어 처리하였음)
    - context length가 모델에 따라 한정적이기 때문에 local (노트북) 환경에서 테스트 할 시 속도 저하, 답변 성능 저하, 멀티턴 불가능 현상 발생 할 수 있음.
    - test_0823.py 파일로 실행해야 함.
    - 벡터 데이터베이스 갱신 방법도 변경하였음.
    - 현재 H100 서버에서 제공하는 llama3.1 70B 모델로는 멀티턴이 가능
    - llama3.1 모델의 고질적인 문제로 한글이 아닌 다른 언어를 무작위로 뱉어주기도 함.. (한글 특화모델이 아니기 때문)

- 0827 
    - DockerFile 생성 및 도커 이미지 테스트 (RAG-CHAT 폴더 하위)