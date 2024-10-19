from dotenv import load_dotenv
from langchain_upstage import UpstageDocumentParseLoader

file_path = "/Users/parksunyoung/Desktop/git/Projects/rag-chatbot/2017110542_기전1.hwp"

loader = UpstageDocumentParseLoader(
    file_path,
    output_type="text",
    split = "page",
    use_ocr = True,
    exclude = ["header", "footer"],
    
)

docs = loader.load()

for doc in docs[:3]:
    print(doc)