from datetime import datetime
import pdb
from bs4 import BeautifulSoup
import re
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.schema import Document
import numpy as np

# 조항 추출 함수
def clean_text(text):
    cleaned_large_blank = re.sub(r'\s+', ' ', text)
    cleaned = re.sub(r'\\n|\n|', '', cleaned_large_blank)
    return cleaned.strip()


def extract_articles(soup, article_name, article_year):
    
    # soup = BeautifulSoup(contents, 'html.parser')
    articles = []
    current_article = ""
    current_meta = {}
    
    for div in soup.find_all('div'):
        text = clean_text(div.get_text())
        if re.match(r'제\d+조', text):
            if current_article:
                articles.append(Document(page_content=current_article, metadata=current_meta))
            current_article = text
            current_meta = {"article_number": re.search(r'(제\d+조)', text).group(1),
                            "article_name" : article_name,
                            "article_year" : article_year}
        elif current_article:
            current_article += " " + text
        
        # 페이지 번호 추출
        if 'name' in div.attrs and div['name'].startswith('Page'):
            current_meta['page'] = div['name']
    
    if current_article:
        
        articles.append(Document(page_content=current_article, metadata=current_meta))

    return articles

if __name__ == "__main__":
    html_file_path = "/home/ubuntu/rag_chatbot/law1.html"
    with open(html_file_path, 'r', encoding='utf-8') as file : 
        soup = BeautifulSoup(file, 'html.parser')
    
    date = "2024-01-04"
    extract_articles(soup, "표준 개인정보 보호지침", date)