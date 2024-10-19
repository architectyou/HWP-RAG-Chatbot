from PyPDF2 import PdfReader
import os
import pdb

def content_extractor(file_path):
    pdf_meta = []

    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        file_name = os.path.basename(file_path)
        total_pages = len(pdf_reader.pages)

        for page_num, page in enumerate(pdf_reader.pages, start=1):
            content = page.extract_text()
            
            page_info = {
                "file_name": file_name,
                "page_num": page_num,
                "content": content,
                "total_pages": total_pages
            }
            
            pdf_meta.append(page_info)
    # pdb.set_trace()
    return pdf_meta


# metadatase = content_extractor("ZeRO.pdf")