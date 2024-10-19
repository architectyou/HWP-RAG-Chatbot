from libhwp import HWPReader

file_path = "/Users/parksunyoung/Desktop/git/Projects/rag-chatbot/2017110542_기전1.hwp"
hwp = HWPReader(file_path)

# 모든 문단 출력 (표, 캡션 포함)
for paragraph in hwp.find_all('paragraph'):
    print(paragraph)

# 테이블 내용 출력
for table in hwp.find_all('table'):
    for cell in table.cells:
        for paragraph in cell.paragraphs:
            print(paragraph)

# 문서에 사용된 파일 저장
for file in hwp.bin_data:
    with open(file.name, 'wb') as f:
        f.write(file.data)