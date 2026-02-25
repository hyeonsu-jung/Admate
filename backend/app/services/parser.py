import fitz  # PyMuPDF
import pandas as pd
from typing import List, Dict, Any
import io
import logging

logger = logging.getLogger(__name__)

class DocumentParser:
    """PDF 및 Excel 문서를 파싱하여 텍스트 및 메타데이터를 추출하는 클래스"""

    @staticmethod
    def parse_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """PDF 파일을 페이지별로 파싱하여 텍스트 추출"""
        chunks = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            logger.info(f"Parsing PDF: {filename} ({len(doc)} pages)")
            
            for page_num, page in enumerate(doc):
                # 1. 텍스트 추출
                text = page.get_text()
                
                # 2. 이미지 추출
                images = []
                image_list = page.get_images(full=True)
                skipped_small = 0
                for img_index, img in enumerate(image_list):
                    # 스마트 필터링: 너무 작은 이미지(아이콘, 불필요한 장식) 건너뛰기
                    # img[2]는 너비, img[3]은 높이 (PyMuPDF 이미지 정보 구조)
                    width = img[2]
                    height = img[3]
                    
                    # 100x100 미만의 작은 이미지는 분석 가치가 낮으므로 제외 (속도 개선 핵심)
                    if width < 100 or height < 100:
                        skipped_small += 1
                        continue
                        
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append(image_bytes)
                
                if text.strip() or images:
                    logger.info(
                        f"Page {page_num+1}: images_found={len(image_list)}, images_used={len(images)}, skipped_small={skipped_small}"
                    )
                    chunks.append({
                        "content": text,
                        "images": images,  # 추출된 이미지 바이트 리스트
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1,
                            "type": "pdf"
                        }
                    })
            doc.close()
        except Exception as e:
            logger.error(f"PDF Parsing error: {str(e)}")
            
        return chunks

    @staticmethod
    def parse_excel(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Excel 파일을 시트별로 파싱하여 텍스트(CSV/Text 형태) 추출"""
        chunks = []
        try:
            excel_file = io.BytesIO(file_bytes)
            # 모든 시트 읽기
            all_sheets = pd.read_excel(excel_file, sheet_name=None)
            logger.info(f"Parsing Excel: {filename} ({len(all_sheets)} sheets)")
            
            for sheet_name, df in all_sheets.items():
                # 데이터 정제: 병합된 셀로 인한 NaN 값을 앞의 값으로 채움 (Forward Fill)
                # 병합된 셀은 주로 첫 셀에만 값이 있고 나머지는 NaN임
                df = df.ffill(axis=0) 
                
                # 빈 행 및 빈 열 제거
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                # 데이터프레임을 문자열로 변환
                text_content = df.to_string(index=False)
                if text_content.strip():
                    chunks.append({
                        "content": f"Sheet: {sheet_name}\n\n{text_content}",
                        "metadata": {
                            "source": filename,
                            "sheet": sheet_name,
                            "type": "excel"
                        }
                    })
        except Exception as e:
            logger.error(f"Excel Parsing error: {str(e)}")
            
        return chunks

    @staticmethod
    def parse_txt(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """일반 텍스트(.txt) 파일 파싱"""
        chunks = []
        try:
            # BOM(Byte Order Mark) 대응을 위해 utf-8-sig 선호
            text = file_bytes.decode('utf-8-sig', errors='replace')
            logger.info(f"Parsing TXT: {filename}")
            
            if text.strip():
                chunks.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "type": "txt"
                    }
                })
        except Exception as e:
            logger.error(f"TXT Parsing error: {str(e)}")
            
        return chunks

    def parse_file(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """파일 확장자에 따른 파싱 분기"""
        f_lower = filename.lower()
        if f_lower.endswith('.pdf'):
            return self.parse_pdf(file_bytes, filename)
        elif f_lower.endswith(('.xlsx', '.xls')):
            return self.parse_excel(file_bytes, filename)
        elif f_lower.endswith('.csv'):
            # CSV 지원 추가
            df = pd.read_csv(io.BytesIO(file_bytes))
            return [{"content": df.to_string(index=False), "metadata": {"source": filename, "type": "csv"}}]
        elif f_lower.endswith('.txt'):
            # TXT 지원 추가
            return self.parse_txt(file_bytes, filename)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {filename}")
