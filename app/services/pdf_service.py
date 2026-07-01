import fitz  # PyMuPDF
import re

class PDFService:
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def is_noise_page(self, text: str) -> bool:
        """
        Aggressive enterprise heuristic filter to isolate true manual content 
        from metadata, copyright blocks, and navigational indexes.
        """
        lower_text = text.lower()
        
        # 1. Direct Empty/Low-Content Pages
        if len(text.strip()) < 100:
            return True
            
        # 2. Strict Copyright / License Agreement Filter
        # If it has copyright text and legal markers, it's a legal page, not a setup guide.
        copyright_keywords = ["copyright ©", "all rights reserved", "under a license agreement", "u.s. government end users"]
        if any(kw in lower_text for kw in copyright_keywords):
            return True

        # 3. Enhanced Table of Contents & Index Dot Matching
        if len(re.findall(r'\.{4,}', text)) > 2:
            return True
            
        # 4. Explicit TOC headers
        if "contents" in lower_text[:100] and ("get help" in lower_text or "chapter" in lower_text):
            return True

        return False

    def extract_text_by_page(self, file_path: str) -> list:
        doc = fitz.open(file_path)
        pages_data = []
        
        for page_num, page in enumerate(doc):
            raw_text = page.get_text()
            
            if self.is_noise_page(raw_text):
                continue
                
            cleaned = self.clean_text(raw_text)
            pages_data.append({
                "page_number": page_num + 1, 
                "text": cleaned
            })
            
        return pages_data

pdf_service = PDFService()