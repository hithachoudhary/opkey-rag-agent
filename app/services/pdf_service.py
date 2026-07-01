import fitz  # PyMuPDF
import re

class PDFService:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans excessive horizontal spacing but preserves structural vertical 
        newlines so the chunking service can process layout boundaries.
        """
        # Unify multiple horizontal spaces/tabs into a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Unify excessive consecutive newlines down to a single newline
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def is_noise_page(self, text: str) -> bool:
        lower_text = text.lower()
        if len(text.strip()) < 100:
            return True
            
        copyright_keywords = ["copyright ©", "all rights reserved", "under a license agreement", "u.s. government end users"]
        if any(kw in lower_text for kw in copyright_keywords):
            return True

        if len(re.findall(r'\.{4,}', text)) > 2:
            return True
            
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
                
            # Clean text while preserving our crucial '\n' flags
            cleaned = self.clean_text(raw_text)
            pages_data.append({
                "page_number": page_num + 1, 
                "text": cleaned
            })
            
        return pages_data

pdf_service = PDFService()