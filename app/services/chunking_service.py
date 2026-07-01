import re

class ChunkingService:
    def clean_header_footer_noise(self, text: str) -> str:
        """Strips repetitive headers and structural running page labels."""
        noise_header = r"Oracle Fusion Cloud Financials\s+Getting Started with Your Financials Implementation"
        text = re.sub(noise_header, "", text, flags=re.IGNORECASE)
        text = re.sub(r"Chapter\s+\d+\s+\d+", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _reconstruct_semantic_sentences(self, raw_text: str) -> list:
        """
        Groups fragments into complete semantic units, keeping lists and
        headings intact while fixing mid-sentence line breaks.
        """
        cleaned_text = self.clean_header_footer_noise(raw_text)
        cleaned_text = re.sub(r'\s+\d+\.$', '', cleaned_text)
        lines = cleaned_text.split("\n")
        
        semantic_units = []
        active_sentence = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If it's a structural list item or header, isolate it
            if line.startswith(('•', '-', '*')) or line.endswith(':'):
                if active_sentence:
                    semantic_units.append(active_sentence.strip())
                    active_sentence = ""
                semantic_units.append(line)
                continue
                
            if active_sentence:
                active_sentence += " " + line
            else:
                active_sentence = line
                
            if line.endswith(('.', '!', '?')):
                semantic_units.append(active_sentence.strip())
                active_sentence = ""
                
        if active_sentence:
            semantic_units.append(active_sentence.strip())
            
        return semantic_units

    def split_text_into_chunks(self, raw_text: str, filename: str, page_number: int, chunk_size: int = 750) -> list:
        """
        Slices text into chunks using structural sentence boundaries
        for clean text overlap.
        """
        semantic_units = self._reconstruct_semantic_sentences(raw_text)
        
        chunks = []
        current_buffer = []
        chunk_idx = 0
        
        for unit in semantic_units:
            # Check the length if we append this unit
            prospective_text = " ".join(current_buffer + [unit])
            
            # If the block exceeds our ceiling constraint, seal it!
            if len(prospective_text) > chunk_size and current_buffer:
                chunk_text = " ".join(current_buffer)
                chunks.append(self._create_chunk_object(chunk_text, filename, page_number, chunk_idx))
                chunk_idx += 1
                
                # Carry over last 2 complete structural entries for semantic context overlap
                current_buffer = current_buffer[-2:] if len(current_buffer) >= 2 else current_buffer
            
            current_buffer.append(unit)
            
        # Flush the final block for the active page
        if current_buffer:
            chunk_text = " ".join(current_buffer)
            chunks.append(self._create_chunk_object(chunk_text, filename, page_number, chunk_idx))
            
        return chunks

    def _create_chunk_object(self, text: str, filename: str, page_number: int, index: int) -> dict:
        char_count = len(text)
        return {
            "text": text,
            "metadata": {
                "document": "Oracle Financials Cloud Guide",
                "source": filename,
                "page_number": page_number,
                "chunk_index": index,
                "chunk_id": f"oracle_p{str(page_number).zfill(3)}_c{str(index).zfill(3)}",
                "character_metrics": char_count,
                "estimated_tokens": int(char_count / 4)
            }
        }

chunking_service = ChunkingService()