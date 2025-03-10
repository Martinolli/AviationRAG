import logging
import PyPDF2
import io

def process_pdf_document(file_path, metadata):
    """Process PDF document and extract text content"""
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            # Extract text content
            text_content = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content.append(page.extract_text())
            
            # Join all pages
            full_text = '\n\n'.join(text_content)
            
            return {
                'text': full_text,
                'metadata': metadata
            }
    except Exception as e:
        logging.error(f"Error processing PDF document {file_path}: {e}")
        return None