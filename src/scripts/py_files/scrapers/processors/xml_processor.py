# xml_processor.py
import xml.etree.ElementTree as ET
import logging

def process_xml_document(file_path, metadata):
    """Process XML document and extract text content"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract text content
        text_sections = []
        
        # Adjust these selectors based on actual XML structure
        for section in root.findall('.//section') + root.findall('.//content'):
            section_text = ' '.join(section.itertext()).strip()
            if section_text:
                text_sections.append(section_text)
        
        # Join all text sections
        full_text = '\n\n'.join(text_sections)
        
        return {
            'text': full_text,
            'metadata': metadata
        }
    except Exception as e:
        logging.error(f"Error processing XML document {file_path}: {e}")
        return None