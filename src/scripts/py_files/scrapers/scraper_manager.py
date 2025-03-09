# scraper_manager.py
import logging
import os
from .sources.easa_scraper import EASAScraper
from .sources.faa_scraper import FAAScraper
from .sources.skybrary_scraper import SkyBraryScraper
from .sources.ntsb_scraper import NTSBScraper
from .processors.xml_processor import process_xml_document
from .processors.html_processor import process_html_document
from .processors.pdf_processor import process_pdf_document

class ScraperManager:
    def __init__(self, download_dir):
        self.download_dir = download_dir
        self.scrapers = {
            'easa': EASAScraper(download_dir),
            'faa': FAAScraper(download_dir),
            'skybrary': SkyBraryScraper(download_dir),
            'ntsb': NTSBScraper(download_dir)
        }
        
    def run_all_scrapers(self):
        """Run all scrapers and return processed documents"""
        all_documents = []
        
        for name, scraper in self.scrapers.items():
            logging.info(f"Starting scraper for {name}")
            downloaded_files = scraper.scrape()
            
            for file_info in downloaded_files:
                file_path = file_info['file_path']
                metadata = file_info['metadata']
                
                # Select appropriate processor based on file type
                if file_path.endswith('.xml'):
                    processed_doc = process_xml_document(file_path, metadata)
                elif file_path.endswith('.html') or file_path.endswith('.htm'):
                    processed_doc = process_html_document(file_path, metadata)
                elif file_path.endswith('.pdf'):
                    processed_doc = process_pdf_document(file_path, metadata)
                else:
                    logging.warning(f"Unsupported file type: {file_path}")
                    continue
                    
                if processed_doc:
                    all_documents.append(processed_doc)
                    
        return all_documents
        
    def run_specific_scraper(self, scraper_name):
        """Run a specific scraper by name"""
        if scraper_name not in self.scrapers:
            logging.error(f"Unknown scraper: {scraper_name}")
            return []
            
        scraper = self.scrapers[scraper_name]
        downloaded_files = scraper.scrape()
        processed_docs = []
        
        # Process files based on type
        for file_info in downloaded_files:
            file_path = file_info['file_path']
            metadata = file_info['metadata']
            
            # Select appropriate processor based on file type
            if file_path.endswith('.xml'):
                processed_doc = process_xml_document(file_path, metadata)
            elif file_path.endswith('.html') or file_path.endswith('.htm'):
                processed_doc = process_html_document(file_path, metadata)
            elif file_path.endswith('.pdf'):
                processed_doc = process_pdf_document(file_path, metadata)
            else:
                logging.warning(f"Unsupported file type: {file_path}")
                continue
                
            if processed_doc:
                processed_docs.append(processed_doc)
                
        return processed_docs