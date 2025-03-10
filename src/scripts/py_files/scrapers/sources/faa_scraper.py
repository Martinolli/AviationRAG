import re
import os
import requests
from bs4 import BeautifulSoup
import logging
from ..base_scraper import BaseScraper

class FAAScraper(BaseScraper):
    def __init__(self, download_dir):
        super().__init__(download_dir, "FAA")
        self.base_url = "https://www.faa.gov/regulations_policies/advisory_circulars"
    
    def scrape(self):
        """Scrape FAA advisory circulars"""
        downloaded_files = []
        
        try:
            # Get the advisory circulars page
            response = requests.get(self.base_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find links to PDF documents
            pdf_links = soup.select('a[href$=".pdf"]')
            
            for link in pdf_links:
                pdf_url = link['href']
                if not pdf_url.startswith('http'):
                    pdf_url = f"https://www.faa.gov{pdf_url}"
                    
                document_name = os.path.basename(pdf_url)
                document_id = re.search(r'(AC\s\d+-\d+)', link.text)
                
                file_path = self.download_file(pdf_url, document_name)
                if file_path:
                    extra_metadata = {
                        'title': link.text.strip(),
                        'document_id': document_id.group(0) if document_id else ''
                    }
                    
                    downloaded_files.append({
                        'file_path': file_path,
                        'metadata': self.get_document_metadata(
                            document_name, 'pdf', extra_metadata
                        )
                    })
        except Exception as e:
            logging.error(f"Error scraping FAA advisory circulars: {e}")
                        
        return downloaded_files