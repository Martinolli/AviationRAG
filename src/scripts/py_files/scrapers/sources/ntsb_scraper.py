import os
import requests
import logging
from bs4 import BeautifulSoup
from ..base_scraper import BaseScraper

class NTSBScraper(BaseScraper):
    def __init__(self, download_dir):
        super().__init__(download_dir, "NTSB")
        self.base_url = "https://www.ntsb.gov/investigations/AccidentReports/Pages/aviation.aspx"
    
    def scrape(self):
        """Scrape NTSB aviation accident reports"""
        downloaded_files = []
        
        try:
            # Get the aviation accident reports page
            response = requests.get(self.base_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find links to PDF reports
            pdf_links = soup.select('a[href$=".pdf"]')
            
            for link in pdf_links[:5]:  # Limit to 5 reports for testing
                pdf_url = link['href']
                if not pdf_url.startswith('http'):
                    pdf_url = f"https://www.ntsb.gov{pdf_url}"
                
                document_name = os.path.basename(pdf_url)
                
                file_path = self.download_file(pdf_url, document_name)
                if file_path:
                    # Extract title from link text or nearby elements
                    title = link.text.strip()
                    if not title:
                        title = "NTSB Aviation Accident Report"
                    
                    extra_metadata = {
                        'title': title,
                        'url': pdf_url
                    }
                    
                    downloaded_files.append({
                        'file_path': file_path,
                        'metadata': self.get_document_metadata(
                            document_name, 'pdf', extra_metadata
                        )
                    })
        except Exception as e:
            logging.error(f"Error scraping NTSB accident reports: {e}")
                    
        return downloaded_files