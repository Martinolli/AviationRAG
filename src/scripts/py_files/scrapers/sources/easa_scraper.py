# easa_scraper.py
import re
import os
import requests
import logging
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from ..base_scraper import BaseScraper

class EASAScraper(BaseScraper):
    def __init__(self, download_dir):
        super().__init__(download_dir, "EASA")
        self.base_url = "https://www.easa.europa.eu/regulations"
    
    def scrape(self):
        """Scrape EASA regulations"""
        downloaded_files = []
        
        try:
            # Get the regulations page
            response = requests.get(self.base_url)
            soup = BeautifulSoup(response.content, 'html.parser')
        
            # Find links to XML regulations
            xml_links = soup.select('a[href$=".xml"]')
        
            for link in xml_links:
                xml_url = link['href']
                if not xml_url.startswith('http'):
                    xml_url = f"https://www.easa.europa.eu{xml_url}"
                
                regulation_name = os.path.basename(xml_url)
                regulation_id = re.search(r'(\d{4}-\d{4})', regulation_name)
            
                file_path = self.download_file(xml_url, regulation_name)
                if file_path:
                    # Extract additional metadata from XML
                    try:
                        tree = ET.parse(file_path)
                        root = tree.getroot()
                        title = root.findtext('.//title', '')
                        pub_date = root.findtext('.//date', '')
                    
                        extra_metadata = {
                            'title': title,
                            'publication_date': pub_date,
                            'regulation_id': regulation_id.group(0) if regulation_id else ''
                        }
                    
                        downloaded_files.append({
                            'file_path': file_path,
                            'metadata': self.get_document_metadata(
                             regulation_name, 'xml', extra_metadata
                            )
                        })
                    except Exception as e:
                        logging.error(f"Error processing {regulation_name}: {e}")
        except Exception as e:
                logging.error(f"Error scraping EASA regulations: {e}")
                    
        return downloaded_files