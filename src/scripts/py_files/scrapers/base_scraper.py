# base_scraper.py
from abc import ABC, abstractmethod
import os
import logging
import requests
from datetime import datetime

class BaseScraper(ABC):
    def __init__(self, download_dir, source_name):
        self.download_dir = download_dir
        self.source_name = source_name
        self.source_dir = os.path.join(download_dir, source_name)
        
        # Create source-specific directory
        if not os.path.exists(self.source_dir):
            os.makedirs(self.source_dir)
    
    @abstractmethod
    def scrape(self):
        """Implement the scraping logic for specific source"""
        pass
    
    def download_file(self, url, filename):
        """Common method to download files"""
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                file_path = os.path.join(self.source_dir, filename)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logging.info(f"Downloaded {filename} from {self.source_name}")
                return file_path
            else:
                logging.error(f"Failed to download {url}: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}")
            return None
    
    def get_document_metadata(self, filename, doc_type, extra_metadata=None):
        """Generate standard metadata for documents"""
        metadata = {
            'source': self.source_name,
            'scrape_date': datetime.now().isoformat(),
            'document_type': doc_type,
            'filename': filename
        }
        
        if extra_metadata:
            metadata.update(extra_metadata)
            
        return metadata