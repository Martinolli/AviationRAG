#!/usr/bin/env python3
"""
Setup script to create the directory structure and placeholder files for the scraper module.
This script will create all necessary directories and files with skeleton code.
"""
import os
import sys

# Base directory for the scraper module
BASE_DIR = "src/scripts/py_files/scrapers"

## Directory structure to create
DIRECTORY_STRUCTURE = [
    "",  # Base directory
    "sources",
    "processors",
]

# Files to create with their content
FILES = {
    "__init__.py": """# Scraper module package

""",
    
"base_scraper.py": """from abc import ABC, abstractmethod
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
""",
    
    "scraper_manager.py": """import logging
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
""",
    
    "update_corpus.py": """import os
import pickle
import logging
from .scraper_manager import ScraperManager

# Define paths - customize these for your environment
base_dir = r'C:\\Users\\Aspire5 15 i7 4G2050\\ProjectRAG\\AviationRAG'
corpus_path = os.path.join(base_dir, 'data', 'raw', 'aviation_corpus.pkl')
download_dir = os.path.join(base_dir, 'data', 'raw', 'scraped')

def update_corpus_with_scraped_documents(scrapers_to_run=None):
    """Update the aviation_corpus.pkl with scraped documents"""
    # Initialize scraper manager
    manager = ScraperManager(download_dir)
    
    # Run scrapers and get processed documents
    scraped_documents = []
    if scrapers_to_run:
        for scraper_name in scrapers_to_run:
            docs = manager.run_specific_scraper(scraper_name)
            scraped_documents.extend(docs)
    else:
        scraped_documents = manager.run_all_scrapers()
    
    # Format documents to match the existing corpus structure
    formatted_documents = []
    for doc in scraped_documents:
        formatted_doc = {
            'filename': doc['metadata'].get('filename', 'unknown'),
            'text': doc['text'],
            'metadata': doc['metadata'],
            'category': doc['metadata'].get('source', 'unknown')
            # Add other fields as needed to match your existing structure
        }
        formatted_documents.append(formatted_doc)
    
    # Load existing corpus
    existing_documents = []
    if os.path.exists(corpus_path):
        try:
            with open(corpus_path, 'rb') as f:
                existing_documents = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading corpus: {e}")
    
    # Combine and save updated corpus
    updated_corpus = existing_documents + formatted_documents
    
    try:
        with open(corpus_path, 'wb') as f:
            pickle.dump(updated_corpus, f)
        logging.info(f"Updated corpus with {len(formatted_documents)} scraped documents")
        return True
    except Exception as e:
        logging.error(f"Error saving corpus: {e}")
        return False

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Update corpus with all scrapers
    success = update_corpus_with_scraped_documents()
    
    if success:
        print("Successfully updated aviation corpus with scraped documents")
    else:
        print("Failed to update aviation corpus")

if __name__ == '__main__':
    main()
""",
    
    "schedule_scraping.py": """import schedule
import time
import logging
from datetime import datetime
from .update_corpus import update_corpus_with_scraped_documents

def job():
    """Scheduled job to update corpus with scraped documents"""
    logging.info(f"Starting scheduled scraping job at {datetime.now()}")
    success = update_corpus_with_scraped_documents()
    if success:
        logging.info("Successfully updated corpus")
    else:
        logging.error("Failed to update corpus")

def setup_schedule():
    """Setup scheduled jobs"""
    # Run job daily at 2 AM
    schedule.every().day.at("02:00").do(job)
    
    # Alternatively, you can schedule different sources at different frequencies
    # schedule.every().monday.do(lambda: update_corpus_with_scraped_documents(['easa']))
    # schedule.every().wednesday.do(lambda: update_corpus_with_scraped_documents(['faa']))
    
    logging.info("Scheduled scraping jobs set up")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='scraper_scheduler.log'
    )
    setup_schedule()
""",
    
    "sources/__init__.py": """# Scraper sources package
""",
    
    "sources/easa_scraper.py": """import re
import os
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import logging
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
""",
    
    "sources/faa_scraper.py": """import re
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
""",
    
    "sources/skybrary_scraper.py": """import os
import requests
from bs4 import BeautifulSoup
import logging
from ..base_scraper import BaseScraper

class SkyBraryScraper(BaseScraper):
    def __init__(self, download_dir):
        super().__init__(download_dir, "SkyBrary")
        self.base_url = "https://skybrary.aero/categories/aircraft-operations"
    
    def scrape(self):
        """Scrape SkyBrary aircraft operations articles"""
        downloaded_files = []
        
        try:
            # Get the aircraft operations page
            response = requests.get(self.base_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find links to article pages
            article_links = soup.select('a.result-title')
            
            for link in article_links[:10]:  # Limit to 10 articles for testing
                article_url = link['href']
                if not article_url.startswith('http'):
                    article_url = f"https://skybrary.aero{article_url}"
                
                try:
                    # Get the article page
                    article_response = requests.get(article_url)
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                    
                    # Extract article title
                    title = article_soup.select_one('h1').text.strip() if article_soup.select_one('h1') else "Unknown"
                    
                    # Extract article content
                    content_div = article_soup.select_one('div.content')
                    if content_div:
                        # Save as HTML file
                        article_name = os.path.basename(article_url) + ".html"
                        file_path = os.path.join(self.source_dir, article_name)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(str(content_div))
                        
                        extra_metadata = {
                            'title': title,
                            'url': article_url,
                            'category': 'Aircraft Operations'
                        }
                        
                        downloaded_files.append({
                            'file_path': file_path,
                            'metadata': self.get_document_metadata(
                                article_name, 'html', extra_metadata
                            )
                        })
                        logging.info(f"Downloaded SkyBrary article: {title}")
                except Exception as e:
                    logging.error(f"Error processing SkyBrary article {article_url}: {e}")
        
        except Exception as e:
            logging.error(f"Error scraping SkyBrary: {e}")
                    
        return downloaded_files
""",
    
    "sources/ntsb_scraper.py": """import os
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
""",
    
    "processors/__init__.py": """# Document processors package
""",
    
    "processors/xml_processor.py": """import xml.etree.ElementTree as ET
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
""",
    
    "processors/html_processor.py": """from bs4 import BeautifulSoup
import logging

def process_html_document(file_path, metadata):
    """Process HTML document and extract text content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return {
            'text': text,
            'metadata': metadata
        }
    except Exception as e:
        logging.error(f"Error processing HTML document {file_path}: {e}")
        return None
""",
    
    "processors/pdf_processor.py": """import logging
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
""",

def main():
    """Create the directory structure and files for the scraper module."""
    print("Setting up scraper module...")
    
    # Create the base directory
    base_path = os.path.join(os.getcwd(), BASE_DIR)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # Create the directory structure
    for directory in DIRECTORY_STRUCTURE:
        dir_path = os.path.join(base_path, directory)
        if directory and not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Create the files
    for rel_file_path, content in FILES.items():
        file_path = os.path.join(base_path, rel_file_path)
        dir_name = os.path.dirname(file_path)
        
        # Create any needed subdirectories
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # Write the file content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Created {file_path}")
    
    print("Scraper module setup complete!")
    print(f"Module created at: {base_path}")

if __name__ == "__main__":
    main()