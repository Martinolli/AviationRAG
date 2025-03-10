import os
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