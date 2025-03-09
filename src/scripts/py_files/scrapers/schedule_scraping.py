# schedule_scraping.py
import schedule
import time
import logging
from datetime import datetime
from update_corpus import update_corpus_with_scraped_documents

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