from pdf_processing_module import EnhancedPDFProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        processor = EnhancedPDFProcessor()
        logger.info("Starting to process PDFs and build vector store...")
        processor.process_pdfs()
        logger.info("Vector store created successfully!")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise