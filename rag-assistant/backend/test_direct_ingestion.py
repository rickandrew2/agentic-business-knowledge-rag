"""Direct test of RAG ingestion pipeline - no API server."""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("=== Testing RAG Ingestion Pipeline ===")
    
    # Import after path setup
    from app.rag.ingestion import RAGIngestionPipeline
    from app.core.config import settings
    
    logger.info(f"OpenAI API key configured: {settings.openai_api_key[:10]}...")
    logger.info(f"Chroma path: {settings.chroma_path}")
    
    # Initialize pipeline
    logger.info("\n1. Initializing RAG pipeline...")
    pipeline = RAGIngestionPipeline()
    logger.info("✅ Pipeline initialized successfully")
    
    # Test with sample data
    sample_file = Path("../data/sample-sales.csv")
    if sample_file.exists():
        logger.info(f"\n2. Testing ingestion with {sample_file.name}...")
        result = pipeline.ingest_file(str(sample_file))
        logger.info(f"✅ Ingestion result: {result}")
    else:
        logger.warning(f"Sample file not found: {sample_file}")
    
    logger.info("\n=== Test completed successfully! ===")
    
except Exception as e:
    logger.error(f"❌ Error during test: {e}", exc_info=True)
    sys.exit(1)
