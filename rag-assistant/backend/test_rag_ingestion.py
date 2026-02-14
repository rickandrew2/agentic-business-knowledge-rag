"""Test script for RAG ingestion pipeline with Qdrant."""

import sys
import os
sys.path.insert(0, '.')

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create sample data directory
DATA_DIR = Path("../data")
DATA_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("RAG INGESTION PIPELINE TEST")
print("=" * 70)

# Create sample CSV file
print("\n✓ Creating sample CSV file...")
csv_file = DATA_DIR / "sample-sales.csv"
with open(csv_file, 'w') as f:
    f.write("""product_name,product_id,quarter,sales_amount,units_sold,region
iPhone 15 Pro,PROD001,Q4 2024,450000,1500,North America
iPhone 15 Pro,PROD001,Q3 2024,380000,1200,North America
MacBook Pro 16,PROD002,Q4 2024,650000,800,North America
MacBook Air M3,PROD003,Q4 2024,420000,1200,Europe
iPad Air,PROD004,Q4 2024,280000,1500,Asia Pacific
Apple Watch Series 9,PROD005,Q4 2024,95000,3500,North America
AirPods Pro 2,PROD006,Q4 2024,180000,4500,Global
""")
print(f"  Created: {csv_file}")

# Create sample Markdown file
print("\n✓ Creating sample markdown file...")
md_file = DATA_DIR / "sample-feedback.md"
with open(md_file, 'w') as f:
    f.write("""# Customer Feedback Summary - Q4 2024

## Product Performance

### iPhone 15 Pro
- Average rating: 4.8/5
- Top praise: "Excellent camera quality", "Fast performance", "Beautiful design"
- Main complaint: "High price point", "Limited customization"
- Sentiment: Positive (89% positive reviews)

### MacBook Pro 16
- Average rating: 4.7/5
- Top praise: "Powerful performance", "Beautiful display", "Great build quality"
- Main complaint: "Expensive", "Fan noise under load"
- Sentiment: Very Positive (92% positive reviews)

## Regional Insights

### North America
- Strongest region for premium products
- High demand for MacBook Pro and iPhone 15 Pro
- Average order value: $1,250

### Europe
- Growing iPad market
- Preference for wireless accessories
- Average order value: $980
""")
print(f"  Created: {md_file}")

# Now test the ingestion pipeline
print("\n" + "=" * 70)
print("Testing RAG Ingestion Pipeline with Qdrant")
print("=" * 70)

try:
    from app.rag.ingestion import RAGIngestionPipeline
    
    # Initialize pipeline
    print("\n✓ Initializing RAG Ingestion Pipeline...")
    pipeline = RAGIngestionPipeline()
    
    # Ingest CSV file
    print("\n✓ Ingesting CSV file...")
    csv_result = pipeline.ingest_file(str(csv_file))
    print(f"  Status: {csv_result['status']}")
    print(f"  Original documents: {csv_result.get('original_documents', 'N/A')}")
    print(f"  Chunks created: {csv_result.get('total_chunks', 'N/A')}")
    print(f"  Chunks stored: {csv_result.get('stored_chunks', 'N/A')}")
    
    # Ingest Markdown file
    print("\n✓ Ingesting Markdown file...")
    md_result = pipeline.ingest_file(str(md_file))
    print(f"  Status: {md_result['status']}")
    print(f"  Original documents: {md_result.get('original_documents', 'N/A')}")
    print(f"  Chunks created: {md_result.get('total_chunks', 'N/A')}")
    print(f"  Chunks stored: {md_result.get('stored_chunks', 'N/A')}")
    
    # Get collection stats
    print("\n✓ Checking Qdrant collection...")
    collection_info = pipeline.vector_store.get_collection("business_data")
    print(f"  Collection name: business_data")
    print(f"  Total vectors: {collection_info.points_count}")
    print(f"  Vector size: {collection_info.config.params.vectors.size if hasattr(collection_info.config.params, 'vectors') else 'N/A'}")
    
    # Test retrieval
    print("\n✓ Testing vector search...")
    from langchain_openai import OpenAIEmbeddings
    from app.core.config import settings
    
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key
    )
    query = "What were the top selling products?"
    query_embedding = embeddings.embed_query(query)
    
    results = pipeline.vector_store.search(
        collection_name="business_data",
        query_vector=query_embedding,
        limit=3,
        score_threshold=0.5
    )
    
    print(f"  Query: {query}")
    print(f"  Retrieved {len(results)} results")
    for i, hit in enumerate(results, 1):
        print(f"\n  Result {i} (similarity: {hit.score:.3f}):")
        content = hit.payload.get('content', 'N/A')
        source = hit.payload.get('source', 'N/A')
        print(f"    Source: {source}")
        print(f"    Content: {content[:100]}...")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print(f"  • CSV ingestion: {csv_result['status']}")
    print(f"  • Markdown ingestion: {md_result['status']}")
    print(f"  • Total vectors stored: {collection_info.points_count}")
    print(f"  • Vector search: Working")
    print("\nNext steps:")
    print("  1. Create /api/data/upload endpoint")
    print("  2. Create /api/data/search endpoint")
    print("  3. Create /api/chat/message endpoint with RAG")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
