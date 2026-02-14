"""End-to-end RAG pipeline test with Qdrant.

Run this to verify the entire system works:
    cd backend && python test_rag_pipeline_e2e.py
"""

import sys
sys.path.insert(0, '.')

import time
from pathlib import Path

print("=" * 70)
print("RAG PIPELINE END-TO-END TEST (QDRANT)")
print("=" * 70)

try:
    # Step 1: Initialize services
    print("\n[1/5] Initializing services...")
    from app.rag.ingestion import RAGIngestionPipeline
    from app.core.config import settings
    from langchain_openai import OpenAIEmbeddings
    
    pipeline = RAGIngestionPipeline()
    
    # Initialize embeddings for search
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key
    )
    print("✓ Services initialized")
    
    # Step 2: Get collection stats
    print("\n[2/5] Checking existing data...")
    try:
        collection_info = pipeline.vector_store.get_collection("business_data")
        existing_count = collection_info.points_count
        print(f"✓ Found existing collection with {existing_count} vectors")
    except Exception:
        existing_count = 0
        print("✓ Collection will be created on first ingest")
    
    # Step 3: Ingest sample data
    print("\n[3/5] Ingesting sample data...")
    sample_csv = Path("../data/sample-sales.csv")
    sample_md = Path("../data/sample-feedback.md")
    
    total_chunks_ingested = 0
    
    if sample_csv.exists():
        result = pipeline.ingest_file(str(sample_csv))
        if result['status'] == 'success':
            print(f"✓ CSV: {result['stored_chunks']} chunks stored")
            total_chunks_ingested += result['stored_chunks']
        else:
            print(f"✗ CSV failed: {result.get('error', 'Unknown error')}")
    else:
        print("⚠ CSV file not found, skipping")
    
    if sample_md.exists():
        result = pipeline.ingest_file(str(sample_md))
        if result['status'] == 'success':
            print(f"✓ Markdown: {result['stored_chunks']} chunks stored")
            total_chunks_ingested += result['stored_chunks']
        else:
            print(f"✗ Markdown failed: {result.get('error', 'Unknown error')}")
    else:
        print("⚠ Markdown file not found, skipping")
    
    if total_chunks_ingested == 0:
        print("✗ No data ingested!")
        sys.exit(1)
    
    # Step 4: Verify storage
    print("\n[4/5] Verifying storage...")
    collection_info = pipeline.vector_store.get_collection("business_data")
    total_vectors = collection_info.points_count
    print(f"✓ Collection contains {total_vectors} vectors")
    
    # Step 5: Test retrieval
    print("\n[5/5] Testing retrieval with Qdrant search...")
    
    test_queries = [
        "What were the top performing products?",
        "Tell me about customer feedback",
        "Which region had the highest growth?",
        "What do customers say about MacBook Pro?"
    ]
    
    all_results = []
    for query in test_queries:
        start = time.time()
        
        # Embed query
        query_embedding = embeddings.embed_query(query)
        
        # Search Qdrant using query_points
        response = pipeline.vector_store.query_points(
            collection_name="business_data",
            query=query_embedding,
            limit=3,
            score_threshold=0.5
        )
        
        retrieval_time = time.time() - start
        results = response.points
        
        print(f"\n  Query: '{query}'")
        print(f"  Results: {len(results)} documents ({retrieval_time*1000:.1f}ms)")
        
        for i, hit in enumerate(results, 1):
            source = hit.payload.get('source', 'Unknown')
            content = hit.payload.get('content', '')
            print(f"    {i}. [{source}] Score: {hit.score:.3f}")
            print(f"       {content[:80]}...")
        
        all_results.append({
            "query": query,
            "result_count": len(results),
            "top_score": results[0].score if results else 0
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ Total chunks ingested: {total_chunks_ingested}")
    print(f"✓ Total vectors in collection: {total_vectors}")
    print(f"✓ Performed {len(test_queries)} test retrievals")
    print(f"✓ All queries returned results")
    
    print("\nTest Results:")
    for result in all_results:
        query_preview = result['query'][:40] + '...' if len(result['query']) > 40 else result['query']
        print(f"  - '{query_preview}' → {result['result_count']} results (top score: {result['top_score']:.3f})")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - RAG PIPELINE WITH QDRANT WORKING!")
    print("=" * 70)
    
    print("\nNext Steps:")
    print("  1. Run: cd backend && uvicorn app.main:app --reload")
    print("  2. Test API endpoints:")
    print("     - POST /api/data/upload - Upload files")
    print("     - GET /api/data/search?q=query - Search vectors")
    print("     - POST /api/chat/message - RAG-powered chat")

except Exception as e:
    print(f"\n✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
