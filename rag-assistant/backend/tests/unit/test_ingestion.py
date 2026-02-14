"""Unit tests for ingestion module."""

import pytest
import tempfile
from pathlib import Path
from app.rag.ingestion import (
    Document,
    CSVLoader,
    MarkdownLoader,
    Chunker,
    DataIngestionPipeline
)


class TestDocument:
    """Test Document class."""
    
    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(text="Hello world", metadata={"source": "test.txt"})
        assert doc.text == "Hello world"
        assert doc.metadata["source"] == "test.txt"


class TestCSVLoader:
    """Test CSV loading."""
    
    def test_load_csv(self):
        """Test loading a CSV file."""
        csv_content = "name,value\nProduct A,100\nProduct B,200"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            docs = CSVLoader.load(temp_path)
            assert len(docs) == 2
            assert "Product A" in docs[0].text
            assert docs[0].metadata["source_type"] == "csv"
        finally:
            Path(temp_path).unlink()


class TestMarkdownLoader:
    """Test Markdown loading."""
    
    def test_load_markdown(self):
        """Test loading markdown file."""
        md_content = """# Title
        
Some content here.

## Section 2

More content."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(md_content)
            temp_path = f.name
        
        try:
            docs = MarkdownLoader.load(temp_path)
            assert len(docs) > 0
            assert any("Title" in doc.text for doc in docs)
            assert docs[0].metadata["source_type"] == "markdown"
        finally:
            Path(temp_path).unlink()


class TestChunker:
    """Test chunking functionality."""
    
    def test_chunk_by_tokens(self):
        """Test chunking text by tokens."""
        chunker = Chunker(chunk_size=100, overlap=20)
        
        long_text = " ".join(["word"] * 500)  # ~500 words
        chunks = chunker.chunk_by_tokens(long_text)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)
    
    def test_chunk_documents(self):
        """Test chunking documents."""
        chunker = Chunker(chunk_size=100, overlap=20)
        
        docs = [
            Document(text="First document " * 50, metadata={"source": "doc1.txt"}),
            Document(text="Second document " * 50, metadata={"source": "doc2.txt"})
        ]
        
        chunks = chunker.chunk_documents(docs)
        
        assert len(chunks) > len(docs)
        assert all(isinstance(chunk, tuple) for chunk in chunks)
        assert all(len(chunk) == 2 for chunk in chunks)  # (text, metadata)


class TestDataIngestionPipeline:
    """Test the full ingestion pipeline."""
    
    def test_ingest_csv(self):
        """Test ingesting a CSV file."""
        csv_content = "product,sales\niPhone,1000\nMac,500"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            pipeline = DataIngestionPipeline()
            chunks = pipeline.ingest_file(temp_path)
            
            assert len(chunks) > 0
            assert all(len(chunk) == 2 for chunk in chunks)
            assert any("iPhone" in chunk[0] for chunk in chunks)
        finally:
            Path(temp_path).unlink()
    
    def test_ingest_markdown(self):
        """Test ingesting a markdown file."""
        md_content = "# Sales Report\n\nQ4 was great! Sales up 50%."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(md_content)
            temp_path = f.name
        
        try:
            pipeline = DataIngestionPipeline()
            chunks = pipeline.ingest_file(temp_path)
            
            assert len(chunks) > 0
            assert any("Sales" in chunk[0] for chunk in chunks)
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
