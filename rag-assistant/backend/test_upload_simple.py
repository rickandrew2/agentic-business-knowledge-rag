"""Simple test script to upload a file to RAG system."""

import requests
from pathlib import Path

# File to upload
file_path = Path("../data/sample-sales.csv")

# Upload endpoint
url = "http://localhost:8000/api/data/upload"

print(f"Uploading {file_path.name}...")

try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.name, f, 'text/csv')}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Upload successful!")
        print(f"File ID: {result['file_id']}")
        print(f"Chunks created: {result['chunks_created']}")
        print(f"Message: {result['message']}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Exception occurred: {e}")

# Test status endpoint
print("\n" + "="*50)
print("Checking data status...")

try:
    response = requests.get("http://localhost:8000/api/data/status")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        status = response.json()
        print(f"Total files: {status['total_files']}")
        print(f"Total chunks: {status['total_chunks']}")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Exception: {e}")
