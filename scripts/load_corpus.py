import asyncio
import json
import os
import sys
import httpx

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CORPUS_JSON_PATH = os.path.join(DATA_DIR, "corpus.json")
API_URL = "http://localhost:8000/tenants/acme/ingest"


async def load_and_ingest():
    if not os.path.exists(CORPUS_JSON_PATH):
        print(f"Error: {CORPUS_JSON_PATH} not found.")
        sys.exit(1)

    with open(CORPUS_JSON_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    documents = []
    for item in manifest:
        # If it's a file entry
        if "file" in item and item["file"]:
            file_path = os.path.join(DATA_DIR, item["file"])
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping.")
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({
                "text": content,
                "source": item.get("source") or item["file"],
                "brand": item.get("brand", "Unknown")
            })
        # If it's a valid URL entry
        elif "url" in item and item["url"] and item["url"].startswith("http"):
            documents.append({
                "url": item["url"],
                "source": item.get("source") or item["url"],
                "brand": item.get("brand", "Unknown")
            })

    print(f"Prepared {len(documents)} documents for ingestion.")
    payload = {"documents": documents}

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            print(f"POSTing to {API_URL}...")
            response = await client.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                print("Ingestion Successful!")
                print(f"  Documents Stored: {data.get('documents_stored')}")
                print(f"  Chunks Stored:    {data.get('chunks_stored')}")
            else:
                print(f"Ingestion Failed with HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Failed to connect to OmniBound service: {e}")


if __name__ == "__main__":
    asyncio.run(load_and_ingest())
