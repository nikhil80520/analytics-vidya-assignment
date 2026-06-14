#!/usr/bin/env python3
# scripts/upload_to_pinecone.py
# Run ONCE locally to populate Pinecone index

import sys
import os
import boto3
from tqdm import tqdm

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.utils.preprocessing import preprocess_stackoverflow_data, load_processed_documents
from app.utils.pinecone_store import PineconeManager
from app.config import get_settings

def download_and_upload_kaggle_to_s3(s3_bucket: str):
    s3_client = boto3.client('s3')
    files_to_check = ['Questions.csv', 'Answers.csv', 'Tags.csv']
    print("⬆️ Checking and uploading missing local datasets to S3...")
    download_path = "app/data/raw"
    
    import threading
    from tqdm import tqdm

    class ProgressPercentage(object):
        def __init__(self, filename):
            self._filename = filename
            self._size = float(os.path.getsize(filename))
            self._seen_so_far = 0
            self._lock = threading.Lock()
            self._pbar = tqdm(total=self._size, unit='B', unit_scale=True, desc=os.path.basename(filename))

        def __call__(self, bytes_amount):
            with self._lock:
                self._seen_so_far += bytes_amount
                self._pbar.update(bytes_amount)
                if self._seen_so_far >= self._size:
                    self._pbar.close()
    
    for file in files_to_check:
        try:
            s3_client.head_object(Bucket=s3_bucket, Key=file)
            print(f"✅ {file} already exists in S3. Skipping upload.")
            continue # Skip this file!
        except Exception:
            pass # File is missing, proceed to upload

        local_path = os.path.join(download_path, file)
        if os.path.exists(local_path):
            print(f"📦 Uploading {file} to S3...")
            s3_client.upload_file(
                local_path, 
                s3_bucket, 
                file,
                Callback=ProgressPercentage(local_path)
            )
            print(f"   ✅ Uploaded {file} to s3://{s3_bucket}/{file}.")
        else:
            print(f"❌ Error: Local file not found: {local_path}")
            sys.exit(1)
            
    print("✅ Upload to S3 complete.")


def main():
    print("=" * 60)
    print("Uploading Stack Overflow Data to Pinecone Cloud")
    print("=" * 60)
    
    pinecone_manager = PineconeManager()
    
    # Step 1: Create Pinecone index
    pinecone_manager.create_index(dimension=1024)  # Titan v2 = 1024 dims
    
    # Step 2: Load or preprocess data
    s3_bucket = pinecone_manager.settings.AWS_S3_BUCKET_NAME
    if not s3_bucket:
        print("❌ AWS_S3_BUCKET_NAME environment variable is not set. Please set it in .env")
        sys.exit(1)

    # Step 1.5: Ensure raw data is in S3 (download from Kaggle if needed)
    download_and_upload_kaggle_to_s3(s3_bucket)

    print(f"\n📂 Checking for preprocessed documents in S3 ({s3_bucket})...")
    docs = load_processed_documents(s3_bucket)
    
    if not docs:
        print("🔄 Preprocessing raw data from LOCAL files (much faster than S3)...")
        docs = preprocess_stackoverflow_data(
            questions_path="app/data/raw/Questions.csv",
            answers_path="app/data/raw/Answers.csv",
            tags_path="app/data/raw/Tags.csv",
            s3_bucket=s3_bucket
        )
    
    # Step 3: Upload to Pinecone
    pinecone_manager.upload_documents(
        documents=docs,
        chunk_size=1000,
        chunk_overlap=200,
        batch_size=100  # Free tier limit
    )
    
    # Step 4: Verify
    stats = pinecone_manager.get_stats()
    print(f"\n📊 Pinecone Index Stats:")
    print(f"   Total Vectors: {stats['total_vectors']:,}")
    print(f"   Dimension: {stats['dimension']}")
    print(f"   Fullness: {stats['index_fullness']:.2%}")
    
    print("\n✅ Upload complete! Your EC2 app will now use Pinecone cloud vectors.")

if __name__ == "__main__":
    main()
