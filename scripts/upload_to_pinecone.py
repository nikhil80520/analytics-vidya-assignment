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
    
    # Check if files already exist
    all_exist = True
    for file in files_to_check:
        try:
            s3_client.head_object(Bucket=s3_bucket, Key=file)
            print(f"✅ {file} already exists in S3.")
        except Exception:
            all_exist = False
            break
            
    if all_exist:
        print("✅ All raw files exist in S3. Skipping Kaggle download.")
        return
        
    print("⬇️ Downloading dataset from Kaggle...")
    settings = get_settings()
    if not settings.KAGGLE_USERNAME or not settings.KAGGLE_KEY:
        print("❌ KAGGLE_USERNAME and KAGGLE_KEY must be set in .env to download data.")
        sys.exit(1)
        
    os.environ['KAGGLE_USERNAME'] = settings.KAGGLE_USERNAME
    os.environ['KAGGLE_KEY'] = settings.KAGGLE_KEY
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    dataset = "stackoverflow/pythonquestions"
    download_path = "app/data/raw"
    os.makedirs(download_path, exist_ok=True)
    
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print("✅ Download and extraction complete.")
    
    print("☁️ Uploading raw files to S3...")
    for file in files_to_check:
        local_path = os.path.join(download_path, file)
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            print(f"   Uploading {file} ({file_size / (1024*1024):.2f} MB) to s3://{s3_bucket}/{file}...")
            
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=file) as pbar:
                s3_client.upload_file(
                    local_path, 
                    s3_bucket, 
                    file,
                    Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                )
            os.remove(local_path)
        else:
            print(f"⚠️ Warning: {file} not found in downloaded data.")


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
        print("🔄 Preprocessing raw data from S3...")
        docs = preprocess_stackoverflow_data(
            questions_path=f"s3://{s3_bucket}/Questions.csv",
            answers_path=f"s3://{s3_bucket}/Answers.csv",
            tags_path=f"s3://{s3_bucket}/Tags.csv",
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
