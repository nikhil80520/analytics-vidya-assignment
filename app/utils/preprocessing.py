import os
import pandas as pd
import pickle
import s3fs
from typing import List, Optional
from langchain.schema import Document

def preprocess_stackoverflow_data(questions_path: str, answers_path: str, tags_path: str, s3_bucket: str) -> List[Document]:
    """
    Reads the StackOverflow Kaggle dataset, merges Questions and top Answers,
    and returns a list of LangChain Document objects.
    """
    print("Loading datasets (this may take a minute depending on size)...")
    
    # We load a small chunk or the whole file depending on the system limits
    # Assuming CSVs are available. In real usage, you might use chunksize.
    try:
        df_q = pd.read_csv(questions_path, encoding="ISO-8859-1")
        df_a = pd.read_csv(answers_path, encoding="ISO-8859-1")
        df_t = pd.read_csv(tags_path, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please ensure Questions.csv, Answers.csv, and Tags.csv exist at the specified S3 paths or local paths.")
        return []

    # Merge top answer (highest score) per question
    print("Merging datasets...")
    # Get highest scoring answer per question
    idx = df_a.groupby('ParentId')['Score'].idxmax()
    top_answers = df_a.loc[idx]
    
    # Group tags
    tags_grouped = df_t.groupby('Id')['Tag'].apply(lambda x: ' '.join(str(v) for v in x)).reset_index()
    
    # Merge
    merged = pd.merge(df_q, top_answers, left_on='Id', right_on='ParentId', suffixes=('_q', '_a'))
    # After suffixes, Questions.Id is now 'Id_q'. Merge tags on that column.
    merged = pd.merge(merged, tags_grouped, left_on='Id_q', right_on='Id', how='left')
    # Keep a clean 'Id' column for downstream use
    merged['Id'] = merged['Id_q']

    # Optional: Filter out negative scoring questions/answers to ensure quality
    merged = merged[(merged['Score_q'] > 0) & (merged['Score_a'] > 0)]
    
    # SORT by Score_q and LIMIT to top 1000 to prevent massive upload times
    print(f"Total questions available: {len(merged)}. Limiting to top 1000 for faster processing...")
    merged = merged.sort_values(by='Score_q', ascending=False).head(1000)
    
    print(f"Creating documents for {len(merged)} records...")
    documents = []
    
    for _, row in merged.iterrows():
        # Clean HTML tags minimally (LangChain splitters can handle text, but simple clean is good)
        import re
        q_body = re.sub('<[^<]+>', '', str(row['Body_q']))
        a_body = re.sub('<[^<]+>', '', str(row['Body_a']))
        
        text = f"Question: {row['Title']}\n\nDetails: {q_body}\n\nAnswer: {a_body}"
        
        doc = Document(
            page_content=text,
            metadata={
                'question_id': str(row['Id']),
                'title': str(row['Title']),
                'question_score': int(row['Score_q']),
                'answer_score': int(row['Score_a']),
                'tags': str(row['Tag']) if pd.notna(row['Tag']) else ''
            }
        )
        documents.append(doc)
    
    # Save the processed docs to S3 to avoid doing this again
    print(f"Saving processed documents to s3://{s3_bucket}/documents.pkl...")
    fs = s3fs.S3FileSystem()
    try:
        with fs.open(f"s3://{s3_bucket}/documents.pkl", 'wb') as f:
            pickle.dump(documents, f)
    except Exception as e:
        print(f"Failed to save to S3: {e}")
        
    return documents

def load_processed_documents(s3_bucket: str) -> Optional[List[Document]]:
    """Loads previously processed Document objects from S3 pickle."""
    fs = s3fs.S3FileSystem()
    path = f"s3://{s3_bucket}/documents.pkl"
    try:
        if fs.exists(path):
            with fs.open(path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error accessing S3: {e}")
        
    print("No processed documents found in S3.")
    return None
