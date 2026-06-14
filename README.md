# Python Programming Q&A Assistant

This project is an AI-powered Python Q&A system built for the Analytics Vidhya AI Engineer Assessment. It utilizes a robust Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-grounded answers based on the Stack Overflow Python Questions & Answers dataset.

## Architecture

The system is optimized to run entirely on **Free Tier** cloud services:
- **Vector Database**: Pinecone Cloud (Free Tier)
- **LLM & Embeddings**: AWS Bedrock (Qwen via Amazon Bedrock / Titan Text Embeddings v2)
- **Deployment**: AWS EC2 `t2.micro` (Free Tier)
- **Backend API**: FastAPI

```text
User → Nginx (EC2 t2.micro) → FastAPI → Pinecone (free) → AWS Bedrock Qwen/Titan
```

## Features

- **RAG Pipeline**: Leverages LangChain with a Pinecone vector store for fast, accurate retrieval.
- **FastAPI Backend**: Exposes a POST `/ask` endpoint for questions and a GET `/health` endpoint for monitoring.
- **Dockerized**: Containerized for easy deployment, optimized for the `t2.micro` memory constraints.

---

## 🛠️ Setup Instructions (Local)

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy the example config and fill in your actual credentials.
   ```bash
   cp .env.example .env
   ```
   *Required variables*: AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), Pinecone credentials (`PINECONE_API_KEY`).

4. **Prepare the Data**:
   Download the [Stack Overflow dataset from Kaggle](https://www.kaggle.com/datasets/stackoverflow/pythonquestions) and upload `Questions.csv`, `Answers.csv`, and `Tags.csv` to an Amazon S3 bucket. Ensure your AWS credentials have read access to this bucket. Add the bucket name to your `.env` file as `AWS_S3_BUCKET_NAME`.

5. **Upload to Pinecone**:
   Run the data ingestion script to build your vector index in the cloud.
   ```bash
   python scripts/upload_to_pinecone.py
   ```

6. **Run the API locally**:
   ```bash
   uvicorn app.main:app --reload
   ```

---

## 🚀 Deployment Guide (AWS EC2 Free Tier)

1. **Launch a Free Tier EC2 Instance**:
   - Ubuntu Server 22.04 LTS (t2.micro)
   - Ensure Security Group allows HTTP (80) and SSH (22).

2. **Transfer Code to EC2**:
   *You don't need to copy the dataset or vector database since vectors are in Pinecone!*
   ```bash
   scp -i your-key.pem -r app/ infrastructure/ docker-compose.yml Dockerfile requirements.txt .env setup_free_tier.sh ubuntu@<ec2-public-ip>:~/python-qa-aws/
   ```

3. **SSH and Setup**:
   ```bash
   ssh -i your-key.pem ubuntu@<ec2-public-ip>
   cd ~/python-qa-aws
   chmod +x setup_free_tier.sh
   ./setup_free_tier.sh
   ```

4. **Start the Application**:
   ```bash
   docker-compose up -d --build
   ```

The application will be live at `http://<ec2-public-ip>`.

---

## 🧪 Testing

A simulated test notebook/script is provided. To run real API testing once deployed, you can use:
```bash
curl -X POST "http://<your-ec2-ip>/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I reverse a list in Python?"}'
```

See `test_results.md` for a documented set of diverse queries, expected responses, and failure case observations.
