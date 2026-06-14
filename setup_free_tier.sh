#!/bin/bash
# setup_free_tier.sh - Optimized for AWS Free Tier (t2.micro)

set -e

echo "🚀 Setting up Python Q&A on AWS Free Tier EC2..."

# Update
sudo apt update && sudo apt upgrade -y

# Install Docker (lightweight)
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install -y unzip
unzip -q awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Install Nginx (lightweight reverse proxy)
sudo apt install -y nginx

# Create swap file (t2.micro has only 1GB RAM)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Optimize Docker for t2.micro
sudo mkdir -p /etc/docker
echo '{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker

# Create app directory
mkdir -p ~/python-qa-aws
cd ~/python-qa-aws

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy project files:"
echo "   scp -i your-key.pem -r ./python-qa-aws/* ubuntu@ec2-ip:~/python-qa-aws/"
echo ""
echo "2. Start app:"
echo "   cd ~/analytics-vidya && docker compose up -d --build"
echo ""
echo "3. Access app at: http://your-ec2-public-ip"
echo ""
echo "⚠️  Free Tier Limits:"
echo "   - EC2 t2.micro: 750 hrs/month (1 year)"
echo "   - Pinecone: 100K vectors, 2GB storage"
echo "   - Bedrock: Pay per use (~$0.001-0.003 per 1K tokens)"
