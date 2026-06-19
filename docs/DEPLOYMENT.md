# Deployment Guide - v2.0 Beta

Quick deployment guide for Temporal Graph Agent.

## ⚠️ Beta Status

**Current Version:** 2.0-beta

**Production-Ready Features:**
- ✅ Modular architecture
- ✅ YAML-based configuration
- ✅ REST API with basic auth
- ✅ Docker support
- ✅ Comprehensive documentation

**Known Limitations (coming soon):**
- ⚠️ API is synchronous (blocks on LLM calls)
- ⚠️ No rate limiting (add via reverse proxy)
- ⚠️ Basic authentication only (consider OAuth for production)
- ⚠️ No caching (regenerates reports each time)
- ⚠️ Logs via print statements (structured logging coming)

**Use Case:** Perfect for prototyping, demos, internal tools, research projects.  
**Not Yet:** High-traffic production APIs without reverse proxy.

---

## 🚀 Quick Start (Docker)

### Prerequisites
- Docker & Docker Compose installed
- Knowledge graph file (`thesis_graph.pkl`) or data to ingest
- AWS credentials (if using Bedrock)

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd temporal-graph-agent
```

### 2. Set Environment Variables

Create `.env` file:

```bash
# Security
TGA_API_KEY=your-secret-key-here

# AWS Bedrock (if using)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-2
```

### 3. Ensure Graph File Exists

```bash
# If you have a graph file:
cp /path/to/your/graph.pkl ./thesis_graph.pkl

# OR ingest new data first:
python run_pipeline.py --config configs/pipeline_amazon_reviews.yaml
```

### 4. Start Services

```bash
docker-compose up -d
```

### 5. Test It

```bash
# Health check
curl http://localhost:8000/health

# List entities
curl http://localhost:8000/entities \
  -H "X-API-Key: your-secret-key-here"

# Run audit
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key-here" \
  -d '{"entity": "Apple", "year": 2021}'
```

### 6. View Logs

```bash
docker-compose logs -f api
```

### 7. Stop Services

```bash
docker-compose down
```

---

## 🌐 Deployment Options

### Option 1: Docker Hub

**Build and Push:**
```bash
docker build -t yourusername/temporal-graph-agent:2.0-beta .
docker push yourusername/temporal-graph-agent:2.0-beta
```

**Run Anywhere:**
```bash
docker run -d \
  -p 8000:8000 \
  -e TGA_API_KEY=your-key \
  -v $(pwd)/thesis_graph.pkl:/app/thesis_graph.pkl \
  yourusername/temporal-graph-agent:2.0-beta
```

### Option 2: AWS (ECS/Fargate)

**Steps:**
1. Push image to ECR
2. Create ECS task definition
3. Create ECS service with load balancer
4. Set environment variables in task definition
5. Mount EFS for graph file (or include in image)

**Example task definition:**
```json
{
  "family": "temporal-graph-agent",
  "containerDefinitions": [{
    "name": "api",
    "image": "your-ecr-url/temporal-graph-agent:2.0-beta",
    "portMappings": [{"containerPort": 8000}],
    "environment": [
      {"name": "TGA_API_KEY", "value": "your-key"},
      {"name": "AWS_REGION", "value": "us-east-2"}
    ],
    "memory": 2048,
    "cpu": 1024
  }]
}
```

### Option 3: Google Cloud Run

**Deploy:**
```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/temporal-graph-agent

# Deploy
gcloud run deploy temporal-graph-agent \
  --image gcr.io/PROJECT-ID/temporal-graph-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars TGA_API_KEY=your-key
```

### Option 4: Hugging Face Spaces

**Steps:**
1. Create new Space (Gradio App)
2. Upload `hf_space/app.py` and dependencies
3. Upload `thesis_graph.pkl` and configs
4. Set secrets in Space settings
5. Deploy automatically

### Option 5: DigitalOcean App Platform

**Deploy:**
```bash
# Use app.yaml
doctl apps create --spec app.yaml
```

**app.yaml:**
```yaml
name: temporal-graph-agent
services:
  - name: api
    dockerfile_path: Dockerfile
    github:
      repo: your-username/temporal-graph-agent
      branch: main
    http_port: 8000
    envs:
      - key: TGA_API_KEY
        value: your-secret-key
    instance_size_slug: professional-xs
```

---

## 🔒 Security Hardening

### For Production Deployment:

1. **Use HTTPS**
   ```bash
   # Add nginx reverse proxy with SSL
   # See docker-compose.yml nginx section (commented)
   ```

2. **Rotate API Keys**
   ```bash
   # Use secrets management
   export TGA_API_KEY=$(openssl rand -hex 32)
   ```

3. **Add Rate Limiting**
   ```nginx
   # nginx.conf
   limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
   
   location /audit {
       limit_req zone=api burst=5;
       proxy_pass http://api:8000;
   }
   ```

4. **Network Security**
   ```bash
   # Don't expose API directly
   # Use reverse proxy with firewall rules
   # Whitelist IP ranges if possible
   ```

5. **Monitoring**
   ```bash
   # Add logging service
   # Monitor error rates
   # Set up alerts for anomalies
   ```

---

## 📊 Performance Tuning

### Current Performance:
- Health check: < 50ms
- List entities: < 100ms  
- Single audit: 3-15 seconds (LLM-dependent)
- Concurrent requests: 1 at a time (synchronous)

### Improvements:

**1. Use Faster LLM:**
```yaml
# configs/your_config.yaml
llm:
  model_id: us.meta.llama3-2-3b-instruct-v1:0  # Smaller = faster
```

**2. Cache Results:**
```bash
# Add Redis caching layer (coming in v2.1)
```

**3. Queue Long-Running Tasks:**
```bash
# Add Celery/RQ for background processing (coming in v2.1)
```

**4. Scale Horizontally:**
```bash
# Run multiple API containers behind load balancer
docker-compose up --scale api=3
```

---

## 🐛 Troubleshooting

### Issue: "Pipeline not initialized"
**Solution:** Check that `TGA_CONFIG` points to valid YAML and graph file exists.

### Issue: Slow responses
**Solution:** 
- Use smaller/faster LLM model
- Increase container resources (CPU/memory)
- Consider caching frequent queries

### Issue: "AWS credentials not found"
**Solution:** Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables.

### Issue: Out of memory
**Solution:**
- Increase Docker memory limit
- Use smaller graph (ingest with `limit` parameter)
- For large graphs, consider database backend (future feature)

### Issue: Container won't start
**Solution:**
```bash
# Check logs
docker-compose logs api

# Verify graph file exists
ls -lh thesis_graph.pkl

# Test locally first
python run_pipeline.py --config configs/pipeline_query_only.yaml --entity Apple --year 2021
```

---

## 📈 Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Docker health
docker ps | grep temporal-graph-agent
```

### Basic Metrics

```bash
# Request duration (add to nginx logs)
log_format timing '$request_time $upstream_response_time';

# Container stats
docker stats temporal-graph-agent_api_1
```

### Recommended Tools:
- **Uptime monitoring:** UptimeRobot, Pingdom
- **Log aggregation:** Papertrail, Loggly (coming: structured logs)
- **APM:** New Relic, DataDog (when needed)

---

## 🔄 Updates & Maintenance

### Update to New Version:

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose build
docker-compose up -d

# Or pull from Docker Hub
docker pull yourusername/temporal-graph-agent:latest
docker-compose up -d
```

### Backup Graph File:

```bash
# Regular backups
cp thesis_graph.pkl thesis_graph.$(date +%Y%m%d).pkl

# Or automated
0 2 * * * cp /path/to/thesis_graph.pkl /backup/thesis_graph.$(date +\%Y\%m\%d).pkl
```

---

## 📞 Support

**Issues:** Open GitHub issue with:
- Error message / logs
- Configuration used
- Steps to reproduce

**Questions:** See documentation:
- [README.md](README.md) - Overview
- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Configuration
- [API_GUIDE.md](API_GUIDE.md) - API usage

---

## 🗺️ Roadmap

**Coming in v2.1:**
- ✅ Async API endpoints
- ✅ Redis caching layer
- ✅ Rate limiting middleware
- ✅ Structured logging (JSON)
- ✅ Background job queue

**Coming in v3.0:**
- ✅ Database backend (Neo4j/PostgreSQL)
- ✅ Multi-tenancy support
- ✅ GraphQL API
- ✅ Real-time streaming
- ✅ Advanced visualizations

---

**Ready to deploy?** Follow the Quick Start above and you'll be running in < 5 minutes!
