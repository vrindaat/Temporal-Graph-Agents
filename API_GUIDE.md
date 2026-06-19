# API Guide

REST API for Temporal Graph Agent with fact-checking capabilities.

## Quick Start

### 1. Set Configuration

```bash
# Set config file path (optional, defaults to configs/pipeline_query_only.yaml)
export TGA_CONFIG=configs/pipeline_query_only.yaml

# Set API key for basic authentication (optional)
export TGA_API_KEY=your-secret-key-here
```

### 2. Start Server

```bash
# Install dependencies
pip install fastapi uvicorn

# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Test It

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

## Endpoints

### GET /health
Health check and system stats.

**Response:**
```json
{
  "status": "healthy",
  "entities": 29,
  "edges": 337,
  "config": "Query Existing Graph"
}
```

### GET /entities
List available entities in the knowledge graph.

**Headers:**
- `X-API-Key`: API key (if configured)

**Response:**
```json
{
  "count": 29,
  "entities": ["Apple", "Samsung", "Sony", "..."]
}
```

### POST /audit
Generate temporal audit report for an entity and year.

**Request:**
```json
{
  "entity": "Apple",
  "year": 2021
}
```

**Response:**
```json
{
  "entity": "Apple",
  "year": 2021,
  "report": "## Executive Summary\nApple's brand health in 2021...",
  "verification": {
    "status": "PASS",
    "reasoning": "Passed all checks",
    "issues_found": []
  }
}
```

### POST /compare
Compare entity across two years.

**Request:**
```json
{
  "entity": "Samsung",
  "baseline_year": 2020,
  "comparison_year": 2023
}
```

**Response:**
```json
{
  "entity": "Samsung",
  "baseline": {
    "entity": "Samsung",
    "year": 2020,
    "report": "...",
    "verification": {...}
  },
  "comparison": {
    "entity": "Samsung",
    "year": 2023,
    "report": "...",
    "verification": {...}
  }
}
```

### GET /config
View current pipeline configuration (non-sensitive).

**Response:**
```json
{
  "project_name": "Query Existing Graph",
  "description": "...",
  "historian_type": "default",
  "critic_type": "robust",
  "date_range": {"min": 2014, "max": 2024}
}
```

## Authentication

Set `TGA_API_KEY` environment variable to enable API key authentication:

```bash
export TGA_API_KEY=your-secret-key-here
```

Include key in requests:
```bash
curl http://localhost:8000/entities \
  -H "X-API-Key: your-secret-key-here"
```

**No authentication:** If `TGA_API_KEY` is not set, API is open (use for development only).

## Configuration

API uses YAML configuration files. Default: `configs/pipeline_query_only.yaml`

Override with environment variable:
```bash
export TGA_CONFIG=configs/my_custom_config.yaml
```

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for configuration options.

## Error Handling

**400 Bad Request:** Invalid input
```json
{
  "detail": "Invalid entity or year"
}
```

**401 Unauthorized:** Missing or invalid API key
```json
{
  "detail": "Invalid or missing API key"
}
```

**404 Not Found:** Entity not in graph
```json
{
  "detail": "Entity 'XYZ' not found. Available: [...]"
}
```

**500 Internal Server Error:** Processing failed
```json
{
  "detail": "Audit failed: [error message]"
}
```

**503 Service Unavailable:** Pipeline not initialized
```json
{
  "detail": "Pipeline not initialized"
}
```

## Interactive Documentation

Visit http://localhost:8000/docs for Swagger UI with interactive API testing.

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Set configuration
ENV TGA_CONFIG=/app/configs/pipeline_query_only.yaml
ENV TGA_API_KEY=change-me-in-production

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t temporal-graph-agent .

# Run
docker run -p 8000:8000 \
  -e TGA_API_KEY=your-secret-key \
  -v $(pwd)/thesis_graph.pkl:/app/thesis_graph.pkl \
  temporal-graph-agent
```

### Environment Variables

- `TGA_CONFIG` - Path to config file (default: configs/pipeline_query_only.yaml)
- `TGA_API_KEY` - API key for authentication (optional)
- `AWS_ACCESS_KEY_ID` - AWS credentials (if using Bedrock)
- `AWS_SECRET_ACCESS_KEY` - AWS credentials
- `AWS_REGION` - AWS region

### Security Considerations

**For Production:**
1. ✅ Set strong `TGA_API_KEY`
2. ✅ Use HTTPS (reverse proxy with nginx/caddy)
3. ✅ Add rate limiting (nginx or API gateway)
4. ✅ Run with non-root user
5. ✅ Keep dependencies updated
6. ✅ Monitor logs for suspicious activity

## Performance

**Typical Response Times:**
- Health check: < 50ms
- List entities: < 100ms
- Single audit: 3-15 seconds (depends on LLM)
- Comparison: 6-30 seconds (two audits)

**Concurrency:**
- Current: Synchronous (one request at a time)
- For high load: Consider async implementation or queue system

## Examples

### Python Client

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-secret-key-here"

headers = {"X-API-Key": API_KEY}

# Get entities
response = requests.get(f"{API_URL}/entities", headers=headers)
entities = response.json()["entities"]

# Run audit
audit_request = {"entity": "Apple", "year": 2021}
response = requests.post(
    f"{API_URL}/audit",
    json=audit_request,
    headers=headers
)
result = response.json()

print(f"Status: {result['verification']['status']}")
print(f"Report: {result['report'][:200]}...")
```

### JavaScript/Node.js Client

```javascript
const API_URL = "http://localhost:8000";
const API_KEY = "your-secret-key-here";

const headers = {
  "Content-Type": "application/json",
  "X-API-Key": API_KEY
};

// Run audit
const response = await fetch(`${API_URL}/audit`, {
  method: "POST",
  headers: headers,
  body: JSON.stringify({
    entity: "Apple",
    year: 2021
  })
});

const result = await response.json();
console.log("Status:", result.verification.status);
console.log("Report:", result.report);
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# List entities
curl http://localhost:8000/entities \
  -H "X-API-Key: your-key"

# Single audit
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "entity": "Apple",
    "year": 2021
  }'

# Compare years
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "entity": "Samsung",
    "baseline_year": 2020,
    "comparison_year": 2023
  }'
```

## Troubleshooting

**"Pipeline not initialized"**
- Check `TGA_CONFIG` points to valid YAML file
- Verify graph file exists at path specified in config

**"Entity not found"**
- Use `/entities` endpoint to see available entities
- Check entity name matches exactly (case-insensitive)

**Slow responses**
- LLM calls are synchronous and take 3-15 seconds
- Consider caching frequent queries
- Use faster LLM model (e.g., smaller Llama)

**"AWS credentials not found"**
- Set AWS environment variables if using Bedrock
- Or use local LLM backend in config

## Next Steps

- See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for configuration
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- See [README.md](README.md) for project overview
