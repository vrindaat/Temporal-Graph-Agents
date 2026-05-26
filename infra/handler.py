import json
import os
import pickle
import re
import boto3
from datetime import datetime
from collections import Counter

graph = None
bedrock = None


def load_graph():
    global graph
    if graph is not None:
        return graph
    s3 = boto3.client("s3")
    s3.download_file(os.environ["GRAPH_S3_BUCKET"], os.environ["GRAPH_S3_KEY"], "/tmp/graph.pkl")
    with open("/tmp/graph.pkl", "rb") as f:
        graph = pickle.load(f)
    return graph


def get_bedrock():
    global bedrock
    if bedrock is None:
        bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    return bedrock


def invoke_llm(prompt, max_tokens=512, temperature=0.1):
    resp = get_bedrock().converse(
        modelId=os.environ.get("BEDROCK_MODEL_ID", "meta.llama3-8b-instruct-v1:0"),
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return resp["output"]["message"]["content"][0]["text"]


def get_facts(g, brand, year):
    range_start = datetime(year, 1, 1)
    range_end = datetime(year, 12, 31, 23, 59, 59)
    facts = []
    for u, v, data in g.edges(data=True):
        if u.lower() != brand.lower() and v.lower() != brand.lower():
            continue
        d = data.get("start_date")
        if d and range_start <= d <= range_end:
            review_id = v if u.lower() == brand.lower() else u
            node = g.nodes.get(review_id, {})
            facts.append({
                "topic": data.get("topic", "General"),
                "sentiment": data.get("sentiment", "NEUTRAL"),
                "text": node.get("text", "")[:300],
                "date": d.strftime("%Y-%m-%d"),
            })
    return facts


def run_historian(g, brand, year):
    facts = get_facts(g, brand, year)
    if not facts:
        return f"No data for {brand} in {year}."
    context = "\n".join(
        [f"[{f['sentiment']}] [{f['topic']}] ({f['date']}) {f['text']}" for f in facts[:80]]
    )
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are a Brand Health Analyst. Use ONLY the data below. Do NOT hallucinate.\n"
        f"DATA ({len(facts)} reviews for {brand}, {year}):\n{context}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Generate a brand health report with: Executive Summary, Critical Issues, Strengths, Sentiment Breakdown.\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return invoke_llm(prompt, max_tokens=1024, temperature=0.4)


def run_critic(g, brand, draft, year):
    facts = get_facts(g, brand, year)
    pos = sum(1 for f in facts if f["sentiment"] == "POSITIVE")
    neg = sum(1 for f in facts if f["sentiment"] == "NEGATIVE")
    topics = Counter(f["topic"] for f in facts)
    ground_truth = (
        f"Reviews: {len(facts)} ({pos} positive, {neg} negative)\n"
        f"Topics: {dict(topics)}\n"
        + "\n".join([f"[{f['sentiment']}] {f['text'][:150]}" for f in facts[:20]])
    )
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are a Fact-Checker. Verify this report against ground truth. "
        f"Flag hallucinations, wrong sentiment, or fabricated claims.\n"
        f"GROUND TRUTH:\n{ground_truth}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"REPORT:\n{draft}\n\n"
        f"Output: STATUS: [PASS/FAIL]  REASONING: [explanation]\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    raw = invoke_llm(prompt, max_tokens=256, temperature=0.1)
    status = "PASS" if re.search(r"\bPASS\b", raw, re.I) else "FAIL" if re.search(r"\bFAIL\b", raw, re.I) else "UNKNOWN"
    return {"status": status, "reasoning": raw[:300]}


def lambda_handler(event, context):
    g = load_graph()
    path = event.get("path", "") or event.get("resource", "")
    method = event.get("httpMethod", "GET")

    if "/health" in path:
        return {"statusCode": 200, "body": json.dumps({"status": "ok", "edges": g.number_of_edges()})}

    if "/brands" in path:
        brands = sorted([n for n, d in g.nodes(data=True) if d.get("type") == "Brand"])
        return {"statusCode": 200, "body": json.dumps({"brands": brands})}

    if "/audit" in path and method == "POST":
        try:
            body = json.loads(event.get("body", "{}"))
            brand, y1, y2 = body["brand"], body["baseline_year"], body["comparison_year"]
            r1 = run_historian(g, brand, y1)
            v1 = run_critic(g, brand, r1, y1)
            r2 = run_historian(g, brand, y2)
            v2 = run_critic(g, brand, r2, y2)
            return {"statusCode": 200, "body": json.dumps({
                "brand": brand,
                "baseline_report": r1, "baseline_verdict": v1,
                "comparison_report": r2, "comparison_verdict": v2,
            })}
        except Exception as e:
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    return {"statusCode": 404, "body": json.dumps({"error": "not found"})}
