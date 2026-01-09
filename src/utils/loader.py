import csv
import glob
import os
import spacy
import json
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline 
from src.graph.schema import Node, TemporalEdge, MarketingTopic, Sentiment

class UnsupervisedLoader:
    def __init__(self, graph_engine, llm_engine):
        self.graph = graph_engine
        self.llm = llm_engine
        
        # 1. NER
        print("[LOADER] Initializing SpaCy...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("[ERROR] SpaCy model not found.")
            raise

        # 2. VADER
        print("[LOADER] Initializing VADER...")
        self.vader = SentimentIntensityAnalyzer()

        # 3. TOPIC CLASSIFIER (CPU MODE)
        # CRITICAL FIX: We set device=-1 to run on CPU. 
        # This saves GPU memory for Llama-3.
        print("[LOADER] Initializing Zero-Shot Classifier (CPU Mode)...")
        
        # Check for local model first (from your download step)
        local_model = "/projectnb/cs599x1/students/akhilg/directed_study_v/brand_audit/models/bart-large-mnli"
        
        if os.path.exists(local_model):
            print(f"   -> Loading from local: {local_model}")
            self.classifier = pipeline("zero-shot-classification", model=local_model, device=-1)
        else:
            print("   -> Loading from Hub (Requires Internet on Login Node)")
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

        self.topic_labels = [t.value for t in MarketingTopic] 
        self.topic_map = {t.value: t for t in MarketingTopic}

    def load_directory(self, data_dir: str):
        print(f"\n[LOADER] Scanning: {data_dir}")
        files = glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "*.json"))
        
        if not files:
            print("[CRITICAL] No data files found.")
            return

        print(f"[LOADER] Found {len(files)} files. Starting Ingestion...")
        
        total_edges = 0
        for file_path in files:
            try:
                edges = self._process_file(file_path)
                total_edges += edges
            except Exception as e:
                print(f"   [Error] {os.path.basename(file_path)}: {e}")

        print(f"\n[LOADER] Ingestion Complete. Total Data Points: {total_edges}")

    def _process_file(self, file_path):
        filename = os.path.basename(file_path)
        print(f" -> Mining {filename}...", end=" ")
        
        # Determine File Type
        is_json = file_path.endswith('.json')
        
        if is_json:
            with open(file_path, 'r', encoding='utf-8') as f:
                iterator = (json.loads(line) for line in f)
                return self._process_stream(iterator, filename)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames: return 0
                return self._process_stream(reader, filename)

    def _process_stream(self, iterator, filename):
        batch_data = []
        valid_count = 0
        
        # Reduce batch size to prevent RAM crashes
        BATCH_SIZE = 16 
        
        for i, row in enumerate(iterator):
            if i > 200: break # Keep limit small for testing
            
            # Normalize
            row_lower = {k.lower(): v for k, v in row.items()}
            
            text = row_lower.get('reviewtext', row_lower.get('text', ''))
            summary = row_lower.get('summary', row_lower.get('title', ''))
            full_text = f"{summary} . {text}"
            
            try:
                rating = float(row_lower.get('overall', row_lower.get('rating', 3.0)))
            except:
                rating = 3.0
                
            raw_date = row_lower.get('unixreviewtime', row_lower.get('timestamp', None))

            if len(full_text) < 15: continue

            # NER Check
            doc = self.nlp(full_text)
            brands = [e.text for e in doc.ents if e.label_ == "ORG" and len(e.text) > 2]
            valid_brands = [b for b in brands if b.lower() not in ["amazon", "seller", "usa", "china"]]
            
            if not valid_brands: continue
            
            # Add to Batch
            batch_data.append({
                'brand': valid_brands[0],
                'text': full_text,
                'rating': rating,
                'date': raw_date,
                'id': f"Rev_{filename}_{i}"
            })
            
            # Process Batch when full
            if len(batch_data) >= BATCH_SIZE:
                valid_count += self._process_batch(batch_data)
                batch_data = []

        # Process remaining
        if batch_data:
            valid_count += self._process_batch(batch_data)
            
        print(f" -> Extracted {valid_count} reviews.")
        return valid_count

    def _process_batch(self, batch):
        texts = [d['text'][:512] for d in batch] # Truncate for speed
        
        # Zero-Shot (Runs on CPU now)
        try:
            results = self.classifier(texts, candidate_labels=self.topic_labels)
        except:
            return 0

        count = 0
        for i, item in enumerate(batch):
            # Topic
            top_label = results[i]['labels'][0]
            topic_enum = self.topic_map.get(top_label, MarketingTopic.GENERAL)
            
            # Sentiment
            if item['rating'] >= 4.5: sent = Sentiment.POSITIVE
            elif item['rating'] <= 2.0: sent = Sentiment.NEGATIVE
            else:
                vs = self.vader.polarity_scores(item['text'])
                if vs['compound'] >= 0.05: sent = Sentiment.POSITIVE
                elif vs['compound'] <= -0.05: sent = Sentiment.NEGATIVE
                else: sent = Sentiment.NEUTRAL
            
            # Date
            try:
                date_obj = datetime.fromtimestamp(float(item['date'])) if item['date'] else datetime(2021, 1, 1)
            except:
                date_obj = datetime(2021, 1, 1)

            # Graph
            b_node = Node(item['brand'], "Brand")
            r_node = Node(item['id'], "Review", {"text": item['text'][:200]})
            edge = TemporalEdge(item['brand'], item['id'], "REVIEWED_IN", topic_enum, sent, date_obj)
            
            self.graph.add_data(b_node, r_node, edge)
            count += 1
        return count