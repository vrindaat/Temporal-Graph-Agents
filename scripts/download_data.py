import os
import json
import datetime
from datasets import load_dataset

# CONFIGURATION
# We download these 5 major categories.
TARGET_CATEGORIES = [
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Video_Games",
    "Home_and_Kitchen",
    "Sports_and_Outdoors"
]

# Limit per category to prevent disk overflow on SCC (50k is plenty for a thesis)
LIMIT_PER_CATEGORY = 50000 

# Path (Must match main.py)
OUTPUT_DIR = "/projectnb/cs599x1/students/akhilg/directed_study_v/brand_audit/data/amazon_data"

def download_and_transform(category):
    print(f"\n[HF-DOWNLOADER] Streaming 'raw_review_{category}' from Hugging Face...")
    
    try:
        # Stream the dataset so we don't download 50GB at once
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            f"raw_review_{category}", 
            split="full", 
            streaming=True
        )
    except Exception as e:
        print(f"   [Error] Could not load {category}: {e}")
        return

    output_file = os.path.join(OUTPUT_DIR, f"{category}_2023.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        # Iterate through the stream
        for row in dataset:
            if count >= LIMIT_PER_CATEGORY:
                break
                
            # TRANSFORM 2023 FORMAT -> COMPATIBLE FORMAT
            # 2023 Data has: 'rating', 'title', 'text', 'timestamp'
            # Loader wants: 'overall', 'summary', 'reviewText', 'unixReviewTime'
            
            clean_row = {
                "overall": row.get("rating", 3.0),
                "summary": row.get("title", ""),
                "reviewText": row.get("text", ""),
                "unixReviewTime": row.get("timestamp", 0) / 1000, # Convert ms to seconds
                "asin": row.get("asin", ""),
                "brand": "" # 2023 data sometimes hides brand in metadata, but NER will fix this
            }
            
            # Write line-delimited JSON (NDJSON)
            f.write(json.dumps(clean_row) + "\n")
            count += 1
            
            if count % 5000 == 0:
                print(f"   -> Collected {count} reviews...", end='\r')

    print(f"\n   [Success] Saved {count} reviews to {output_file}")

if __name__ == "__main__":
    print("==================================================")
    print("   AMAZON 2023 DATA DOWNLOADER (HUGGING FACE)     ")
    print("==================================================")
    
    # Verify login
    print("Note: Ensure you have run 'huggingface-cli login' if accessing gated repos.")
    
    for cat in TARGET_CATEGORIES:
        download_and_transform(cat)
        
    print("\n[Done] Data is ready. Run 'python main.py' to ingest.")