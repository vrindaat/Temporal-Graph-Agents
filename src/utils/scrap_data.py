# src/utils/scraper.py
import praw
import csv
import os
import datetime

class RedditScraper:
    def __init__(self):
        # TODO: Replace with your actual credentials
        self.reddit = praw.Reddit(
            client_id="YOUR_CLIENT_ID",
            client_secret="YOUR_CLIENT_SECRET",
            user_agent="BrandAuditBot/1.0"
        )

    def fetch_brand_data(self, brand_name: str, limit=500) -> str:
        """
        Scrapes data for a specific brand and saves it to a CSV.
        Returns the path to the new file.
        """
        print(f"\n[SCRAPER] 🌍 Live-fetching {limit} posts about '{brand_name}' from Reddit...")
        
        # Search all of Reddit for high-relevance posts
        subreddit = self.reddit.subreddit("all")
        # We use 'relevance' to ensure we get on-topic discussions
        search_results = subreddit.search(f'"{brand_name}"', limit=limit, sort='relevance')
        
        # Save to the same folder your Loader watches
        filename = f"data/reddit_data/live_{brand_name}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        count = 0
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Use the standard headers your Loader expects
            writer.writerow(['title', 'created_utc', 'subreddit', 'score', 'selftext'])
            
            for post in search_results:
                # Convert date to standard format
                dt = datetime.datetime.fromtimestamp(post.created_utc)
                date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                
                writer.writerow([
                    post.title,
                    date_str,
                    post.subreddit.display_name,
                    post.score,
                    post.selftext.replace('\n', ' ')[:500]
                ])
                count += 1
        
        print(f"[SCRAPER] ✅ Saved {count} new data points to {filename}")
        return filename