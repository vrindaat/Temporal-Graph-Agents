import networkx as nx
import pickle
import os
from datetime import datetime
from src.graph.schema import Node, TemporalEdge, MarketingTopic, Sentiment # Ensure all enums are imported

class TemporalGraphEngine:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.edge_count = 0

    # --- CRITICAL MISSING FUNCTION: RESTORED ---
    def add_data(self, node_a: Node, node_b: Node, edge: TemporalEdge):
        """
        Adds a semantic connection (edge) and its endpoints (nodes) to the graph.
        """
        # 1. Cast IDs to Strings (Crucial for NetworkX stability)
        src_id = str(node_a.id)
        tgt_id = str(node_b.id)

        # 2. Add Nodes with properties
        # Note: NetworkX stores the dataclass fields (id, type, properties) as node attributes
        self.graph.add_node(src_id, type=node_a.type, properties=node_a.properties)
        self.graph.add_node(tgt_id, type=node_b.type, properties=node_b.properties)
        
        # 3. Handle Enums Safely
        # Ensures that Topic and Sentiment are stored as simple strings in the graph
        topic_val = edge.topic.value if hasattr(edge.topic, 'value') else str(edge.topic)
        sentiment_val = edge.sentiment.value if hasattr(edge.sentiment, 'value') else str(edge.sentiment)

        # 4. Add Edge
        self.graph.add_edge(
            src_id, 
            tgt_id, 
            key=self.edge_count, 
            relation=str(edge.relation), 
            start=edge.start_date, 
            end=edge.end_date,
            topic=topic_val,
            sentiment=sentiment_val
        )
        
        self.edge_count += 1
        
        # Optionally print for debugging ingestion (you can comment this out later)
        if self.edge_count % 1000 == 0:
             print(f"[Engine] Edge #{self.edge_count} added.")

    # --- EXISTING SNAPSHOT LOGIC ---
    def get_snapshot(self, date: datetime, target_brand: str = None) -> str:
        """
        Returns facts enriched with actual review text.
        """
        facts = []
        for u, v, data in self.graph.edges(data=True):
            start = data.get('start')
            end = data.get('end')
            
            # 1. TIME FILTER
            if start and start <= date and (end is None or end >= date):
                
                # 2. BRAND FILTER
                if target_brand:
                    if target_brand.lower() not in u.lower() and target_brand.lower() not in v.lower():
                        continue
                
                # 3. RETRIEVE TEXT CONTENT (The Fix!)
                # v is the Review ID. We need to look up the node properties to get the text.
                try:
                    node_props = self.graph.nodes[v].get('properties', {})
                    review_text = node_props.get('text', 'No text available')
                    # Truncate text to save tokens
                    snippet = review_text[:100] + "..." if len(review_text) > 100 else review_text
                except:
                    snippet = "(Text missing)"

                rel = data.get('relation', 'RELATED')
                topic = data.get('topic', 'General')
                sentiment = data.get('sentiment', 'Neutral')
                
                # We now include the snippet in the fact string
                facts.append(f"- Review: '{snippet}' (Topic: {topic}, Sentiment: {sentiment})")
        
        if not facts:
            return "No recorded events found for this brand in this period."
            
        # Limit to 50 facts
        return "\n".join(facts[-50:])
        
    # --- EXISTING PERSISTENCE LOGIC ---
    def save_to_disk(self, filename="graph_state.pkl"):
        """Saves the NetworkX graph object to disk."""
        print(f"[Engine] Saving graph with {self.graph.number_of_edges()} edges to {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump(self.graph, f)
        print("[Engine] Save complete.")

    def load_from_disk(self, filename="graph_state.pkl"):
        """Loads the NetworkX graph object from disk."""
        if not os.path.exists(filename):
            print(f"[Engine] Error: File {filename} not found.")
            return False
        
        print(f"[Engine] Loading graph from {filename}...")
        with open(filename, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"[Engine] Graph loaded! Contains {self.graph.number_of_edges()} edges.")
        return True