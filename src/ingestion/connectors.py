"""
Concrete data connector implementations.
Add new connectors here to support different data sources.
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from .base import DataConnector, Record


class CSVConnector(DataConnector):
    """
    Load data from CSV files.

    Example config:
    {
        "path": "data/reviews.csv",
        "entity_col": "brand",
        "text_col": "review_text",
        "date_col": "timestamp",
        "sentiment_col": "sentiment",  # optional
        "topic_col": "category",  # optional
        "date_format": "%Y-%m-%d"  # optional, defaults to ISO format
    }
    """

    def validate_config(self) -> bool:
        required = ["path", "entity_col", "text_col", "date_col"]
        return all(k in self.config for k in required)

    def load_records(self) -> Iterator[Record]:
        if not self.validate_config():
            raise ValueError(f"CSVConnector missing required config fields: {list(self.config.keys())}")

        path = Path(self.config["path"])
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        entity_col = self.config["entity_col"]
        text_col = self.config["text_col"]
        date_col = self.config["date_col"]
        sentiment_col = self.config.get("sentiment_col")
        topic_col = self.config.get("topic_col")
        date_format = self.config.get("date_format", None)

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Parse date
                    date_str = row[date_col]
                    if date_format:
                        date = datetime.strptime(date_str, date_format)
                    else:
                        # Try common formats
                        date = self._parse_date(date_str)

                    # Extract fields
                    entity = row[entity_col].strip()
                    text = row[text_col].strip()

                    if not entity or not text or len(text) < 10:
                        continue

                    sentiment = row.get(sentiment_col) if sentiment_col else None
                    topic = row.get(topic_col) if topic_col else None

                    # Collect other columns as metadata
                    metadata = {k: v for k, v in row.items()
                               if k not in [entity_col, text_col, date_col, sentiment_col, topic_col]}

                    yield Record(
                        entity=entity,
                        text=text,
                        date=date,
                        sentiment=sentiment,
                        topic=topic,
                        metadata=metadata
                    )

                except (ValueError, KeyError) as e:
                    # Skip malformed rows
                    continue

    def _parse_date(self, date_str: str) -> datetime:
        """Try to parse date in multiple formats"""
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        # Fallback: try ISO format
        return datetime.fromisoformat(date_str)


class JSONLinesConnector(DataConnector):
    """
    Load data from JSONL (newline-delimited JSON) files.

    Example config:
    {
        "path": "data/feedback.jsonl",
        "entity_field": "product_name",
        "text_field": "comment",
        "date_field": "created_at",
        "sentiment_field": "sentiment",  # optional
        "topic_field": "category"  # optional
    }
    """

    def validate_config(self) -> bool:
        required = ["path", "entity_field", "text_field", "date_field"]
        return all(k in self.config for k in required)

    def load_records(self) -> Iterator[Record]:
        if not self.validate_config():
            raise ValueError(f"JSONLinesConnector missing required config")

        path = Path(self.config["path"])
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        entity_field = self.config["entity_field"]
        text_field = self.config["text_field"]
        date_field = self.config["date_field"]
        sentiment_field = self.config.get("sentiment_field")
        topic_field = self.config.get("topic_field")

        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)

                    entity = self._get_nested(data, entity_field)
                    text = self._get_nested(data, text_field)
                    date_str = self._get_nested(data, date_field)

                    if not entity or not text or len(text) < 10:
                        continue

                    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

                    sentiment = self._get_nested(data, sentiment_field) if sentiment_field else None
                    topic = self._get_nested(data, topic_field) if topic_field else None

                    # Store entire JSON as metadata
                    metadata = {"line_number": line_num, "raw": data}

                    yield Record(
                        entity=entity,
                        text=text,
                        date=date,
                        sentiment=sentiment,
                        topic=topic,
                        metadata=metadata
                    )

                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

    def _get_nested(self, data: Dict, field: str) -> Optional[str]:
        """Support nested fields like 'user.name' """
        if '.' in field:
            keys = field.split('.')
            value = data
            for key in keys:
                value = value.get(key)
                if value is None:
                    return None
            return str(value)
        return str(data.get(field, ''))


class DirectoryConnector(DataConnector):
    """
    Load multiple CSV/JSONL files from a directory.
    Useful for data split into multiple files by date/category.

    Example config:
    {
        "directory": "data/reviews/",
        "pattern": "*.csv",
        "connector_type": "csv",
        "connector_config": {...}  # Config for underlying CSVConnector
    }
    """

    def validate_config(self) -> bool:
        required = ["directory", "connector_type", "connector_config"]
        return all(k in self.config for k in required)

    def load_records(self) -> Iterator[Record]:
        if not self.validate_config():
            raise ValueError("DirectoryConnector missing required config")

        directory = Path(self.config["directory"])
        pattern = self.config.get("pattern", "*.*")
        connector_type = self.config["connector_type"]
        base_config = self.config["connector_config"]

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Get connector class
        connector_map = {
            "csv": CSVConnector,
            "jsonl": JSONLinesConnector,
        }
        ConnectorClass = connector_map.get(connector_type)
        if not ConnectorClass:
            raise ValueError(f"Unknown connector type: {connector_type}")

        # Load from each matching file
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                config = {**base_config, "path": str(file_path)}
                connector = ConnectorClass(config)
                yield from connector.load_records()
