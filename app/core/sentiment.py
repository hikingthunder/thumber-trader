"""Sentiment Analysis Module — monitors news feeds and Fear & Greed index for Black Swan detection."""

import asyncio
import logging
import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from app.config import settings

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Scrapes crypto news RSS feeds and Fear & Greed index for trading signals."""
    
    FEEDS = {
        "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "cointelegraph": "https://cointelegraph.com/rss",
    }
    
    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
    
    def __init__(self):
        self.last_score: float = 50.0  # Neutral default
        self.fear_greed_value: int = 50
        self.fear_greed_label: str = "Neutral"
        self.recent_headlines: List[str] = []
        self.black_swan_detected: bool = False
        self.last_update: float = 0.0
    
    async def update(self) -> Dict[str, Any]:
        """Fetch latest sentiment data from all sources."""
        tasks = [
            self._fetch_fear_greed(),
            self._fetch_news_sentiment(),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate composite score (0-100, where 0 = extreme fear, 100 = extreme greed)
        scores = []
        
        if isinstance(results[0], dict):
            scores.append(results[0].get("value", 50))
        
        if isinstance(results[1], dict):
            scores.append(results[1].get("sentiment_score", 50))
        
        if scores:
            self.last_score = sum(scores) / len(scores)
        
        # Black Swan detection: extreme fear below threshold
        self.black_swan_detected = self.last_score < 15  # Extreme fear
        
        import time
        self.last_update = time.time()
        
        return self.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Return current sentiment status."""
        return {
            "composite_score": round(self.last_score, 1),
            "fear_greed_value": self.fear_greed_value,
            "fear_greed_label": self.fear_greed_label,
            "black_swan_detected": self.black_swan_detected,
            "recent_headlines": self.recent_headlines[:5],
            "recommendation": self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Generate trading recommendation based on sentiment."""
        if self.black_swan_detected:
            return "HALT — Black Swan event detected. Recommend pausing all trading."
        elif self.last_score < 25:
            return "CAUTION — Extreme fear. Reduce position sizes."
        elif self.last_score < 40:
            return "CAREFUL — Fear in market. Consider tightening grid."
        elif self.last_score > 75:
            return "CAUTION — Extreme greed. Watch for reversals."
        else:
            return "NORMAL — Market sentiment within normal range."
    
    async def _fetch_fear_greed(self) -> Dict[str, Any]:
        """Fetch the Crypto Fear & Greed Index."""
        def _fetch():
            req = urllib.request.Request(
                self.FEAR_GREED_URL,
                headers={"User-Agent": "ThumberTrader/2.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        
        try:
            data = await asyncio.to_thread(_fetch)
            fng_data = data.get("data", [{}])[0]
            self.fear_greed_value = int(fng_data.get("value", 50))
            self.fear_greed_label = fng_data.get("value_classification", "Neutral")
            
            return {
                "value": self.fear_greed_value,
                "label": self.fear_greed_label
            }
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed index: {e}")
            return {"value": 50, "label": "Unknown"}
    
    async def _fetch_news_sentiment(self) -> Dict[str, Any]:
        """Fetch and analyze RSS news headlines for sentiment signals."""
        headlines = []
        
        for name, url in self.FEEDS.items():
            try:
                def _fetch(feed_url=url):
                    req = urllib.request.Request(
                        feed_url,
                        headers={"User-Agent": "ThumberTrader/2.0"}
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        return resp.read().decode("utf-8")
                
                xml_str = await asyncio.to_thread(_fetch)
                root = ET.fromstring(xml_str)
                
                # Parse RSS items
                for item in root.findall(".//item")[:5]:
                    title = item.findtext("title", "")
                    if title:
                        headlines.append(title)
                        
            except Exception as e:
                logger.debug(f"Failed to fetch {name} RSS: {e}")
        
        self.recent_headlines = headlines
        
        # Simple keyword-based sentiment (production would use NLP)
        negative_keywords = ["crash", "plunge", "hack", "fraud", "ban", "collapse",
                           "panic", "crisis", "dump", "fear", "warning", "scam"]
        positive_keywords = ["surge", "rally", "bullish", "adoption", "approve",
                           "record", "growth", "institutional", "milestone"]
        
        neg_count = sum(1 for h in headlines for k in negative_keywords if k.lower() in h.lower())
        pos_count = sum(1 for h in headlines for k in positive_keywords if k.lower() in h.lower())
        
        total = neg_count + pos_count
        if total > 0:
            sentiment_score = 50 + ((pos_count - neg_count) / total) * 50
        else:
            sentiment_score = 50  # Neutral
        
        return {"sentiment_score": max(0, min(100, sentiment_score))}
    
    def should_pause_trading(self) -> bool:
        """Returns True if sentiment conditions warrant pausing the bot."""
        return self.black_swan_detected
