import pandas as pd
import numpy as np
import re
from collections import Counter

class SentimentAnalyzer:
    def __init__(self):
        # Dictionary kata positif dan negatif Bahasa Indonesia
        self.positive_words = {
            'membantu', 'jelas', 'bagus', 'baik', 'puas', 'terbantu', 'mudah', 'suka', 
            'recommended', 'mantap', 'keren', 'seru', 'asyik', 'menyenangkan', 'bermanfaat',
            'paham', 'mengerti', 'senang', 'bangga', 'berhasil', 'lancar', 'terbaik',
            'oke', 'ok', 'great', 'good', 'excellent', 'amazing', 'fantastic', 'recommend',
            'rekomendasi', 'top', 'mantul', 'gacor', 'wow', 'puas', 'mantap', 'keren'
        }
        
        self.negative_words = {
            'sulit', 'cepat', 'kurang', 'tidak', 'bisa', 'error', 'buffering', 'terlalu', 
            'membingungkan', 'lambat', 'jelek', 'buruk', 'payah', 'gagal', 'ngerti',
            'bingung', 'stress', 'tegang', 'berat', 'capek', 'lelah', 'pusing',
            'ribet', 'repot', 'mahal', 'tidak', 'ga', 'enggak', 'bukan', 'belum',
            'pernah', 'jarang', 'minim', 'dikit', 'sedikit', 'ngebug', 'lemot'
        }
        
        self.negation_words = {'tidak', 'ga', 'enggak', 'bukan', 'belum', 'pernah', 'jangan', 'tak', 'nggak', 'gak'}
        
    def analyze_feedback(self, text):
        """Analyze single feedback text without TextBlob"""
        if not isinstance(text, str):
            return {
                'sentiment': 'neutral',
                'polarity': 0,
                'subjectivity': 0,
                'keywords_found': [],
                'text': ''
            }
        
        # Clean text
        text_clean = text.lower()
        words = text_clean.split()
        
        # Check for negation
        has_negation = any(neg in words for neg in self.negation_words)
        
        # Count positive and negative words
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        
        # If there's negation, swap the effect
        if has_negation:
            pos_count, neg_count = neg_count, pos_count
        
        # Calculate polarity
        total = pos_count + neg_count
        if total > 0:
            polarity = (pos_count - neg_count) / total
        else:
            polarity = 0
        
        # Determine sentiment
        if polarity > 0.3:
            sentiment = 'positive'
        elif polarity < -0.3:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Extract keywords found
        keywords_found = []
        for w in words:
            if w in self.positive_words:
                keywords_found.append({'word': w, 'category': 'positive'})
            elif w in self.negative_words:
                keywords_found.append({'word': w, 'category': 'negative'})
        
        return {
            'sentiment': sentiment,
            'polarity': round(polarity, 2),
            'subjectivity': 0.5,
            'keywords_found': keywords_found[:5],
            'text': text_clean
        }
    
    def analyze_batch(self, feedbacks):
        """Analyze multiple feedback texts"""
        results = []
        for text in feedbacks:
            results.append(self.analyze_feedback(text))
        return results
    
    def get_sentiment_summary(self, df):
        """Get summary statistics of sentiment analysis"""
        if 'feedback_text' not in df.columns:
            return {}
        
        results = self.analyze_batch(df['feedback_text'].tolist())
        
        # Summary statistics
        sentiments = [r['sentiment'] for r in results]
        polarities = [r['polarity'] for r in results]
        
        summary = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'), 
            'neutral': sentiments.count('neutral'),
            'avg_polarity': np.mean(polarities) if polarities else 0,
            'total_analyzed': len(results)
        }
        
        # Most common keywords
        all_keywords = []
        for r in results:
            for kw in r['keywords_found']:
                all_keywords.append(kw['word'])
        
        keyword_counts = Counter(all_keywords).most_common(10)
        summary['top_keywords'] = keyword_counts
        
        return summary

if __name__ == "__main__":
    # Test sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    sample_texts = [
        "Materi sangat membantu dan mudah dipahami!",
        "Terlalu cepat, saya ketinggalan.",
        "Platformnya bagus, tapi sering buffering.",
        "Saya suka belajar di Quipper, sangat bermanfaat",
        "Kurang jelas penjelasannya, saya bingung"
    ]
    
    for text in sample_texts:
        result = analyzer.analyze_feedback(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Polarity: {result['polarity']}")
        print(f"Keywords: {result['keywords_found']}")
        print("-" * 50)