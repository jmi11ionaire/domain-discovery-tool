"""
AI-Enhanced B2B Publisher Discovery & Vetting System
Advanced implementation leveraging LLMs and AI for intelligent publisher analysis
"""

import asyncio
import aiohttp
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import re
import ssl
import socket
from urllib.parse import urlparse, urljoin
import json
import logging
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import openai
from anthropic import Anthropic
import tiktoken
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
import hashlib
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIPublisherScore:
    """Enhanced data class for AI-powered publisher scoring results"""
    domain: str
    brand_safety_score: float
    b2b_relevance_score: float
    authority_score: float
    overall_score: float
    
    # AI-enhanced metrics
    content_quality_ai: float
    professional_tone_score: float
    topic_relevance_scores: Dict[str, float]
    sentiment_analysis: Dict[str, float]
    fraud_risk_score: float
    competitor_similarity: float
    
    # LLM-generated insights
    content_summary: str
    quality_assessment: str
    recommendation: str
    risk_factors: List[str]
    opportunities: List[str]
    
    # Technical metrics
    traffic_estimate: Optional[int] = None
    domain_authority: Optional[float] = None
    ssl_enabled: bool = False
    has_contact_page: bool = False
    has_about_page: bool = False
    social_signals: Optional[Dict[str, int]] = None
    
    # Legacy compatibility
    content_quality: float = 0.0
    professional_keywords: int = 0
    red_flags: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.red_flags is None:
            self.red_flags = []
        if self.social_signals is None:
            self.social_signals = {}
        if self.topic_relevance_scores is None:
            self.topic_relevance_scores = {}

class AIPublisherDiscovery:
    """AI-Enhanced Publisher Discovery and Vetting System"""
    
    def __init__(self, db_path: str = "ai_publishers.db", openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        self.db_path = db_path
        self.session = None
        self.setup_database()
        
        # Initialize AI models and services
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.anthropic_client = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        
        # Load pre-trained models
        self.setup_ai_models()
        
        # Enhanced B2B categories with more nuance
        self.b2b_categories = {
            'enterprise_software': {
                'keywords': ['saas', 'enterprise', 'cloud computing', 'software solutions', 'crm', 'erp'],
                'weight': 1.2
            },
            'business_strategy': {
                'keywords': ['strategy', 'consulting', 'management', 'leadership', 'governance', 'transformation'],
                'weight': 1.1
            },
            'finance_fintech': {
                'keywords': ['fintech', 'banking', 'investment', 'financial services', 'accounting', 'treasury'],
                'weight': 1.0
            },
            'hr_talent': {
                'keywords': ['human resources', 'talent management', 'recruitment', 'workforce', 'employee'],
                'weight': 0.9
            },
            'marketing_sales': {
                'keywords': ['b2b marketing', 'sales enablement', 'lead generation', 'marketing automation'],
                'weight': 0.8
            },
            'supply_chain': {
                'keywords': ['supply chain', 'logistics', 'procurement', 'operations', 'manufacturing'],
                'weight': 1.0
            },
            'cybersecurity': {
                'keywords': ['cybersecurity', 'information security', 'data protection', 'compliance', 'privacy'],
                'weight': 1.3
            },
            'industry_specific': {
                'keywords': ['healthcare', 'real estate', 'construction', 'energy', 'telecommunications'],
                'weight': 0.9
            }
        }
        
        # Advanced fraud detection patterns
        self.fraud_indicators = {
            'domain_patterns': [
                r'[0-9]{4,}',  # Excessive numbers in domain
                r'(.)\1{3,}',  # Repeated characters
                r'[a-z]{20,}',  # Extremely long words
            ],
            'content_patterns': [
                'get rich quick', 'make money fast', 'guaranteed returns',
                'limited time offer', 'act now', 'exclusive deal',
                'miracle cure', 'secret formula', 'insider information'
            ],
            'technical_indicators': [
                'excessive_redirects', 'suspicious_ads', 'malware_detected',
                'phishing_attempt', 'fake_testimonials'
            ]
        }

    def setup_database(self):
        """Initialize enhanced SQLite database with AI metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_publishers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT UNIQUE,
                brand_safety_score REAL,
                b2b_relevance_score REAL,
                authority_score REAL,
                overall_score REAL,
                
                -- AI-enhanced metrics
                content_quality_ai REAL,
                professional_tone_score REAL,
                topic_relevance_scores TEXT,
                sentiment_analysis TEXT,
                fraud_risk_score REAL,
                competitor_similarity REAL,
                
                -- LLM insights
                content_summary TEXT,
                quality_assessment TEXT,
                recommendation TEXT,
                risk_factors TEXT,
                opportunities TEXT,
                
                -- Technical data
                traffic_estimate INTEGER,
                domain_authority REAL,
                ssl_enabled BOOLEAN,
                has_contact_page BOOLEAN,
                has_about_page BOOLEAN,
                social_signals TEXT,
                
                -- Metadata
                last_updated TIMESTAMP,
                ai_model_version TEXT,
                analysis_cost REAL,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                page_url TEXT,
                content_hash TEXT,
                extracted_text TEXT,
                ai_classification TEXT,
                topics_detected TEXT,
                quality_metrics TEXT,
                analyzed_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS competitor_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                competitor_domain TEXT,
                similarity_score REAL,
                shared_advertisers TEXT,
                content_overlap REAL,
                audience_overlap REAL,
                discovered_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def setup_ai_models(self):
        """Initialize AI models and pipelines"""
        try:
            # Load spaCy model for NLP
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        try:
            # Initialize sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except Exception as e:
            logger.warning(f"Could not load sentiment analyzer: {e}")
            self.sentiment_analyzer = None
        
        try:
            # Initialize text classification for content quality
            self.quality_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert"
            )
        except Exception as e:
            logger.warning(f"Could not load quality classifier: {e}")
            self.quality_classifier = None
        
        # Initialize TF-IDF vectorizer for content similarity
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    async def get_session(self):
        """Get or create aiohttp session with enhanced headers"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def llm_content_analysis(self, content: str, domain: str) -> Dict[str, Any]:
        """Use LLM to perform comprehensive content analysis"""
        if not self.openai_client and not self.anthropic_client:
            logger.warning("No LLM client available for content analysis")
            return self.fallback_content_analysis(content)
        
        # Truncate content for API limits
        content_sample = content[:4000] if len(content) > 4000 else content
        
        prompt = f"""
        Analyze this website content from {domain} for B2B advertising suitability:

        Content: {content_sample}

        Please provide a JSON response with:
        1. content_quality (0-100): Overall content quality and professionalism
        2. b2b_relevance (0-100): Relevance to business professionals
        3. brand_safety (0-100): Safety for brand advertising
        4. professional_tone (0-100): Professional vs casual tone
        5. topic_categories: List of detected business topics
        6. sentiment: Overall sentiment (positive/negative/neutral)
        7. fraud_risk (0-100): Risk of being fraudulent or scam
        8. summary: Brief content summary (2-3 sentences)
        9. recommendation: Accept/Review/Reject with reasoning
        10. risk_factors: List of potential risks
        11. opportunities: List of advertising opportunities

        Focus on: editorial quality, business relevance, professional audience appeal, brand safety.
        """
        
        try:
            if self.openai_client:
                response = await self.openai_content_analysis(prompt)
            else:
                response = await self.anthropic_content_analysis(prompt)
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self.fallback_content_analysis(content)

    async def openai_content_analysis(self, prompt: str) -> str:
        """OpenAI-specific content analysis"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert B2B advertising analyst. Provide accurate JSON responses only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def anthropic_content_analysis(self, prompt: str) -> str:
        """Anthropic Claude-specific content analysis"""
        if not self.anthropic_client:
            raise Exception("Anthropic client not initialized")
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def fallback_content_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback analysis when LLM is unavailable"""
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text().lower()
        
        # Basic keyword-based analysis
        b2b_score = 0
        for category, data in self.b2b_categories.items():
            for keyword in data['keywords']:
                if keyword in text:
                    b2b_score += data['weight'] * 10
        
        b2b_score = min(100, b2b_score)
        
        # Basic fraud detection
        fraud_score = 0
        for pattern in self.fraud_indicators['content_patterns']:
            if pattern.lower() in text:
                fraud_score += 20
        
        return {
            'content_quality': 60,  # Default moderate quality
            'b2b_relevance': b2b_score,
            'brand_safety': max(0, 90 - fraud_score),
            'professional_tone': 50,
            'topic_categories': ['general'],
            'sentiment': 'neutral',
            'fraud_risk': min(100, fraud_score),
            'summary': 'Automated analysis - LLM unavailable',
            'recommendation': 'Review',
            'risk_factors': ['Limited analysis capabilities'],
            'opportunities': ['Manual review recommended']
        }

    async def advanced_fraud_detection(self, domain: str, content: str) -> float:
        """AI-powered fraud and scam detection"""
        fraud_score = 0
        
        # Domain-based analysis
        for pattern in self.fraud_indicators['domain_patterns']:
            if re.search(pattern, domain):
                fraud_score += 15
        
        # Content-based analysis
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text().lower()
        
        for indicator in self.fraud_indicators['content_patterns']:
            if indicator in text:
                fraud_score += 10
        
        # Technical indicators
        suspicious_elements = [
            len(soup.find_all('iframe')) > 5,  # Excessive iframes
            len(soup.find_all('script')) > 20,  # Too many scripts
            'window.location' in content,  # Redirects
            'document.write' in content,  # Dynamic content injection
        ]
        
        fraud_score += sum(suspicious_elements) * 5
        
        # AI-based toxicity detection
        if self.quality_classifier:
            try:
                sample_text = text[:500]  # Sample for analysis
                toxicity_result = self.quality_classifier(sample_text)
                if toxicity_result[0]['label'] == 'TOXIC' and toxicity_result[0]['score'] > 0.7:
                    fraud_score += 25
            except Exception as e:
                logger.warning(f"Toxicity detection failed: {e}")
        
        return min(100, fraud_score)

    async def competitor_similarity_analysis(self, domain: str, content: str) -> float:
        """Analyze similarity to known high-quality B2B publishers"""
        # This would be enhanced with a database of competitor content
        # For now, return a basic similarity score
        
        if not hasattr(self, 'reference_publishers'):
            self.reference_publishers = {
                'forbes.com': 'business leadership innovation technology',
                'hbr.org': 'management strategy leadership business',
                'techcrunch.com': 'technology startups venture capital',
                'wired.com': 'technology innovation digital transformation'
            }
        
        soup = BeautifulSoup(content, 'html.parser')
        content_text = soup.get_text()
        
        # Calculate TF-IDF similarity
        try:
            all_texts = list(self.reference_publishers.values()) + [content_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculate similarity with reference publishers
            content_vector = tfidf_matrix[-1]
            reference_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(content_vector, reference_vectors)[0]
            max_similarity = float(np.max(similarities)) * 100
            
            return min(100, max_similarity)
            
        except Exception as e:
            logger.warning(f"Similarity analysis failed: {e}")
            return 50  # Default moderate similarity

    async def social_signals_analysis(self, domain: str) -> Dict[str, int]:
        """Analyze social media presence and engagement"""
        # This would integrate with social media APIs in production
        # For demonstration, return mock data
        
        social_signals = {
            'linkedin_followers': 0,
            'twitter_followers': 0,
            'facebook_likes': 0,
            'linkedin_shares': 0,
            'twitter_mentions': 0,
            'total_social_score': 0
        }
        
        # Mock some data based on domain authority
        if domain in ['forbes.com', 'techcrunch.com', 'hbr.org']:
            social_signals.update({
                'linkedin_followers': 500000,
                'twitter_followers': 1000000,
                'linkedin_shares': 10000,
                'total_social_score': 90
            })
        
        return social_signals

    async def analyze_domain_ai(self, domain: str) -> AIPublisherScore:
        """AI-enhanced comprehensive domain analysis"""
        logger.info(f"AI analyzing domain: {domain}")
        
        try:
            session = await self.get_session()
            
            # Ensure domain has protocol
            if not domain.startswith(('http://', 'https://')):
                url = f"https://{domain}"
            else:
                url = domain
                domain = urlparse(domain).netloc
            
            # Fetch content
            content = await self.fetch_page_content(session, url)
            if not content:
                logger.warning(f"Could not fetch content for {domain}")
                return self.create_empty_score(domain)
            
            # Store content analysis
            await self.store_content_analysis(domain, url, content)
            
            # Perform AI-enhanced analysis
            llm_analysis = await self.llm_content_analysis(content, domain)
            fraud_risk = await self.advanced_fraud_detection(domain, content)
            competitor_sim = await self.competitor_similarity_analysis(domain, content)
            social_signals = await self.social_signals_analysis(domain)
            
            # Traditional checks
            ssl_enabled = await self.check_ssl(domain)
            has_contact = self.check_contact_page(content)
            has_about = self.check_about_page(content)
            
            # Sentiment analysis
            sentiment_scores = await self.analyze_sentiment(content)
            
            # Calculate weighted overall score
            overall_score = self.calculate_ai_overall_score(
                llm_analysis, fraud_risk, competitor_sim, ssl_enabled, has_contact, has_about
            )
            
            # Create comprehensive score object
            score = AIPublisherScore(
                domain=domain,
                brand_safety_score=llm_analysis.get('brand_safety', 0),
                b2b_relevance_score=llm_analysis.get('b2b_relevance', 0),
                authority_score=competitor_sim,
                overall_score=overall_score,
                
                # AI-enhanced metrics
                content_quality_ai=llm_analysis.get('content_quality', 0),
                professional_tone_score=llm_analysis.get('professional_tone', 0),
                topic_relevance_scores={topic: 80 for topic in llm_analysis.get('topic_categories', [])},
                sentiment_analysis=sentiment_scores,
                fraud_risk_score=fraud_risk,
                competitor_similarity=competitor_sim,
                
                # LLM insights
                content_summary=llm_analysis.get('summary', ''),
                quality_assessment=f"Quality: {llm_analysis.get('content_quality', 0)}/100",
                recommendation=llm_analysis.get('recommendation', 'Review'),
                risk_factors=llm_analysis.get('risk_factors', []),
                opportunities=llm_analysis.get('opportunities', []),
                
                # Technical
                ssl_enabled=ssl_enabled,
                has_contact_page=has_contact,
                has_about_page=has_about,
                social_signals=social_signals
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error in AI analysis of {domain}: {str(e)}")
            return self.create_empty_score(domain)

    def create_empty_score(self, domain: str) -> AIPublisherScore:
        """Create empty score for failed analysis"""
        return AIPublisherScore(
            domain=domain,
            brand_safety_score=0,
            b2b_relevance_score=0,
            authority_score=0,
            overall_score=0,
            content_quality_ai=0,
            professional_tone_score=0,
            topic_relevance_scores={},
            sentiment_analysis={'neutral': 1.0},
            fraud_risk_score=100,
            competitor_similarity=0,
            content_summary='Analysis failed',
            quality_assessment='Unable to assess',
            recommendation='Reject',
            risk_factors=['Analysis failed'],
            opportunities=[]
        )

    async def analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze content sentiment using AI"""
        if not self.sentiment_analyzer:
            return {'neutral': 1.0}
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text_sample = soup.get_text()[:1000]  # Sample for analysis
            
            result = self.sentiment_analyzer(text_sample)
            sentiment_dict = {item['label']: item['score'] for item in result[0]}
            
            return sentiment_dict
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {'neutral': 1.0}

    def calculate_ai_overall_score(self, llm_analysis: Dict, fraud_risk: float, 
                                 competitor_sim: float, ssl_enabled: bool, 
                                 has_contact: bool, has_about: bool) -> float:
        """Calculate AI-enhanced overall score with sophisticated weighting"""
        
        # Base scores from LLM analysis
        brand_safety = llm_analysis.get('brand_safety', 0)
        b2b_relevance = llm_analysis.get('b2b_relevance', 0)
        content_quality = llm_analysis.get('content_quality', 0)
        professional_tone = llm_analysis.get('professional_tone', 0)
        
        # Weighted combination
        base_score = (
            brand_safety * 0.25 +
            b2b_relevance * 0.25 +
            content_quality * 0.20 +
            professional_tone * 0.15 +
            competitor_sim * 0.15
        )
        
        # Apply penalties and bonuses
        penalty_multiplier = 1.0
        
        # Fraud penalty
        if fraud_risk > 30:
            penalty_multiplier *= (1 - (fraud_risk - 30) / 100)
        
        # Technical requirements
        if not ssl_enabled:
            penalty_multiplier *= 0.85
        if not has_contact:
            penalty_multiplier *= 0.90
        if not has_about:
            penalty_multiplier *= 0.95
        
        # Quality bonus
        if content_quality > 85 and professional_tone > 80:
            penalty_multiplier *= 1.1
        
        final_score = base_score * penalty_multiplier
        return max(0, min(100, final_score))

    async def store_content_analysis(self, domain: str, url: str, content: str):
        """Store detailed content analysis in database"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Extract text content
        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO content_analysis 
            (domain, page_url, content_hash, extracted_text, analyzed_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (domain, url, content_hash, text_content[:5000], datetime.now()))  # Limit text storage
        
        conn.commit()
        conn.close()

    async def fetch_page_content(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Enhanced page content fetching with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return content
                    elif response.status in [301, 302, 307, 308]:
                        # Handle redirects
                        redirect_url = response.headers.get('Location')
                        if redirect_url:
                            url = redirect_url
                            continue
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url}, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                
        return None

    def save_ai_publisher_score(self, score: AIPublisherScore):
        """Save AI-enhanced publisher score to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO ai_publishers 
            (domain, brand_safety_score, b2b_relevance_score, authority_score, 
             overall_score, content_quality_ai, professional_tone_score, 
             topic_relevance_scores, sentiment_analysis, fraud_risk_score, 
             competitor_similarity, content_summary, quality_assessment, 
             recommendation, risk_factors, opportunities, ssl_enabled, 
             has_contact_page, has_about_page, social_signals, last_updated, 
             ai_model_version, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            score.domain, score.brand_safety_score, score.b2b_relevance_score,
            score.authority_score, score.overall_score, score.content_quality_ai,
            score.professional_tone_score, json.dumps(score.topic_relevance_scores),
            json.dumps(score.sentiment_analysis), score.fraud_risk_score,
            score.competitor_similarity, score.content_summary, score.quality_assessment,
            score.recommendation, json.dumps(score.risk_factors),
            json.dumps(score.opportunities), score.ssl_enabled,
            score.has_contact_page, score.has_about_page, json.dumps(score.social_signals),
            datetime.now(), "ai_v1.0",
            'approved' if score.overall_score >= 75 else 'rejected'
        ))
        
        conn.commit()
        conn.close()

    async def check_ssl(self, domain: str) -> bool:
        """Enhanced SSL certificate validation"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    # Additional certificate validation could be added here
                    return True
        except Exception as e:
            logger.debug(f"SSL check failed for {domain}: {e}")
            return False

    def check_contact_page(self, content: str) -> bool:
        """Enhanced contact page detection"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Look for contact links with various patterns
        contact_patterns = [
            r'contact', r'about', r'reach\s+us', r'get\s+in\s+touch',
            r'support', r'help', r'feedback'
        ]
        
        for pattern in contact_patterns:
            if soup.find('a', href=re.compile(pattern, re.I)):
                return True
            if soup.find(text=re.compile(pattern, re.I)):
                return True
        
        # Look for email addresses
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        if email_pattern.search(content):
            return True
        
        return False

    def check_about_page(self, content: str) -> bool:
        """Enhanced about page detection"""
        soup = BeautifulSoup(content, 'html.parser')
        
        about_patterns = [
            r'about\s+us', r'about', r'who\s+we\s+are', r'our\s+story',
            r'company', r'team', r'mission', r'vision'
        ]
        
        for pattern in about_patterns:
            if soup.find('a', href=re.compile(pattern, re.I)):
                return True
            if soup.find(text=re.compile(pattern, re.I)):
                return True
        
        return False

    async def batch_analyze_ai(self, domains: List[str], max_concurrent: int = 5) -> List[AIPublisherScore]:
        """Analyze multiple domains concurrently with AI enhancement"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(domain):
            async with semaphore:
                score = await self.analyze_domain_ai(domain)
                self.save_ai_publisher_score(score)
                return score
        
        results = await asyncio.gather(
            *[analyze_with_semaphore(domain) for domain in domains],
            return_exceptions=True
        )
        
        return [r for r in results if isinstance(r, AIPublisherScore)]

    def get_ai_insights_report(self, limit: int = 50) -> Dict[str, Any]:
        """Generate comprehensive AI insights report"""
        conn = sqlite3.connect(self.db_path)
        
        # Get top publishers
        top_publishers_query = '''
            SELECT domain, overall_score, content_quality_ai, professional_tone_score,
                   fraud_risk_score, competitor_similarity, recommendation,
                   content_summary, risk_factors, opportunities
            FROM ai_publishers 
            WHERE overall_score >= 75
            ORDER BY overall_score DESC 
            LIMIT ?
        '''
        
        cursor = conn.cursor()
        cursor.execute(top_publishers_query, (limit,))
        top_publishers = cursor.fetchall()
        
        # Get topic distribution
        topic_distribution = {}
        cursor.execute('SELECT topic_relevance_scores FROM ai_publishers WHERE topic_relevance_scores IS NOT NULL')
        for row in cursor.fetchall():
            try:
                topics = json.loads(row[0])
                for topic, score in topics.items():
                    if topic not in topic_distribution:
                        topic_distribution[topic] = []
                    topic_distribution[topic].append(score)
            except:
                continue
        
        # Calculate averages
        cursor.execute('''
            SELECT AVG(overall_score), AVG(content_quality_ai), AVG(fraud_risk_score),
                   AVG(professional_tone_score), COUNT(*) as total_analyzed
            FROM ai_publishers
        ''')
        avg_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'top_publishers': [
                {
                    'domain': row[0],
                    'overall_score': row[1],
                    'content_quality': row[2],
                    'professional_tone': row[3],
                    'fraud_risk': row[4],
                    'competitor_similarity': row[5],
                    'recommendation': row[6],
                    'summary': row[7],
                    'risk_factors': json.loads(row[8]) if row[8] else [],
                    'opportunities': json.loads(row[9]) if row[9] else []
                } for row in top_publishers
            ],
            'analytics': {
                'average_overall_score': avg_stats[0] or 0,
                'average_content_quality': avg_stats[1] or 0,
                'average_fraud_risk': avg_stats[2] or 0,
                'average_professional_tone': avg_stats[3] or 0,
                'total_analyzed': avg_stats[4] or 0
            },
            'topic_distribution': {
                topic: {
                    'count': len(scores),
                    'avg_score': sum(scores) / len(scores)
                } for topic, scores in topic_distribution.items()
            }
        }


# Example usage and testing
async def main_ai():
    """Example usage of the AI-enhanced publisher discovery system"""
    # Initialize with API keys (replace with actual keys)
    discovery = AIPublisherDiscovery(
        openai_api_key="your-openai-key",  # Replace with actual key
        anthropic_api_key="your-anthropic-key"  # Replace with actual key  
    )
    
    try:
        # Test domains for AI analysis
        test_domains = [
            'techcrunch.com', 'forbes.com', 'businessinsider.com',
            'hbr.org', 'wired.com', 'arstechnica.com', 'fastcompany.com'
        ]
        
        print(f"AI analyzing {len(test_domains)} domains...")
        scores = await discovery.batch_analyze_ai(test_domains, max_concurrent=3)
        
        print(f"\nAI Analysis Results for {len(scores)} domains:")
        print("-" * 100)
        
        for score in scores:
            print(f"\nDomain: {score.domain}")
            print(f"Overall Score: {score.overall_score:.1f}")
            print(f"AI Content Quality: {score.content_quality_ai:.1f}")
            print(f"Professional Tone: {score.professional_tone_score:.1f}")
            print(f"B2B Relevance: {score.b2b_relevance_score:.1f}")
            print(f"Brand Safety: {score.brand_safety_score:.1f}")
            print(f"Fraud Risk: {score.fraud_risk_score:.1f}")
            print(f"Recommendation: {score.recommendation}")
            print(f"Summary: {score.content_summary}")
            if score.risk_factors:
                print(f"Risk Factors: {', '.join(score.risk_factors)}")
            if score.opportunities:
                print(f"Opportunities: {', '.join(score.opportunities[:2])}")
        
        # Generate insights report
        insights = discovery.get_ai_insights_report(limit=20)
        
        print("\n" + "="*80)
        print("AI INSIGHTS REPORT")
        print("="*80)
        
        analytics = insights['analytics']
        print(f"Total Domains Analyzed: {analytics['total_analyzed']}")
        print(f"Average Overall Score: {analytics['average_overall_score']:.1f}")
        print(f"Average Content Quality: {analytics['average_content_quality']:.1f}")
        print(f"Average Professional Tone: {analytics['average_professional_tone']:.1f}")
        print(f"Average Fraud Risk: {analytics['average_fraud_risk']:.1f}")
        
        print("\nTop Business Topics Detected:")
        for topic, data in insights['topic_distribution'].items():
            print(f"  {topic}: {data['count']} domains, avg score {data['avg_score']:.1f}")
        
        print(f"\nTop {len(insights['top_publishers'])} Approved Publishers:")
        for pub in insights['top_publishers'][:5]:  # Show top 5
            print(f"  {pub['domain']}: {pub['overall_score']:.1f} - {pub['recommendation']}")
        
    finally:
        await discovery.close_session()


if __name__ == "__main__":
    asyncio.run(main_ai())
