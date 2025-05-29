"""
B2B Publisher Discovery & Vetting System
Core implementation for discovering and scoring publishers for B2B advertising
"""

import asyncio
import aiohttp
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import re
import ssl
import socket
from urllib.parse import urlparse, urljoin
import json
import logging
from dataclasses import dataclass
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PublisherScore:
    """Data class for publisher scoring results"""
    domain: str
    brand_safety_score: float
    b2b_relevance_score: float
    authority_score: float
    overall_score: float
    traffic_estimate: Optional[int] = None
    domain_authority: Optional[float] = None
    ssl_enabled: bool = False
    has_contact_page: bool = False
    has_about_page: bool = False
    content_quality: float = 0.0
    professional_keywords: int = 0
    red_flags: List[str] = None
    
    def __post_init__(self):
        if self.red_flags is None:
            self.red_flags = []

class PublisherDiscovery:
    """Main class for discovering and vetting publishers"""
    
    def __init__(self, db_path: str = "publishers.db"):
        self.db_path = db_path
        self.session = None
        self.setup_database()
        
        # B2B relevant keywords and phrases
        self.b2b_keywords = {
            'business': ['business', 'enterprise', 'corporate', 'company', 'startup', 'entrepreneur'],
            'technology': ['technology', 'tech', 'software', 'saas', 'cloud', 'ai', 'artificial intelligence'],
            'finance': ['finance', 'financial', 'investment', 'banking', 'accounting', 'economics'],
            'professional': ['leadership', 'management', 'executive', 'professional', 'career', 'workplace'],
            'industry': ['industry', 'manufacturing', 'supply chain', 'logistics', 'operations'],
            'marketing': ['marketing', 'advertising', 'sales', 'branding', 'digital marketing'],
            'legal': ['legal', 'compliance', 'regulation', 'law', 'attorney', 'lawyer']
        }
        
        # Red flag keywords that indicate inappropriate content
        self.red_flag_keywords = [
            'adult', 'casino', 'gambling', 'porn', 'xxx', 'dating', 'hookup',
            'conspiracy', 'fake news', 'clickbait', 'scam', 'fraud',
            'weapons', 'guns', 'violence', 'hate', 'extremist'
        ]
        
        # High-authority domains for reference
        self.authority_domains = [
            'wsj.com', 'ft.com', 'bloomberg.com', 'reuters.com', 'forbes.com',
            'harvard.edu', 'mit.edu', 'techcrunch.com', 'wired.com', 'espn.com',
            'washingtonpost.com', 'nytimes.com', 'cnn.com', 'bbc.com'
        ]

    def setup_database(self):
        """Initialize SQLite database for storing publisher data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS publishers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT UNIQUE,
                brand_safety_score REAL,
                b2b_relevance_score REAL,
                authority_score REAL,
                overall_score REAL,
                traffic_estimate INTEGER,
                domain_authority REAL,
                ssl_enabled BOOLEAN,
                has_contact_page BOOLEAN,
                has_about_page BOOLEAN,
                content_quality REAL,
                professional_keywords INTEGER,
                red_flags TEXT,
                last_updated TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discovery_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                source_type TEXT,
                source_url TEXT,
                discovered_at TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()

    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
        return self.session

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    def discover_from_competitors(self, competitor_domains: List[str]) -> List[str]:
        """Discover publishers by analyzing competitor ad placements"""
        # This would integrate with ad intelligence platforms in production
        # For now, return sample domains for demonstration
        sample_discoveries = [
            'businessinsider.com', 'fastcompany.com', 'inc.com', 'entrepreneur.com',
            'venturebeat.com', 'techcrunch.com', 'arstechnica.com', 'zdnet.com'
        ]
        
        # Store discoveries in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for domain in sample_discoveries:
            cursor.execute('''
                INSERT OR IGNORE INTO discovery_sources 
                (domain, source_type, source_url, discovered_at)
                VALUES (?, ?, ?, ?)
            ''', (domain, 'competitor_analysis', 'sample', datetime.now()))
        
        conn.commit()
        conn.close()
        
        return sample_discoveries

    def discover_from_industry_directories(self) -> List[str]:
        """Discover publishers from professional industry directories"""
        # In production, this would scrape industry associations, trade publications
        industry_publications = [
            'hbr.org', 'mckinsey.com', 'pwc.com', 'deloitte.com',
            'mit.edu', 'sloan.mit.edu', 'kellogg.northwestern.edu',
            'industryweek.com', 'manufacturingnews.com'
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for domain in industry_publications:
            cursor.execute('''
                INSERT OR IGNORE INTO discovery_sources 
                (domain, source_type, source_url, discovered_at)
                VALUES (?, ?, ?, ?)
            ''', (domain, 'industry_directory', 'sample', datetime.now()))
        
        conn.commit()
        conn.close()
        
        return industry_publications

    async def analyze_domain(self, domain: str) -> PublisherScore:
        """Comprehensive analysis of a single domain"""
        logger.info(f"Analyzing domain: {domain}")
        
        try:
            session = await self.get_session()
            
            # Ensure domain has protocol
            if not domain.startswith(('http://', 'https://')):
                url = f"https://{domain}"
            else:
                url = domain
                domain = urlparse(domain).netloc
            
            # Fetch homepage content
            content = await self.fetch_page_content(session, url)
            if not content:
                logger.warning(f"Could not fetch content for {domain}")
                return PublisherScore(domain, 0, 0, 0, 0)
            
            # Calculate individual scores
            brand_safety = await self.calculate_brand_safety_score(domain, content)
            b2b_relevance = await self.calculate_b2b_relevance_score(content)
            authority = await self.calculate_authority_score(domain, content)
            
            # Technical checks
            ssl_enabled = await self.check_ssl(domain)
            has_contact = self.check_contact_page(content)
            has_about = self.check_about_page(content)
            content_quality = self.assess_content_quality(content)
            
            # Calculate overall score (weighted average)
            overall_score = (
                brand_safety * 0.4 +
                b2b_relevance * 0.3 +
                authority * 0.3
            )
            
            # Apply penalties for missing technical requirements
            if not ssl_enabled:
                overall_score *= 0.8
            if not (has_contact and has_about):
                overall_score *= 0.9
            
            score = PublisherScore(
                domain=domain,
                brand_safety_score=brand_safety,
                b2b_relevance_score=b2b_relevance,
                authority_score=authority,
                overall_score=overall_score,
                ssl_enabled=ssl_enabled,
                has_contact_page=has_contact,
                has_about_page=has_about,
                content_quality=content_quality
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error analyzing {domain}: {str(e)}")
            return PublisherScore(domain, 0, 0, 0, 0)

    async def fetch_page_content(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch and return page content"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return content
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    async def calculate_brand_safety_score(self, domain: str, content: str) -> float:
        """Calculate brand safety score based on content analysis"""
        score = 100.0
        
        # Parse content with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text().lower()
        
        # Check for red flag keywords
        red_flag_count = 0
        for keyword in self.red_flag_keywords:
            if keyword in text_content:
                red_flag_count += 1
                score -= 15  # Heavy penalty for red flags
        
        # Check for excessive ads (high ad-to-content ratio)
        ad_indicators = ['advertisement', 'sponsored', 'ad-container', 'google-ads']
        ad_count = sum(1 for indicator in ad_indicators if indicator in content.lower())
        if ad_count > 10:  # Threshold for excessive ads
            score -= 20
        
        # Check for editorial standards
        if soup.find('meta', attrs={'name': 'author'}):
            score += 5
        if soup.find('meta', attrs={'name': 'description'}):
            score += 5
        
        # Domain reputation check (simplified)
        if domain in self.authority_domains:
            score = min(100, score + 10)
        
        return max(0, min(100, score))

    async def calculate_b2b_relevance_score(self, content: str) -> float:
        """Calculate B2B relevance based on content keywords and context"""
        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text().lower()
        
        # Remove extra whitespace and normalize
        text_content = ' '.join(text_content.split())
        
        total_keywords = 0
        matched_categories = set()
        
        # Count B2B keyword matches
        for category, keywords in self.b2b_keywords.items():
            category_matches = 0
            for keyword in keywords:
                count = text_content.count(keyword)
                if count > 0:
                    category_matches += count
                    matched_categories.add(category)
            total_keywords += category_matches
        
        # Base score from keyword density
        text_length = len(text_content.split())
        if text_length > 0:
            keyword_density = (total_keywords / text_length) * 1000  # Per thousand words
        else:
            keyword_density = 0
        
        # Score based on keyword density and category diversity
        base_score = min(50, keyword_density * 10)  # Max 50 points from density
        category_bonus = len(matched_categories) * 8  # Up to 56 points for 7 categories
        
        # Check for business-focused navigation and sections
        nav_sections = soup.find_all(['nav', 'menu'])
        business_nav_terms = ['business', 'finance', 'technology', 'industry', 'professional']
        nav_bonus = 0
        for nav in nav_sections:
            nav_text = nav.get_text().lower()
            for term in business_nav_terms:
                if term in nav_text:
                    nav_bonus += 2
        
        total_score = base_score + category_bonus + min(nav_bonus, 10)
        return min(100, total_score)

    async def calculate_authority_score(self, domain: str, content: str) -> float:
        """Calculate domain authority and traffic quality score"""
        score = 0
        
        # Check if it's a known high-authority domain
        if domain in self.authority_domains:
            score += 50
        
        # Analyze domain characteristics
        soup = BeautifulSoup(content, 'html.parser')
        
        # Check for quality indicators
        if soup.find('meta', attrs={'name': 'description'}):
            score += 10
        if soup.find('meta', attrs={'property': 'og:image'}):
            score += 5
        if soup.find('meta', attrs={'name': 'twitter:card'}):
            score += 5
        
        # Check for professional content structure
        if soup.find_all('article'):
            score += 10
        if soup.find('time') or soup.find(attrs={'class': re.compile('date')}):
            score += 5
        
        # Check for author bylines
        author_indicators = soup.find_all(text=re.compile(r'by\s+\w+', re.I))
        if author_indicators:
            score += 10
        
        # Domain age estimation (simplified)
        if len(domain.split('.')) == 2:  # Not a subdomain
            score += 10
        
        # Check for SSL and technical quality
        if 'https://' in content or 'ssl' in content.lower():
            score += 5
        
        return min(100, score)

    async def check_ssl(self, domain: str) -> bool:
        """Check if domain has valid SSL certificate"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    return True
        except:
            return False

    def check_contact_page(self, content: str) -> bool:
        """Check if site has contact information"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Look for contact links
        contact_links = soup.find_all('a', href=re.compile(r'contact', re.I))
        if contact_links:
            return True
        
        # Look for contact text
        contact_text = soup.find(text=re.compile(r'contact\s+us', re.I))
        if contact_text:
            return True
        
        # Look for email addresses
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        if email_pattern.search(content):
            return True
        
        return False

    def check_about_page(self, content: str) -> bool:
        """Check if site has about page"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Look for about links
        about_links = soup.find_all('a', href=re.compile(r'about', re.I))
        return len(about_links) > 0

    def assess_content_quality(self, content: str) -> float:
        """Assess overall content quality"""
        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text()
        
        # Basic quality metrics
        word_count = len(text_content.split())
        if word_count < 100:
            return 20  # Very short content
        
        # Check for proper sentence structure
        try:
            blob = TextBlob(text_content[:1000])  # Sample first 1000 chars
            sentence_count = len(blob.sentences)
            if sentence_count > 0:
                avg_sentence_length = word_count / sentence_count
                if 10 <= avg_sentence_length <= 25:  # Reasonable sentence length
                    quality_score = 70
                else:
                    quality_score = 50
            else:
                quality_score = 30
        except:
            quality_score = 50
        
        # Check for headlines and structure
        headlines = soup.find_all(['h1', 'h2', 'h3'])
        if len(headlines) >= 3:
            quality_score += 20
        elif len(headlines) >= 1:
            quality_score += 10
        
        # Check for images (indicates multimedia content)
        images = soup.find_all('img')
        if len(images) >= 3:
            quality_score += 10
        
        return min(100, quality_score)

    def save_publisher_score(self, score: PublisherScore):
        """Save publisher score to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO publishers 
            (domain, brand_safety_score, b2b_relevance_score, authority_score, 
             overall_score, ssl_enabled, has_contact_page, has_about_page, 
             content_quality, red_flags, last_updated, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            score.domain, score.brand_safety_score, score.b2b_relevance_score,
            score.authority_score, score.overall_score, score.ssl_enabled,
            score.has_contact_page, score.has_about_page, score.content_quality,
            json.dumps(score.red_flags), datetime.now(),
            'approved' if score.overall_score >= 70 else 'rejected'
        ))
        
        conn.commit()
        conn.close()

    def get_top_publishers(self, limit: int = 100) -> pd.DataFrame:
        """Get top scoring publishers from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT domain, brand_safety_score, b2b_relevance_score, 
                   authority_score, overall_score, status, last_updated
            FROM publishers 
            WHERE overall_score >= 70
            ORDER BY overall_score DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df

    async def batch_analyze(self, domains: List[str], max_concurrent: int = 10):
        """Analyze multiple domains concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(domain):
            async with semaphore:
                score = await self.analyze_domain(domain)
                self.save_publisher_score(score)
                return score
        
        results = await asyncio.gather(
            *[analyze_with_semaphore(domain) for domain in domains],
            return_exceptions=True
        )
        
        return [r for r in results if isinstance(r, PublisherScore)]

# Example usage and testing
async def main():
    """Example usage of the publisher discovery system"""
    discovery = PublisherDiscovery()
    
    try:
        # Discover publishers from various sources
        print("Discovering publishers...")
        competitor_domains = discovery.discover_from_competitors(['example.com'])
        industry_domains = discovery.discover_from_industry_directories()
        
        # Sample domains for testing
        test_domains = [
            'techcrunch.com', 'forbes.com', 'businessinsider.com',
            'hbr.org', 'wired.com', 'arstechnica.com'
        ]
        
        print(f"Analyzing {len(test_domains)} domains...")
        scores = await discovery.batch_analyze(test_domains, max_concurrent=5)
        
        print(f"\nAnalyzed {len(scores)} domains:")
        for score in scores:
            print(f"{score.domain}: Overall={score.overall_score:.1f}, "
                  f"Safety={score.brand_safety_score:.1f}, "
                  f"B2B={score.b2b_relevance_score:.1f}, "
                  f"Authority={score.authority_score:.1f}")
        
        # Get top publishers
        top_publishers = discovery.get_top_publishers(limit=10)
        print(f"\nTop {len(top_publishers)} approved publishers:")
        print(top_publishers.to_string(index=False))
        
    finally:
        await discovery.close_session()

if __name__ == "__main__":
    asyncio.run(main())
