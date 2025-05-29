#!/usr/bin/env python3
"""
Enterprise Discovery System - True 5K Domain Analysis
Generates and analyzes 5,000+ actual domains with Anthropic LLM validation
"""

import asyncio
import aiohttp
import pandas as pd
import sqlite3
import re
import json
import logging
import os
import time
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Anthropic LLM integration
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdInventoryResult:
    domain: str
    has_ads_txt: bool
    ads_txt_entries: int
    premium_dsps: List[str]
    direct_deals: int
    reseller_deals: int
    estimated_ad_slots: int
    inventory_score: float
    b2b_relevance: float
    overall_score: float
    final_score: Optional[float] = None
    llm_recommendation: Optional[str] = None
    llm_score: Optional[float] = None
    llm_analysis: Optional[str] = None

class EnterpriseDiscovery5K:
    """True 5K domain discovery with Anthropic LLM validation"""
    
    def __init__(self, db_path: str = "enterprise_discovery_5k.db"):
        self.db_path = db_path
        self.session = None
        self.session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
        self.setup_database()
        
        # Anthropic API setup
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if self.anthropic_key and HAS_ANTHROPIC:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
            print("✅ Anthropic LLM integration enabled")
        else:
            self.anthropic_client = None
            print("⚠️  Anthropic LLM not configured. Set ANTHROPIC_API_KEY environment variable")
        
        # Performance settings
        self.max_concurrent = 25   # Reduced for stability
        self.batch_size = 250      # Smaller batches
        self.request_delay = 0.3   # Respectful crawling
        self.timeout = 10          # Shorter timeout
        
        # Premium DSPs for scoring
        self.premium_platforms = {
            'google.com', 'googlesyndication.com', 'doubleclick.net',
            'amazon-adsystem.com', 'rubiconproject.com', 'openx.com',
            'pubmatic.com', 'appnexus.com', 'criteo.com', 'medianet.com',
            'sovrn.com', 'indexexchange.com', 'sharethrough.com', 'triplelift.com'
        }
        
        # B2B keywords
        self.b2b_keywords = [
            'business', 'enterprise', 'technology', 'finance', 'professional',
            'marketing', 'sales', 'management', 'startup', 'entrepreneur',
            'investment', 'corporate', 'industry', 'software', 'cloud', 'saas'
        ]

    def setup_database(self):
        """Setup database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discovery_results (
                id INTEGER PRIMARY KEY,
                domain TEXT UNIQUE,
                has_ads_txt BOOLEAN,
                ads_txt_entries INTEGER,
                premium_dsps TEXT,
                direct_deals INTEGER,
                reseller_deals INTEGER,
                estimated_ad_slots INTEGER,
                inventory_score REAL,
                b2b_relevance REAL,
                overall_score REAL,
                final_score REAL,
                llm_recommendation TEXT,
                llm_score REAL,
                llm_analysis TEXT,
                discovered_at TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attempted_domains (
                domain TEXT PRIMARY KEY,
                status TEXT,
                last_checked TIMESTAMP,
                attempts INTEGER DEFAULT 1,
                error_message TEXT,
                session_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    async def get_session(self):
        """Get aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=3)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; EnterpriseBot/3.0)'}
            )
        return self.session

    async def close_session(self):
        """Close session"""
        if self.session:
            await self.session.close()
            self.session = None

    def load_existing_domains(self) -> Set[str]:
        """Load existing domains"""
        existing = set()
        
        # Load from existing_domains.txt
        try:
            with open('existing_domains.txt', 'r') as f:
                for line in f:
                    domain = line.strip().replace('www.', '').lower()
                    if domain:
                        existing.add(domain)
        except FileNotFoundError:
            pass
        
        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT domain FROM discovery_results")
        for (domain,) in cursor.fetchall():
            existing.add(domain.lower())
        
        cursor.execute("SELECT domain FROM attempted_domains")
        for (domain,) in cursor.fetchall():
            existing.add(domain.lower())
        
        conn.close()
        return existing

    def generate_5k_targets(self, existing: Set[str], count: int = 5000) -> List[str]:
        """Generate 5,000+ actual domain targets"""
        print(f"🔍 Generating {count:,} domain targets...")
        
        targets = []
        
        # Tier 1: Known high-quality publishers (500 domains)
        tier1_bases = [
            'cnn', 'bbc', 'reuters', 'apnews', 'npr', 'washingtonpost', 'nytimes', 
            'usatoday', 'latimes', 'theguardian', 'wsj', 'ft', 'economist',
            'bloomberg', 'forbes', 'fortune', 'businessinsider', 'cnbc', 'marketwatch',
            'techcrunch', 'wired', 'arstechnica', 'theverge', 'engadget', 'gizmodo'
        ]
        
        for base in tier1_bases:
            extensions = ['com', 'org', 'net', 'co.uk', 'ca', 'com.au', 'de', 'fr']
            for ext in extensions:
                targets.append(f"{base}.{ext}")
                targets.append(f"www.{base}.{ext}")
                targets.append(f"news.{base}.{ext}")
                targets.append(f"business.{base}.{ext}")
        
        # Tier 2: City + News combinations (2000 domains)
        cities = [
            'atlanta', 'austin', 'baltimore', 'boston', 'charlotte', 'chicago', 'cleveland',
            'columbus', 'dallas', 'denver', 'detroit', 'houston', 'indianapolis', 'jacksonville',
            'kansas', 'lasvegas', 'losangeles', 'memphis', 'miami', 'milwaukee', 'minneapolis',
            'nashville', 'neworleans', 'newyork', 'oakland', 'oklahoma', 'omaha', 'philadelphia',
            'phoenix', 'pittsburgh', 'portland', 'raleigh', 'sacramento', 'saltlake', 'sanantonio',
            'sandiego', 'sanfrancisco', 'sanjose', 'seattle', 'stlouis', 'tampa', 'tucson',
            'virginia', 'washington', 'albuquerque', 'anchorage', 'bakersfield', 'birmingham',
            'buffalo', 'chandler', 'chesapeake', 'cincinnati', 'colorado', 'corpus', 'durham',
            'fontana', 'fortworth', 'fremont', 'fresno', 'garland', 'gilbert', 'glendale',
            'greensboro', 'henderson', 'hialeah', 'honolulu', 'huntington', 'irvine', 'jersey',
            'lexington', 'lincoln', 'lubbock', 'madison', 'mesa', 'modesto', 'montgomery',
            'newark', 'newport', 'norfolk', 'northlas', 'ontario', 'orlando', 'overland',
            'oxnard', 'plano', 'providence', 'reno', 'richmond', 'riverside', 'rochester',
            'rockford', 'santaana', 'scottsdale', 'spokane', 'springfield', 'stockton',
            'tacoma', 'tallahassee', 'toledo', 'tulsa', 'vancouver', 'wichita', 'yonkers'
        ]
        
        news_types = ['news', 'times', 'post', 'herald', 'gazette', 'daily', 'weekly', 'observer']
        
        for city in cities:
            for news_type in news_types:
                targets.append(f"{city}{news_type}.com")
                targets.append(f"{city}-{news_type}.com")
                targets.append(f"the{city}{news_type}.com")
                targets.append(f"{city}{news_type}.org")
        
        # Tier 3: Industry publications (1500 domains)
        industries = [
            'automotive', 'healthcare', 'finance', 'technology', 'retail', 'manufacturing',
            'energy', 'construction', 'education', 'agriculture', 'transportation', 'hospitality',
            'insurance', 'banking', 'pharmaceutical', 'aerospace', 'defense', 'mining',
            'oil', 'gas', 'renewable', 'solar', 'chemical', 'steel', 'textile', 'food',
            'beverage', 'entertainment', 'media', 'advertising', 'marketing', 'consulting',
            'legal', 'accounting', 'architecture', 'engineering', 'software', 'hardware',
            'telecommunications', 'cybersecurity', 'fintech', 'biotech', 'medtech', 'cleantech'
        ]
        
        industry_suffixes = ['news', 'today', 'weekly', 'magazine', 'report', 'insider', 'digest', 'wire']
        
        for industry in industries:
            for suffix in industry_suffixes:
                targets.append(f"{industry}{suffix}.com")
                targets.append(f"{industry}-{suffix}.com")
                targets.append(f"{industry}{suffix}.org")
                targets.append(f"{industry}{suffix}.net")
        
        # Tier 4: Generic news patterns (1000 domains)
        prefixes = ['local', 'daily', 'weekly', 'national', 'global', 'regional', 'metro', 'city']
        roots = ['news', 'times', 'post', 'herald', 'gazette', 'chronicle', 'observer', 'register',
                'record', 'standard', 'express', 'sun', 'star', 'mirror', 'voice', 'review']
        
        for prefix in prefixes:
            for root in roots:
                targets.append(f"{prefix}{root}.com")
                targets.append(f"{prefix}-{root}.com")
                targets.append(f"{prefix}{root}.org")
                targets.append(f"the{prefix}{root}.com")
        
        # Remove duplicates and existing domains
        unique_targets = list(set(targets))
        new_targets = [d for d in unique_targets if d.replace('www.', '').lower() not in existing]
        
        # Shuffle for variety
        random.shuffle(new_targets)
        
        print(f"✅ Generated {len(new_targets):,} unique new targets")
        return new_targets[:count]

    async def check_ads_txt(self, domain: str) -> Tuple[bool, Dict]:
        """Check ads.txt file"""
        session = await self.get_session()
        try:
            async with session.get(f"https://{domain}/ads.txt") as response:
                if response.status == 200:
                    content = await response.text()
                    return True, self.parse_ads_txt(content)
                return False, {}
        except:
            return False, {}

    def parse_ads_txt(self, content: str) -> Dict:
        """Parse ads.txt content"""
        analysis = {
            'total_entries': 0,
            'direct_deals': 0,
            'reseller_deals': 0,
            'premium_platforms': []
        }
        
        for line in content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 3:
                    platform = parts[0].strip()
                    relationship = parts[2].strip().upper()
                    
                    analysis['total_entries'] += 1
                    
                    if relationship == 'DIRECT':
                        analysis['direct_deals'] += 1
                    elif relationship == 'RESELLER':
                        analysis['reseller_deals'] += 1
                    
                    # Check for premium platforms
                    for premium in self.premium_platforms:
                        if premium in platform:
                            analysis['premium_platforms'].append(premium)
                            break
        
        return analysis

    async def analyze_page_content(self, domain: str) -> Tuple[int, float]:
        """Analyze page content"""
        session = await self.get_session()
        try:
            async with session.get(f"https://{domain}") as response:
                if response.status == 200:
                    content = await response.text()
                    ad_slots = self.count_ad_slots(content)
                    b2b_score = self.calculate_b2b_relevance(content)
                    return ad_slots, b2b_score
                return 0, 0.0
        except:
            return 0, 0.0

    def count_ad_slots(self, content: str) -> int:
        """Count potential ad slots"""
        ad_patterns = [
            r'<div[^>]*class="[^"]*ad[^"]*"',
            r'<ins[^>]*class="[^"]*adsbygoogle',
            r'<iframe[^>]*googlesyndication',
            r'<div[^>]*data-ad-slot'
        ]
        
        total = 0
        for pattern in ad_patterns:
            total += len(re.findall(pattern, content, re.IGNORECASE))
        
        if total == 0:
            div_count = len(re.findall(r'<div', content, re.IGNORECASE))
            total = min(8, div_count // 20)
        
        return total

    def calculate_b2b_relevance(self, content: str) -> float:
        """Calculate B2B relevance"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text().lower()
            
            keyword_count = 0
            for keyword in self.b2b_keywords:
                keyword_count += text.count(keyword)
            
            text_length = len(text.split())
            if text_length > 0:
                density = (keyword_count / text_length) * 1000
                return min(100, density * 20)
            return 0.0
        except:
            return 0.0

    def calculate_inventory_score(self, ads_analysis: Dict, ad_slots: int) -> float:
        """Calculate inventory score"""
        score = 25  # Base score for having ads.txt
        
        premium_count = len(set(ads_analysis.get('premium_platforms', [])))
        score += min(35, premium_count * 3)
        
        direct_deals = ads_analysis.get('direct_deals', 0)
        score += min(25, direct_deals * 1.5)
        
        score += min(15, ad_slots * 1.5)
        
        return min(100, score)

    async def llm_validate_domain(self, result: AdInventoryResult) -> Tuple[float, str, str]:
        """Validate domain using Anthropic LLM"""
        if not self.anthropic_client:
            return None, None, None
        
        try:
            prompt = f"""
Analyze this publisher domain for B2B advertising potential:

Domain: {result.domain}
Ads.txt entries: {result.ads_txt_entries}
Direct deals: {result.direct_deals}
Premium DSPs: {len(result.premium_dsps)}
Ad slots: {result.estimated_ad_slots}
B2B relevance: {result.b2b_relevance:.1f}%
Current score: {result.overall_score:.1f}

Provide:
1. APPROVE or REJECT recommendation
2. Final score (0-100)
3. Brief analysis (1-2 sentences)

Format: RECOMMENDATION|SCORE|ANALYSIS
"""

            message = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response = message.content[0].text.strip()
            parts = response.split('|')
            
            if len(parts) >= 3:
                recommendation = parts[0].strip()
                score = float(parts[1].strip())
                analysis = parts[2].strip()
                return score, recommendation, analysis
            
            return None, None, None
            
        except Exception as e:
            logger.warning(f"LLM validation failed for {result.domain}: {e}")
            return None, None, None

    async def analyze_domain(self, domain: str) -> Optional[AdInventoryResult]:
        """Analyze single domain"""
        try:
            # Check ads.txt first
            has_ads_txt, ads_analysis = await self.check_ads_txt(domain)
            if not has_ads_txt:
                await self.record_attempt(domain, "no_ads_txt")
                return None
            
            # Analyze page content
            ad_slots, b2b_relevance = await self.analyze_page_content(domain)
            
            # Calculate scores
            inventory_score = self.calculate_inventory_score(ads_analysis, ad_slots)
            overall_score = (inventory_score * 0.7) + (b2b_relevance * 0.3)
            
            result = AdInventoryResult(
                domain=domain,
                has_ads_txt=has_ads_txt,
                ads_txt_entries=ads_analysis.get('total_entries', 0),
                premium_dsps=ads_analysis.get('premium_platforms', []),
                direct_deals=ads_analysis.get('direct_deals', 0),
                reseller_deals=ads_analysis.get('reseller_deals', 0),
                estimated_ad_slots=ad_slots,
                inventory_score=inventory_score,
                b2b_relevance=b2b_relevance,
                overall_score=overall_score
            )
            
            # LLM validation for high-scoring domains
            if overall_score >= 50 and self.anthropic_client:
                llm_score, llm_rec, llm_analysis = await self.llm_validate_domain(result)
                result.llm_score = llm_score
                result.llm_recommendation = llm_rec
                result.llm_analysis = llm_analysis
                result.final_score = llm_score if llm_score else overall_score
            else:
                result.final_score = overall_score
            
            # Only return domains with final score >= 40
            if result.final_score >= 40:
                await self.record_attempt(domain, "success")
                self.save_result(result)
                return result
            else:
                await self.record_attempt(domain, "low_score")
                return None
            
        except Exception as e:
            await self.record_attempt(domain, "failed", str(e))
            return None

    async def record_attempt(self, domain: str, status: str, error_msg: str = None):
        """Record attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO attempted_domains 
            (domain, status, last_checked, attempts, error_message, session_id)
            VALUES (?, ?, ?, 1, ?, ?)
        ''', (domain, status, datetime.now(), error_msg, self.session_id))
        
        conn.commit()
        conn.close()

    def save_result(self, result: AdInventoryResult):
        """Save result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO discovery_results 
            (domain, has_ads_txt, ads_txt_entries, premium_dsps, direct_deals,
             reseller_deals, estimated_ad_slots, inventory_score, b2b_relevance,
             overall_score, final_score, llm_recommendation, llm_score, llm_analysis,
             discovered_at, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.domain, result.has_ads_txt, result.ads_txt_entries,
            json.dumps(result.premium_dsps), result.direct_deals,
            result.reseller_deals, result.estimated_ad_slots,
            result.inventory_score, result.b2b_relevance,
            result.overall_score, result.final_score,
            result.llm_recommendation, result.llm_score, result.llm_analysis,
            datetime.now(), self.session_id
        ))
        
        conn.commit()
        conn.close()

    async def batch_analyze(self, domains: List[str]) -> List[AdInventoryResult]:
        """Analyze domains in batches"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def analyze_with_limit(domain):
            async with semaphore:
                result = await self.analyze_domain(domain)
                await asyncio.sleep(self.request_delay)
                return result
        
        tasks = [analyze_with_limit(domain) for domain in domains]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        for result in results:
            if isinstance(result, AdInventoryResult):
                successful_results.append(result)
        
        return successful_results

    async def run_5k_discovery(self, target_count: int = 5000) -> List[AdInventoryResult]:
        """Run true 5K domain discovery"""
        print("🚀 ENTERPRISE 5K DOMAIN DISCOVERY")
        print("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Load existing domains
            existing = self.load_existing_domains()
            print(f"📊 Current domains: {len(existing):,}")
            print(f"📊 Target: {target_count:,} new domains")
            
            # Generate 5K targets
            targets = self.generate_5k_targets(existing, target_count)
            print(f"📊 Processing: {len(targets):,} domains")
            print(f"📊 Concurrency: {self.max_concurrent}")
            print(f"📊 Batch size: {self.batch_size:,}")
            if self.anthropic_client:
                print("📊 LLM validation: Enabled (Anthropic)")
            else:
                print("📊 LLM validation: Disabled")
            
            # Process in batches
            all_results = []
            total_processed = 0
            
            for i in range(0, len(targets), self.batch_size):
                batch = targets[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (len(targets) + self.batch_size - 1) // self.batch_size
                
                print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} domains)")
                
                batch_results = await self.batch_analyze(batch)
                all_results.extend(batch_results)
                total_processed += len(batch)
                
                print(f"   ✅ Batch complete: {len(batch_results)}/{len(batch)} successful")
                print(f"   📊 Total progress: {total_processed:,}/{len(targets):,}")
                
                # Show running totals
                if all_results:
                    avg_score = sum(r.final_score for r in all_results) / len(all_results)
                    print(f"   📈 Running average score: {avg_score:.1f}")
            
            # Final results
            print(f"\n🎯 DISCOVERY COMPLETE")
            print(f"   Total analyzed: {total_processed:,}")
            print(f"   Successful: {len(all_results):,}")
            if total_processed > 0:
                print(f"   Success rate: {(len(all_results)/total_processed)*100:.1f}%")
            
            return all_results
            
        except Exception as e:
            logger.error(f"5K discovery failed: {e}")
            return []
        
        finally:
            await self.close_session()

# Main execution
async def main():
    """Main execution for 5K discovery"""
    
    # Check for Anthropic API key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("\n🔑 ANTHROPIC API SETUP:")
        print("To enable LLM validation, set your API key:")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        print("   # Or add to your ~/.zshrc or ~/.bashrc")
        print("   # Get key from: https://console.anthropic.com/")
        print("\nContinuing without LLM validation...\n")
    
    discovery = EnterpriseDiscovery5K()
    
    try:
        # Run 5K discovery
        results = await discovery.run_5k_discovery(target_count=5000)
        
        print(f"\n🏆 FINAL RESULTS")
        print("=" * 60)
        
        if results:
            # Categorize results
            premium = [r for r in results if r.final_score >= 80]
            high_value = [r for r in results if 60 <= r.final_score < 80]
            medium_value = [r for r in results if 40 <= r.final_score < 60]
            
            print(f"🥇 Premium (80+): {len(premium):,} domains")
            print(f"🥈 High Value (60-79): {len(high_value):,} domains")
            print(f"🥉 Medium Value (40-59): {len(medium_value):,} domains")
            
            # Show top results
            if results:
                print(f"\n🏆 TOP DISCOVERIES:")
                sorted_results = sorted(results, key=lambda x: x.final_score, reverse=True)
                for i, result in enumerate(sorted_results[:10], 1):
                    llm_indicator = " 🤖" if result.llm_score else ""
                    print(f"{i:2d}. {result.domain:<25} Score: {result.final_score:.1f}{llm_indicator}")
                    if result.llm_recommendation:
                        print(f"     LLM: {result.llm_recommendation} - {result.llm_analysis}")
            
            # Export results
            df = pd.DataFrame([asdict(r) for r in results])
            filename = f"enterprise_5k_results_{discovery.session_id}.csv"
            df.to_csv(filename, index=False)
            print(f"\n📁 Results exported to: {filename}")
            
        else:
            print("No results found. Try:")
            print("1. Set ANTHROPIC_API_KEY for LLM validation")
            print("2. Check your internet connection")
            print("3. Run with smaller target count for testing")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    print("🚀 Enterprise 5K Domain Discovery System")
    print("True 5,000 domain analysis with Anthropic LLM validation\n")
    
    asyncio.run(main())
