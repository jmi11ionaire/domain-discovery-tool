#!/usr/bin/env python3
"""
ENHANCED UNIFIED Domain Discovery System
Adds LLM-powered publisher discovery and domain validation optimizations
Eliminates waste by validating domains before expensive operations
"""

import asyncio
import aiohttp
import sqlite3
import re
import json
import logging
import os
import random
import csv
import socket
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from bs4 import BeautifulSoup

# Anthropic import
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScanResult:
    """Standardized result from any scanning strategy"""
    domain: str
    strategy: str  # 'strict', 'flexible', 'api_bulk'
    status: str   # 'approved', 'rejected'
    score: float
    reason: str
    details: Dict
    llm_analysis: Optional[Dict] = None
    analyzed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()

class LLMPublisherDiscovery:
    """LLM-powered discovery of high-quality publisher domains"""
    
    def __init__(self):
        self.categories = [
            'business_finance', 'technology', 'industry_trade', 
            'regional_business', 'b2b_saas', 'marketing_advertising'
        ]
        
    async def discover_publishers_by_category(self, category: str, count: int = 50) -> List[str]:
        """Use LLM to discover high-quality publishers in a category"""
        if not HAS_ANTHROPIC:
            logger.warning("Anthropic not available for publisher discovery")
            return []
            
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set")
            return []
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            
            category_prompts = {
                'business_finance': 'business, finance, economics, and investment publications',
                'technology': 'technology, software, and innovation media sites',
                'industry_trade': 'professional trade publications and industry-specific media',
                'regional_business': 'regional business journals and local business publications',
                'b2b_saas': 'B2B software, SaaS, and enterprise technology content sites',
                'marketing_advertising': 'marketing, advertising, and digital media industry sites'
            }
            
            prompt = f"""As an advertising industry expert, list {count} high-quality {category_prompts.get(category, category)} that would be valuable for B2B advertising.

Focus on:
- Established media companies with strong programmatic advertising
- Business/professional publications with premium audiences
- Sites known to work with ad networks and DSPs
- Publishers that typically have ads.txt files
- IAB member sites and trusted inventory sources
- Sites similar to Forbes, TechCrunch, AdAge, MarketWatch quality level

Requirements:
- Only return the domain name (e.g., "forbes.com")
- One domain per line
- No explanations or additional text
- Focus on domains that actually exist and have content
- Include both large and mid-tier quality publishers

Category: {category_prompts.get(category, category)}"""
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis = getattr(response.content[0], 'text', '') if response.content else ""
            
            # Parse domains from response
            domains = []
            for line in analysis.strip().split('\n'):
                line = line.strip()
                # Extract domain from various formats
                if line and not line.startswith('#'):
                    # Remove common prefixes/suffixes
                    domain = re.sub(r'^[^a-zA-Z0-9]*', '', line)
                    domain = re.sub(r'[^a-zA-Z0-9\.-]*$', '', domain)
                    domain = domain.replace('www.', '').replace('http://', '').replace('https://', '')
                    
                    # Basic domain validation
                    if '.' in domain and len(domain) > 4 and len(domain) < 50:
                        domains.append(domain.lower())
            
            logger.info(f"LLM discovered {len(domains)} domains for category: {category}")
            return domains[:count]  # Limit to requested count
            
        except Exception as e:
            logger.warning(f"LLM publisher discovery failed: {e}")
            return []
    
    async def expand_from_existing(self, known_domains: List[str], count: int = 30) -> List[str]:
        """Use successful domains to discover similar ones"""
        if not HAS_ANTHROPIC or not known_domains:
            return []
            
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return []
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            
            # Sample some successful domains
            sample_domains = random.sample(known_domains, min(10, len(known_domains)))
            
            prompt = f"""Given these successful high-quality publisher domains: {', '.join(sample_domains)}

Suggest {count} similar domains that likely exist and would have similar advertising quality.

Look for:
- Same parent companies with other digital properties
- Similar naming patterns in the industry  
- Related verticals with established players
- Competitive publications in the same space
- Regional equivalents of national publications

Requirements:
- Only return the domain name (e.g., "domain.com")
- One domain per line
- No explanations
- Focus on domains that actually exist
- Similar quality level to the provided examples"""
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis = getattr(response.content[0], 'text', '') if response.content else ""
            
            # Parse domains
            domains = []
            for line in analysis.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    domain = re.sub(r'^[^a-zA-Z0-9]*', '', line)
                    domain = re.sub(r'[^a-zA-Z0-9\.-]*$', '', domain)
                    domain = domain.replace('www.', '').replace('http://', '').replace('https://', '')
                    
                    if '.' in domain and len(domain) > 4 and len(domain) < 50:
                        domains.append(domain.lower())
            
            logger.info(f"LLM expanded {len(domains)} domains from existing examples")
            return domains[:count]
            
        except Exception as e:
            logger.warning(f"LLM domain expansion failed: {e}")
            return []

class DomainValidator:
    """Fast domain validation to avoid wasting resources"""
    
    def __init__(self):
        self.session = None
        self.dns_cache = {}
        
    async def get_session(self):
        """Get validation session with short timeouts"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; DomainValidator/1.0)'}
            )
        return self.session
    
    async def close_session(self):
        """Close validation session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def dns_resolves(self, domain: str) -> bool:
        """Quick DNS resolution check"""
        if domain in self.dns_cache:
            return self.dns_cache[domain]
        
        try:
            socket.gethostbyname(domain)
            self.dns_cache[domain] = True
            return True
        except socket.gaierror:
            self.dns_cache[domain] = False
            return False
    
    async def is_reachable(self, domain: str) -> bool:
        """Quick HEAD request to check if domain is reachable"""
        if not self.dns_resolves(domain):
            return False
        
        session = await self.get_session()
        
        # Try both HTTPS and HTTP
        for protocol in ['https', 'http']:
            try:
                async with session.head(f"{protocol}://{domain}", allow_redirects=True) as response:
                    # Accept any reasonable HTTP status
                    if response.status in [200, 301, 302, 403, 401]:  # Even 403/401 means site exists
                        return True
            except:
                continue
        
        return False
    
    async def quick_validate(self, domain: str) -> bool:
        """Fast validation before expensive operations"""
        logger.debug(f"🔍 Quick validating: {domain}")
        
        # Basic format check
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,}$', domain):
            return False
        
        # DNS + reachability
        is_reachable = await self.is_reachable(domain)
        
        if is_reachable:
            logger.debug(f"✅ {domain} validated")
        else:
            logger.debug(f"❌ {domain} not reachable")
            
        return is_reachable
    
    async def batch_validate(self, domains: List[str], max_concurrent: int = 10) -> List[str]:
        """Validate multiple domains concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_with_semaphore(domain: str) -> Optional[str]:
            async with semaphore:
                if await self.quick_validate(domain):
                    return domain
                return None
        
        tasks = [validate_with_semaphore(domain) for domain in domains]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_domains = []
        for result in results:
            if isinstance(result, str):
                valid_domains.append(result)
        
        logger.info(f"📊 Validation: {len(valid_domains)}/{len(domains)} domains reachable")
        return valid_domains

class EnhancedUnifiedDomainScanner:
    """Enhanced unified domain scanner with LLM discovery and validation"""
    
    def __init__(self, db_path: str = "enhanced_domain_discovery.db"):
        self.db_path = db_path
        self.session = None
        self.validator = DomainValidator()
        self.llm_discovery = LLMPublisherDiscovery()
        
        # Initialize database
        self.setup_database()
        
        # Load existing domains
        self.existing_domains = self.load_existing_domains()
        
        # Premium DSPs for ads.txt analysis
        self.premium_platforms = {
            'google.com', 'googlesyndication.com', 'doubleclick.net',
            'amazon-adsystem.com', 'rubiconproject.com', 'openx.com',
            'pubmatic.com', 'appnexus.com', 'criteo.com', 'medianet.com',
            'sovrn.com', 'indexexchange.com', 'sharethrough.com', 'triplelift.com'
        }
        
        # B2B keywords for relevance scoring
        self.b2b_keywords = [
            'business', 'enterprise', 'technology', 'finance', 'professional',
            'marketing', 'sales', 'management', 'startup', 'entrepreneur',
            'investment', 'corporate', 'industry', 'software', 'cloud'
        ]
    
    def setup_database(self):
        """Setup enhanced database with validation tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domains_analyzed (
                domain TEXT PRIMARY KEY,
                first_analyzed_at TIMESTAMP,
                last_analyzed_at TIMESTAMP,
                analysis_count INTEGER DEFAULT 1,
                current_status TEXT,
                current_score REAL,
                strategy_used TEXT,
                discovery_source TEXT DEFAULT 'llm',
                validated BOOLEAN DEFAULT FALSE,
                validation_time REAL DEFAULT 0
            )
        ''')
        
        # Validation cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domain_validation_cache (
                domain TEXT PRIMARY KEY,
                is_reachable BOOLEAN,
                last_checked TIMESTAMP,
                check_count INTEGER DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Enhanced database initialized")
    
    def load_existing_domains(self) -> Set[str]:
        """Load existing domains from source of truth"""
        existing = set()
        try:
            with open('existing_domains.txt', 'r') as f:
                for line in f:
                    domain = line.strip().replace('www.', '')
                    if domain:
                        existing.add(domain)
                        
            logger.info(f"Loaded {len(existing)} existing domains")
        except FileNotFoundError:
            logger.warning("existing_domains.txt not found")
        
        return existing
    
    async def get_session(self):
        """Get aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; EnhancedDomainScanner/1.0)'}
            )
        return self.session
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
        await self.validator.close_session()
    
    def fallback_domain_discovery(self, count: int = 100) -> List[str]:
        """Fallback domain discovery when LLM is unavailable"""
        logger.info(f"📋 Using fallback domain discovery for {count} domains...")
        
        # High-quality publisher patterns based on industry knowledge
        quality_publishers = [
            # Technology
            'venturebeat.com', 'techreport.com', 'digitaltrends.com', 'engadget.com',
            'theverge.com', 'techworld.com', 'computerworld.com', 'informationweek.com',
            
            # Business/Finance
            'businessinsider.com', 'fortune.com', 'fastcompany.com', 'inc.com',
            'entrepreneur.com', 'marketwatch.com', 'cnbc.com', 'reuters.com',
            
            # Marketing/Advertising
            'adweek.com', 'mediapost.com', 'digiday.com', 'marketingland.com',
            'clickz.com', 'searchengineland.com', 'contentmarketinginstitute.com',
            
            # Industry Trade
            'industrydive.com', 'manufacturingtalk.com', 'automotivenews.com',
            'retaildive.com', 'constructiondive.com', 'supplychaindive.com',
            
            # B2B/SaaS
            'saasmetrics.co', 'firstround.com', 'a16z.com', 'techstars.com',
            'ycombinator.com', 'crunchbase.com', 'pitchbook.com'
        ]
        
        # Filter out existing domains and randomize
        candidates = [d for d in quality_publishers if d not in self.existing_domains]
        random.shuffle(candidates)
        
        return candidates[:count]
    
    async def intelligent_domain_discovery(self, count: int = 100) -> List[str]:
        """Use LLM to discover high-quality publisher domains with fallback"""
        logger.info(f"🧠 LLM discovering {count} high-quality publishers...")
        
        all_domains = []
        
        # Try LLM discovery first
        try:
            # Discover from multiple categories
            per_category = count // len(self.llm_discovery.categories)
            
            for category in self.llm_discovery.categories:
                logger.info(f"🔍 Discovering {category} publishers...")
                category_domains = await self.llm_discovery.discover_publishers_by_category(
                    category, per_category
                )
                all_domains.extend(category_domains)
            
            # Expand from existing successful domains
            if self.existing_domains:
                logger.info("🔄 Expanding from existing successful domains...")
                expanded_domains = await self.llm_discovery.expand_from_existing(
                    list(self.existing_domains), count // 4
                )
                all_domains.extend(expanded_domains)
        
        except Exception as e:
            logger.warning(f"LLM discovery failed: {e}")
        
        # Fallback if LLM discovery didn't work well
        if len(all_domains) < count // 2:
            logger.info("🔄 LLM discovery insufficient, using fallback...")
            fallback_domains = self.fallback_domain_discovery(count)
            all_domains.extend(fallback_domains)
        
        # Remove duplicates and existing domains
        unique_domains = []
        seen = set()
        
        for domain in all_domains:
            if domain not in seen and domain not in self.existing_domains:
                unique_domains.append(domain)
                seen.add(domain)
        
        logger.info(f"🎯 Total discovered {len(unique_domains)} unique candidate domains")
        return unique_domains[:count]
    
    async def validate_and_filter_domains(self, domains: List[str]) -> List[str]:
        """Validate domains before expensive scanning operations"""
        logger.info(f"⚡ Validating {len(domains)} domains...")
        
        validated_domains = await self.validator.batch_validate(domains)
        
        efficiency = (len(validated_domains) / len(domains) * 100) if domains else 0
        logger.info(f"📊 Validation efficiency: {efficiency:.1f}% ({len(validated_domains)}/{len(domains)} reachable)")
        
        return validated_domains
    
    async def check_ads_txt(self, domain: str) -> Tuple[bool, Dict]:
        """Check and analyze ads.txt file"""
        session = await self.get_session()
        
        try:
            async with session.get(f"https://{domain}/ads.txt") as response:
                if response.status == 200:
                    content = await response.text()
                    return True, self.parse_ads_txt(content)
                return False, {}
        except Exception:
            return False, {}
    
    def parse_ads_txt(self, content: str) -> Dict:
        """Parse ads.txt content for metrics"""
        lines = content.strip().split('\n')
        analysis = {
            'total_entries': 0,
            'direct_deals': 0,
            'reseller_deals': 0,
            'premium_platforms': []
        }
        
        for line in lines:
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
    
    async def analyze_content(self, domain: str) -> Tuple[float, Dict]:
        """Analyze page content for ad slots and quality"""
        session = await self.get_session()
        content_details = {
            'has_content': False,
            'ad_slots_detected': 0,
            'quality_indicators': 0,
            'b2b_relevance': 0,
            'page_text': ''
        }
        
        try:
            async with session.get(f"https://{domain}", allow_redirects=True) as response:
                if response.status == 200:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    content_details['has_content'] = True
                    
                    # Get clean text for analysis
                    for script in soup(["script", "style", "nav", "footer"]):
                        script.decompose()
                    
                    text_content = soup.get_text().lower()
                    content_details['page_text'] = text_content[:2000]  # First 2000 chars for LLM
                    
                    # Count ad slots
                    ad_elements = soup.find_all(['div', 'iframe'], 
                                              class_=re.compile(r'ad|banner|advertisement|sponsor'))
                    adsense_scripts = soup.find_all('script', src=re.compile(r'googlesyndication'))
                    content_details['ad_slots_detected'] = len(ad_elements) + len(adsense_scripts)
                    
                    # Quality indicators
                    quality_terms = ['subscribe', 'newsletter', 'premium', 'insights', 'analysis']
                    for term in quality_terms:
                        if term in text_content:
                            content_details['quality_indicators'] += 1
                    
                    # B2B relevance
                    b2b_count = sum(1 for keyword in self.b2b_keywords if keyword in text_content)
                    content_details['b2b_relevance'] = min(100, (b2b_count / len(self.b2b_keywords)) * 100)
                    
        except Exception as e:
            logger.debug(f"Content analysis failed for {domain}: {e}")
        
        # Calculate content score
        score = 0
        if content_details['has_content']:
            score += 20
        score += min(30, content_details['ad_slots_detected'] * 5)
        score += min(25, content_details['quality_indicators'] * 5)
        score += content_details['b2b_relevance'] * 0.25
        
        return min(100, score), content_details
    
    def save_analysis(self, result: ScanResult):
        """Save analysis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO domains_analyzed 
            (domain, first_analyzed_at, last_analyzed_at, analysis_count, 
             current_status, current_score, strategy_used, discovery_source, validated)
            VALUES (?, 
                COALESCE((SELECT first_analyzed_at FROM domains_analyzed WHERE domain = ?), ?),
                ?, 
                COALESCE((SELECT analysis_count FROM domains_analyzed WHERE domain = ?) + 1, 1),
                ?, ?, ?, ?, ?)
        ''', (result.domain, result.domain, result.analyzed_at, result.analyzed_at, 
              result.domain, result.status, result.score, result.strategy, 
              result.details.get('discovery_source', 'llm'),
              result.details.get('validated', False)))
        
        conn.commit()
        conn.close()
    
    async def enhanced_scan_domain(self, domain: str, strategy: str = 'flexible', 
                                 discovery_source: str = 'llm') -> Optional[ScanResult]:
        """Enhanced domain scanning with validation"""
        
        # Skip existing domains
        if domain in self.existing_domains:
            logger.info(f"⏩ Skipping {domain} - in existing domains list")
            return None
        
        # Quick validation before expensive operations
        validation_start = asyncio.get_event_loop().time()
        if not await self.validator.quick_validate(domain):
            validation_time = asyncio.get_event_loop().time() - validation_start
            logger.info(f"❌ {domain} failed validation ({validation_time:.2f}s) - not reachable")
            return ScanResult(
                domain=domain,
                strategy=strategy,
                status='rejected',
                score=0,
                reason='domain_not_reachable',
                details={'discovery_source': discovery_source, 'validated': False, 'validation_time': validation_time}
            )
        
        validation_time = asyncio.get_event_loop().time() - validation_start
        logger.info(f"✅ {domain} validated ({validation_time:.2f}s), proceeding with {strategy} scan")
        
        # Simplified flexible scan for validated domains
        try:
            # Check ads.txt (optional for flexible)
            has_ads_txt, ads_analysis = await self.check_ads_txt(domain)
            
            # Analyze content (critical for flexible strategy)
            content_score, content_details = await self.analyze_content(domain)
            
            if content_score < 20:  # Must have some content
                result = ScanResult(
                    domain=domain,
                    strategy=strategy,
                    status='rejected',
                    score=content_score,
                    reason='insufficient_content',
                    details={'discovery_source': discovery_source, 'validated': True, 'validation_time': validation_time}
                )
            else:
                # Flexible scoring algorithm
                score = content_score * 0.7  # Content is primary
                
                if has_ads_txt:
                    score += 25  # Bonus for ads.txt
                    score += min(15, len(set(ads_analysis.get('premium_platforms', []))) * 3)
                
                final_score = min(100, score)
                status = 'approved' if final_score >= 45 else 'rejected'
                
                result = ScanResult(
                    domain=domain,
                    strategy=strategy,
                    status=status,
                    score=final_score,
                    reason=f"{strategy}_{status}",
                    details={
                        'discovery_source': discovery_source,
                        'validated': True,
                        'validation_time': validation_time,
                        'ads_txt_found': has_ads_txt,
                        'premium_dsps': len(set(ads_analysis.get('premium_platforms', []))) if has_ads_txt else 0,
                        'content_score': content_score
                    }
                )
            
            # Save result
            self.save_analysis(result)
            logger.info(f"🎯 {result.status.upper()}: {domain} (Score: {result.score:.1f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced scan failed for {domain}: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM domains_analyzed")
        total_analyzed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM domains_analyzed WHERE current_status = 'approved'")
        approved = cursor.fetchone()[0]
        
        cursor.execute("SELECT discovery_source, COUNT(*) FROM domains_analyzed GROUP BY discovery_source")
        source_stats = dict(cursor.fetchall())
        
        cursor.execute("SELECT AVG(validation_time) FROM domains_analyzed WHERE validated = TRUE")
        avg_validation_time = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_analyzed': total_analyzed,
            'approved': approved,
            'success_rate': (approved / total_analyzed * 100) if total_analyzed > 0 else 0,
            'discovery_sources': source_stats,
            'avg_validation_time': avg_validation_time
        }

# Enhanced pipeline function
async def run_enhanced_discovery_pipeline(target_count: int = 50) -> Dict:
    """Enhanced discovery pipeline with LLM and validation"""
    scanner = EnhancedUnifiedDomainScanner()
    
    try:
        print("🚀 ENHANCED DOMAIN DISCOVERY PIPELINE")
        print("=====================================")
        
        # Step 1: LLM Discovery
        discovered_domains = await scanner.intelligent_domain_discovery(target_count * 2)
        print(f"🧠 LLM discovered: {len(discovered_domains)} candidate domains")
        
        # Step 2: Validation 
        validated_domains = await scanner.validate_and_filter_domains(discovered_domains)
        print(f"⚡ Validated: {len(validated_domains)} reachable domains")
        
        # Step 3: Enhanced scanning
        approved_results = []
        for i, domain in enumerate(validated_domains[:target_count], 1):
            result = await scanner.enhanced_scan_domain(domain)
            if result and result.status == 'approved':
                approved_results.append(result)
            
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(validated_domains[:target_count])} analyzed, {len(approved_results)} approved")
        
        # Final stats
        stats = scanner.get_stats()
        print(f"\n📈 PIPELINE COMPLETE")
        print(f"   Discovered: {len(discovered_domains)} domains")
        validation_pct = (len(validated_domains)/len(discovered_domains)*100) if discovered_domains else 0
        print(f"   Validated: {len(validated_domains)} domains ({validation_pct:.1f}%)")
        print(f"   New approvals: {len(approved_results)}")
        print(f"   Avg validation time: {stats['avg_validation_time']:.2f}s")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        
        return {
            'status': 'completed',
            'discovered': len(discovered_domains),
            'validated': len(validated_domains),
            'approved': len(approved_results),
            'validation_efficiency': len(validated_domains)/len(discovered_domains)*100 if discovered_domains else 0,
            'success_rate': stats['success_rate']
        }
        
    finally:
        await scanner.close_session()

async def main():
    """Test the enhanced system"""
    result = await run_enhanced_discovery_pipeline(target_count=20)
    print(f"\n🎉 Enhanced pipeline result: {result}")

if __name__ == "__main__":
    print("🚀 Enhanced Domain Discovery System")
    print("LLM-powered discovery + domain validation\n")
    
    asyncio.run(main())
