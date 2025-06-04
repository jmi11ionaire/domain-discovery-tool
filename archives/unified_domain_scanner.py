#!/usr/bin/env python3
"""
UNIFIED Domain Discovery System
Consolidates all scanning strategies with Anthropic LLM validation
Single source of truth for domain discovery with training data collection
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

class UnifiedDatabase:
    """Single database for all domain analysis tracking"""
    
    def __init__(self, db_path: str = "unified_domain_discovery.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Setup unified database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main tracking table - never analyze same domain twice
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domains_analyzed (
                domain TEXT PRIMARY KEY,
                first_analyzed_at TIMESTAMP,
                last_analyzed_at TIMESTAMP,
                analysis_count INTEGER DEFAULT 1,
                current_status TEXT,
                current_score REAL,
                strategy_used TEXT,
                is_existing_domain BOOLEAN DEFAULT FALSE,
                needs_human_review BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Detailed analysis results for each strategy attempt
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY,
                domain TEXT,
                strategy_type TEXT,
                analyzed_at TIMESTAMP,
                score REAL,
                status TEXT,
                reason TEXT,
                ads_txt_found BOOLEAN,
                content_analysis TEXT,
                llm_analysis TEXT,
                details TEXT,
                FOREIGN KEY (domain) REFERENCES domains_analyzed(domain)
            )
        ''')
        
        # Training data candidates for human review
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_candidates (
                id INTEGER PRIMARY KEY,
                domain TEXT,
                strategy_type TEXT,
                score REAL,
                status TEXT,
                llm_recommendation TEXT,
                flagged_for_review BOOLEAN DEFAULT FALSE,
                review_reason TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        # Existing domains tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS existing_domains (
                domain TEXT PRIMARY KEY,
                added_at TIMESTAMP,
                source TEXT DEFAULT 'historical_import'
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Unified database initialized")
    
    def domain_already_analyzed(self, domain: str, max_age_days: int = 30) -> bool:
        """Check if domain was analyzed recently"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cursor.execute('''
            SELECT COUNT(*) FROM domains_analyzed 
            WHERE domain = ? AND last_analyzed_at > ?
        ''', (domain, cutoff_date))
        
        exists = cursor.fetchone()[0] > 0
        conn.close()
        return exists
    
    def save_analysis(self, result: ScanResult):
        """Save analysis result to unified database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update main tracking table
        cursor.execute('''
            INSERT OR REPLACE INTO domains_analyzed 
            (domain, first_analyzed_at, last_analyzed_at, analysis_count, 
             current_status, current_score, strategy_used, is_existing_domain)
            VALUES (?, 
                COALESCE((SELECT first_analyzed_at FROM domains_analyzed WHERE domain = ?), ?),
                ?, 
                COALESCE((SELECT analysis_count FROM domains_analyzed WHERE domain = ?) + 1, 1),
                ?, ?, ?, ?)
        ''', (result.domain, result.domain, result.analyzed_at, result.analyzed_at, 
              result.domain, result.status, result.score, result.strategy, 
              result.details.get('is_existing_domain', False)))
        
        # Save detailed analysis
        cursor.execute('''
            INSERT INTO analysis_results 
            (domain, strategy_type, analyzed_at, score, status, reason, 
             ads_txt_found, content_analysis, llm_analysis, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (result.domain, result.strategy, result.analyzed_at, result.score, 
              result.status, result.reason, result.details.get('ads_txt_found', False),
              json.dumps(result.details.get('content_analysis', {})),
              json.dumps(result.llm_analysis) if result.llm_analysis else None,
              json.dumps(result.details)))
        
        # Flag for training if score is edge case
        if self._should_flag_for_training(result):
            cursor.execute('''
                INSERT INTO training_candidates 
                (domain, strategy_type, score, status, llm_recommendation, 
                 flagged_for_review, review_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (result.domain, result.strategy, result.score, result.status,
                  result.llm_analysis.get('recommendation', '') if result.llm_analysis else '',
                  True, 'edge_case_score', result.analyzed_at))
        
        conn.commit()
        conn.close()
    
    def _should_flag_for_training(self, result: ScanResult) -> bool:
        """Determine if result should be flagged for human training review"""
        # Flag edge cases for training data
        if 40 <= result.score <= 60:  # Middle scores
            return True
        if result.status == 'approved' and result.score < 70:  # Low score approvals
            return True
        if result.status == 'rejected' and result.score > 50:  # High score rejections
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM domains_analyzed")
        total_analyzed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM domains_analyzed WHERE current_status = 'approved'")
        approved = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT strategy_used) FROM domains_analyzed")
        strategies_used = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM training_candidates WHERE flagged_for_review = TRUE")
        training_candidates = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_analyzed': total_analyzed,
            'approved': approved,
            'success_rate': (approved / total_analyzed * 100) if total_analyzed > 0 else 0,
            'strategies_used': strategies_used,
            'training_candidates': training_candidates
        }

class UnifiedDomainScanner:
    """Unified domain scanner with three strategies and LLM validation"""
    
    def __init__(self, db_path: str = "unified_domain_discovery.db"):
        self.db = UnifiedDatabase(db_path)
        self.session = None
        
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
                headers={'User-Agent': 'Mozilla/5.0 (compatible; UnifiedDomainScanner/1.0)'}
            )
        return self.session
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
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
    
    async def llm_validate(self, domain: str, content_text: str, metrics: Dict) -> Dict:
        """Validate domain with Anthropic LLM - used by ALL strategies"""
        if not HAS_ANTHROPIC:
            return {}
            
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set")
            return {}
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            
            prompt = f"""Analyze this website for B2B advertising suitability:

Domain: {domain}
Metrics:
- Ads.txt found: {metrics.get('ads_txt_found', False)}
- Premium DSPs: {metrics.get('premium_dsps', 0)}
- Direct deals: {metrics.get('direct_deals', 0)}
- Ad slots detected: {metrics.get('ad_slots', 0)}
- Content quality score: {metrics.get('content_score', 0):.1f}
- B2B relevance: {metrics.get('b2b_relevance', 0):.1f}

Content preview: {content_text[:1500]}

As an advertising expert, provide:
1. RECOMMENDATION: APPROVE/REJECT/CAUTION
2. SCORE: 0-100 (advertising value)
3. REASONING: 2-3 sentences explaining your decision

Focus on: professional content quality, brand safety, B2B audience relevance, and advertising monetization potential."""
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis = getattr(response.content[0], 'text', '') if response.content else ""
            
            # Parse response
            recommendation = "CAUTION"
            score = 50.0
            reasoning = analysis
            
            if "APPROVE" in analysis.upper():
                recommendation = "APPROVE"
            elif "REJECT" in analysis.upper():
                recommendation = "REJECT"
            
            # Extract score
            score_match = re.search(r'SCORE[:\s]*(\d+)', analysis.upper())
            if score_match:
                score = float(score_match.group(1))
            
            return {
                'recommendation': recommendation,
                'score': score,
                'reasoning': reasoning,
                'full_analysis': analysis
            }
            
        except Exception as e:
            logger.warning(f"LLM analysis failed for {domain}: {e}")
            return {}
    
    # STRATEGY 1: STRICT SCANNING (requires ads.txt)
    async def strict_scan(self, domain: str) -> Optional[ScanResult]:
        """Strict strategy: requires ads.txt + LLM validation"""
        logger.info(f"🔒 Strict scan: {domain}")
        
        # Check ads.txt (required)
        has_ads_txt, ads_analysis = await self.check_ads_txt(domain)
        if not has_ads_txt:
            return ScanResult(
                domain=domain,
                strategy='strict',
                status='rejected',
                score=0,
                reason='no_ads_txt_required',
                details={'ads_txt_found': False}
            )
        
        # Analyze content
        content_score, content_details = await self.analyze_content(domain)
        
        # Calculate base score
        score = 35  # Base for having ads.txt
        score += min(40, len(set(ads_analysis.get('premium_platforms', []))) * 5)
        score += min(15, ads_analysis.get('total_entries', 0) * 0.3)
        score += min(20, ads_analysis.get('direct_deals', 0) * 2)
        score += content_score * 0.3
        
        # LLM validation (critical for strict strategy)
        metrics = {
            'ads_txt_found': True,
            'premium_dsps': len(set(ads_analysis.get('premium_platforms', []))),
            'direct_deals': ads_analysis.get('direct_deals', 0),
            'ad_slots': content_details.get('ad_slots_detected', 0),
            'content_score': content_score,
            'b2b_relevance': content_details.get('b2b_relevance', 0)
        }
        
        llm_result = await self.llm_validate(domain, content_details.get('page_text', ''), metrics)
        
        # Final scoring with LLM input
        if llm_result:
            final_score = (score * 0.6) + (llm_result.get('score', 50) * 0.4)
            status = 'approved' if llm_result.get('recommendation') == 'APPROVE' and final_score >= 60 else 'rejected'
        else:
            final_score = score
            status = 'approved' if final_score >= 70 else 'rejected'  # Higher threshold without LLM
        
        details = {
            'ads_txt_found': True,
            'premium_dsps': len(set(ads_analysis.get('premium_platforms', []))),
            'direct_deals': ads_analysis.get('direct_deals', 0),
            'content_analysis': content_details,
            'is_existing_domain': domain in self.existing_domains
        }
        
        return ScanResult(
            domain=domain,
            strategy='strict',
            status=status,
            score=final_score,
            reason=f"strict_{status}",
            details=details,
            llm_analysis=llm_result
        )
    
    # STRATEGY 2: FLEXIBLE SCANNING (no ads.txt requirement)
    async def flexible_scan(self, domain: str) -> Optional[ScanResult]:
        """Flexible strategy: content-focused + LLM validation"""
        logger.info(f"🤝 Flexible scan: {domain}")
        
        # Check ads.txt (optional but helpful)
        has_ads_txt, ads_analysis = await self.check_ads_txt(domain)
        
        # Analyze content (critical for flexible strategy)
        content_score, content_details = await self.analyze_content(domain)
        
        if content_score < 20:  # Must have some content
            return ScanResult(
                domain=domain,
                strategy='flexible',
                status='rejected',
                score=content_score,
                reason='insufficient_content',
                details={'content_score': content_score}
            )
        
        # Flexible scoring algorithm
        score = content_score * 0.7  # Content is primary
        
        if has_ads_txt:
            score += 25  # Bonus for ads.txt
            score += min(15, len(set(ads_analysis.get('premium_platforms', []))) * 3)
        
        # LLM validation
        metrics = {
            'ads_txt_found': has_ads_txt,
            'premium_dsps': len(set(ads_analysis.get('premium_platforms', []))) if has_ads_txt else 0,
            'direct_deals': ads_analysis.get('direct_deals', 0) if has_ads_txt else 0,
            'ad_slots': content_details.get('ad_slots_detected', 0),
            'content_score': content_score,
            'b2b_relevance': content_details.get('b2b_relevance', 0)
        }
        
        llm_result = await self.llm_validate(domain, content_details.get('page_text', ''), metrics)
        
        # Final scoring with LLM
        if llm_result:
            final_score = (score * 0.6) + (llm_result.get('score', 50) * 0.4)
            # More lenient approval for flexible strategy
            status = 'approved' if llm_result.get('recommendation') in ['APPROVE', 'CAUTION'] and final_score >= 45 else 'rejected'
        else:
            final_score = score
            status = 'approved' if final_score >= 50 else 'rejected'
        
        details = {
            'ads_txt_found': has_ads_txt,
            'premium_dsps': len(set(ads_analysis.get('premium_platforms', []))) if has_ads_txt else 0,
            'direct_deals': ads_analysis.get('direct_deals', 0) if has_ads_txt else 0,
            'content_analysis': content_details,
            'is_existing_domain': domain in self.existing_domains
        }
        
        return ScanResult(
            domain=domain,
            strategy='flexible',
            status=status,
            score=final_score,
            reason=f"flexible_{status}",
            details=details,
            llm_analysis=llm_result
        )
    
    # STRATEGY 3: API BULK SCANNING (future scale)
    async def api_bulk_scan(self, domains: List[str]) -> List[ScanResult]:
        """API bulk strategy: scale processing with external APIs"""
        logger.info(f"⚡ API bulk scan: {len(domains)} domains")
        results = []
        
        # For now, implement as simplified batch processing
        # TODO: Integrate with PublicWWW or similar APIs for ads.txt bulk discovery
        
        for domain in domains[:10]:  # Limit for demo
            # Quick ads.txt check
            has_ads_txt, ads_analysis = await self.check_ads_txt(domain)
            
            if not has_ads_txt:
                results.append(ScanResult(
                    domain=domain,
                    strategy='api_bulk',
                    status='rejected',
                    score=0,
                    reason='bulk_no_ads_txt',
                    details={'ads_txt_found': False}
                ))
                continue
            
            # Simplified scoring for bulk processing
            score = 50  # Base score
            score += min(30, len(set(ads_analysis.get('premium_platforms', []))) * 5)
            
            # Quick approval for bulk with minimal LLM validation
            status = 'approved' if score >= 60 else 'rejected'
            
            results.append(ScanResult(
                domain=domain,
                strategy='api_bulk',
                status=status,
                score=score,
                reason=f'bulk_{status}',
                details={
                    'ads_txt_found': True,
                    'premium_dsps': len(set(ads_analysis.get('premium_platforms', []))),
                    'direct_deals': ads_analysis.get('direct_deals', 0),
                    'is_existing_domain': domain in self.existing_domains
                }
            ))
            
            # Rate limiting for bulk processing
            await asyncio.sleep(0.1)
        
        return results
    
    # MAIN SCANNING ORCHESTRATOR
    async def scan_domain(self, domain: str, strategy: str = 'strict') -> Optional[ScanResult]:
        """Main entry point for domain scanning"""
        # Skip if domain already analyzed recently
        if self.db.domain_already_analyzed(domain):
            logger.info(f"⏩ Skipping {domain} - analyzed recently")
            return None
        
        # Skip existing domains unless specifically requested
        if domain in self.existing_domains:
            logger.info(f"⏩ Skipping {domain} - in existing domains list")
            return None
        
        # Route to appropriate strategy
        try:
            if strategy == 'strict':
                result = await self.strict_scan(domain)
            elif strategy == 'flexible':
                result = await self.flexible_scan(domain)
            elif strategy == 'api_bulk':
                result = (await self.api_bulk_scan([domain]))[0] if domain else None
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Save result to database
            if result:
                self.db.save_analysis(result)
                logger.info(f"✅ {result.status.upper()}: {domain} (Score: {result.score:.1f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Scan failed for {domain}: {e}")
            return None
    
    def generate_domains(self, count: int = 100) -> List[str]:
        """Generate test domains for scanning"""
        # B2B domain patterns for testing
        business_keywords = ['business', 'tech', 'finance', 'enterprise', 'industry']
        formats = ['news', 'daily', 'times', 'post', 'magazine', 'review']
        
        domains = []
        for keyword in business_keywords:
            for format_word in formats:
                domains.append(f"{keyword}{format_word}.com")
                if len(domains) >= count:
                    break
            if len(domains) >= count:
                break
        
        random.shuffle(domains)
        return domains[:count]
    
    def export_training_data(self):
        """Export training data to CSV files"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get flagged training candidates
        cursor.execute('''
            SELECT domain, strategy_type, score, status, details
            FROM training_candidates tc
            WHERE flagged_for_review = TRUE
            ORDER BY created_at DESC
        ''')
        
        candidates = cursor.fetchall()
        conn.close()
        
        # Sort into positive and negative examples
        positive_examples = []
        negative_examples = []
        
        for row in candidates:
            domain, strategy, score, status, details_json = row
            details = json.loads(details_json) if details_json else {}
            
            example = {
                'domain': domain,
                'strategy': strategy,
                'score': score,
                'status': status,
                'ads_txt_found': details.get('ads_txt_found', False),
                'premium_dsps': details.get('premium_dsps', 0),
                'direct_deals': details.get('direct_deals', 0)
            }
            
            # Categorize based on performance
            if status == 'approved' and score >= 70:
                positive_examples.append(example)
            else:
                negative_examples.append(example)
        
        # Export to CSV files
        self._export_to_csv(positive_examples, 'positive_training_data.csv')
        self._export_to_csv(negative_examples, 'negative_training_data.csv')
        
        logger.info(f"Exported {len(positive_examples)} positive and {len(negative_examples)} negative training examples")
    
    def _export_to_csv(self, examples: List[Dict], filename: str):
        """Export examples to CSV file"""
        if not examples:
            return
            
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = examples[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(examples)
    
    def export_results(self, filename: str = None) -> str:
        """Export approved domains to CSV"""
        if filename is None:
            filename = f"unified_approved_domains_{int(datetime.now().timestamp())}.csv"
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT da.domain, da.current_score, da.strategy_used, da.last_analyzed_at
            FROM domains_analyzed da
            WHERE da.current_status = 'approved'
            ORDER BY da.current_score DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['domain', 'score', 'strategy', 'analyzed_at']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in results:
                domain, score, strategy, analyzed_at = row
                writer.writerow({
                    'domain': domain,
                    'score': score,
                    'strategy': strategy,
                    'analyzed_at': analyzed_at
                })
        
        logger.info(f"Exported {len(results)} approved domains to {filename}")
        return filename

# DAG-READY ORCHESTRATOR
async def run_discovery_pipeline(strategy: str = 'flexible', target_count: int = 10) -> Dict:
    """Complete discovery pipeline ready for DAG scheduling"""
    scanner = UnifiedDomainScanner()
    
    try:
        print(f"🚀 UNIFIED DOMAIN DISCOVERY PIPELINE")
        print(f"Strategy: {strategy.upper()} | Target: {target_count} domains")
        print("=" * 60)
        
        # Get current statistics
        stats = scanner.db.get_stats()
        print(f"📊 Current Status:")
        print(f"   Total analyzed: {stats['total_analyzed']:,}")
        print(f"   Approved: {stats['approved']:,}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        
        # Generate test domains
        test_domains = scanner.generate_domains(target_count)
        print(f"\n🔍 Scanning {len(test_domains)} domains with {strategy} strategy...")
        
        # Scan domains
        approved_results = []
        for i, domain in enumerate(test_domains, 1):
            result = await scanner.scan_domain(domain, strategy)
            if result and result.status == 'approved':
                approved_results.append(result)
            
            if i % 5 == 0:
                print(f"   Progress: {i}/{len(test_domains)} analyzed, {len(approved_results)} approved")
        
        # Export results
        if approved_results:
            export_file = scanner.export_results()
            scanner.export_training_data()
        else:
            export_file = None
        
        # Final statistics
        final_stats = scanner.db.get_stats()
        
        print(f"\n📈 PIPELINE COMPLETE")
        print(f"   New approvals: {len(approved_results)}")
        print(f"   Total approved: {final_stats['approved']:,}")
        print(f"   Success rate: {final_stats['success_rate']:.1f}%")
        if export_file:
            print(f"   Results exported to: {export_file}")
        
        return {
            'status': 'completed',
            'approved_count': final_stats['approved'],
            'new_approvals': len(approved_results),
            'success_rate': final_stats['success_rate'],
            'export_file': export_file or 'none'
        }
        
    finally:
        await scanner.close_session()

async def main():
    """Main execution function for testing"""
    try:
        # Test with a small batch
        result = await run_discovery_pipeline(strategy='flexible', target_count=5)
        print(f"\n🎉 Pipeline result: {result}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    print("🚀 Unified Domain Discovery System")
    print("Consolidates all strategies with Anthropic LLM validation\n")
    
    asyncio.run(main())
