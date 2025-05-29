#!/usr/bin/env python3
"""
High-ROI Domain Discovery System
Finds domains with actual display ad inventory using ads.txt analysis
Optimized for maximum advertising ROI
"""

import asyncio
import aiohttp
import pandas as pd
import sqlite3
import re
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Optional LLM imports
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdInventoryResult:
    """Results for domains with confirmed ad inventory"""
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
    # LLM enhancement fields
    llm_recommendation: Optional[str] = None
    llm_score: Optional[float] = None
    llm_analysis: Optional[str] = None
    final_score: Optional[float] = None

class HighROIDiscovery:
    """Optimized discovery system focused on high-value ad inventory"""
    
    def __init__(self, db_path: str = "high_roi_publishers.db"):
        self.db_path = db_path
        self.session = None
        self.setup_database()
        
        # Premium DSPs/SSPs that indicate quality inventory
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
        """Setup SQLite database for results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS high_roi_publishers (
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
                discovered_at TIMESTAMP,
                status TEXT DEFAULT 'approved'
            )
        ''')
        
        conn.commit()
        conn.close()

    async def get_session(self):
        """Get aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; AdInventoryBot/1.0)'}
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
        ads_txt_url = f"https://{domain}/ads.txt"
        
        try:
            async with session.get(ads_txt_url) as response:
                if response.status == 200:
                    content = await response.text()
                    return True, self.parse_ads_txt(content)
                return False, {}
        except Exception:
            return False, {}

    def parse_ads_txt(self, content: str) -> Dict:
        """Parse ads.txt content for inventory metrics"""
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

    async def analyze_page_content(self, domain: str) -> Tuple[int, float]:
        """Analyze page for ad slots and B2B relevance"""
        session = await self.get_session()
        url = f"https://{domain}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    ad_slots = self.count_ad_slots(content)
                    b2b_score = self.calculate_b2b_relevance(content)
                    return ad_slots, b2b_score
                return 0, 0.0
        except Exception:
            return 0, 0.0

    def count_ad_slots(self, content: str) -> int:
        """Count potential ad slots on page"""
        ad_patterns = [
            r'<div[^>]*class="[^"]*ad[^"]*"',
            r'<ins[^>]*class="[^"]*adsbygoogle',
            r'<iframe[^>]*googlesyndication',
            r'<div[^>]*data-ad-slot',
            r'<div[^>]*class="[^"]*banner[^"]*"'
        ]
        
        total = 0
        for pattern in ad_patterns:
            total += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Fallback estimation
        if total == 0:
            div_count = len(re.findall(r'<div', content, re.IGNORECASE))
            total = min(8, div_count // 12)  # Conservative estimate
        
        return total

    def calculate_b2b_relevance(self, content: str) -> float:
        """Calculate B2B relevance score"""
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text().lower()
        
        keyword_count = 0
        for keyword in self.b2b_keywords:
            keyword_count += text.count(keyword)
        
        # Normalize score (0-100)
        text_length = len(text.split())
        if text_length > 0:
            density = (keyword_count / text_length) * 1000
            return min(100, density * 20)
        return 0.0

    def calculate_inventory_score(self, ads_analysis: Dict, ad_slots: int) -> float:
        """Calculate ad inventory quality score"""
        score = 0.0
        
        # Base score for having ads.txt
        score += 25
        
        # Premium platforms (0-35 points)
        premium_count = len(set(ads_analysis.get('premium_platforms', [])))
        score += min(35, premium_count * 3)
        
        # Direct deals (0-25 points)
        direct_deals = ads_analysis.get('direct_deals', 0)
        score += min(25, direct_deals * 1.5)
        
        # Ad slots (0-15 points)
        score += min(15, ad_slots * 1.5)
        
        return min(100, score)

    async def analyze_domain(self, domain: str) -> Optional[AdInventoryResult]:
        """Comprehensive domain analysis focused on ad inventory"""
        logger.info(f"Analyzing: {domain}")
        
        # Step 1: Check ads.txt (filter out domains without it)
        has_ads_txt, ads_analysis = await self.check_ads_txt(domain)
        if not has_ads_txt:
            return None  # Skip domains without ads.txt for efficiency
        
        # Step 2: Analyze page content
        ad_slots, b2b_relevance = await self.analyze_page_content(domain)
        
        # Step 3: Calculate scores
        inventory_score = self.calculate_inventory_score(ads_analysis, ad_slots)
        overall_score = (inventory_score * 0.7) + (b2b_relevance * 0.3)
        
        return AdInventoryResult(
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

    async def get_page_content_for_llm(self, domain: str) -> str:
        """Get clean page content for LLM analysis"""
        session = await self.get_session()
        url = f"https://{domain}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer"]):
                        script.decompose()
                    
                    # Get clean text
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    # Limit to first 3000 characters for LLM analysis
                    return text[:3000]
                return ""
        except Exception:
            return ""

    async def llm_analyze_with_openai(self, domain: str, content: str, metrics: Dict) -> Dict:
        """Analyze domain with OpenAI for final recommendation"""
        if not HAS_OPENAI:
            return {}
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {}
            
        try:
            client = openai.OpenAI(api_key=api_key)
            
            prompt = f"""Analyze this website for B2B advertising suitability:

Domain: {domain}
Ad Inventory: {metrics['ad_slots']} slots, {metrics['direct_deals']} direct deals
Premium DSPs: {len(metrics['premium_dsps'])}
Content Preview: {content[:1500]}

As an advertising expert, provide:
1. RECOMMENDATION: APPROVE/REJECT/CAUTION
2. SCORE: 0-100 (advertising value)
3. REASONING: 2-3 sentences explaining your decision

Focus on: professional content quality, brand safety, B2B audience relevance, and advertising inventory quality."""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content or ""
            
            # Parse response
            recommendation = "CAUTION"
            score = 50.0
            reasoning = analysis
            
            if "APPROVE" in analysis.upper():
                recommendation = "APPROVE"
            elif "REJECT" in analysis.upper():
                recommendation = "REJECT"
                
            # Extract score if present
            import re
            score_match = re.search(r'SCORE[:\s]*(\d+)', analysis.upper())
            if score_match:
                score = float(score_match.group(1))
                
            return {
                'recommendation': recommendation,
                'score': score,
                'analysis': reasoning
            }
            
        except Exception as e:
            logger.warning(f"OpenAI analysis failed for {domain}: {e}")
            return {}

    async def llm_analyze_with_anthropic(self, domain: str, content: str, metrics: Dict) -> Dict:
        """Analyze domain with Anthropic for final recommendation"""
        if not HAS_ANTHROPIC:
            return {}
            
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {}
            
        try:
            client = anthropic.Anthropic(api_key=api_key)
            
            prompt = f"""Analyze this website for B2B advertising suitability:

Domain: {domain}
Ad Inventory: {metrics['ad_slots']} slots, {metrics['direct_deals']} direct deals
Premium DSPs: {len(metrics['premium_dsps'])}
Content Preview: {content[:1500]}

As an advertising expert, provide:
1. RECOMMENDATION: APPROVE/REJECT/CAUTION
2. SCORE: 0-100 (advertising value)
3. REASONING: 2-3 sentences explaining your decision

Focus on: professional content quality, brand safety, B2B audience relevance, and advertising inventory quality."""
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis = getattr(response.content[0], 'text', '') or ""
            
            # Parse response
            recommendation = "CAUTION"
            score = 50.0
            reasoning = analysis
            
            if "APPROVE" in analysis.upper():
                recommendation = "APPROVE"
            elif "REJECT" in analysis.upper():
                recommendation = "REJECT"
                
            # Extract score if present
            import re
            score_match = re.search(r'SCORE[:\s]*(\d+)', analysis.upper())
            if score_match:
                score = float(score_match.group(1))
                
            return {
                'recommendation': recommendation,
                'score': score,
                'analysis': reasoning
            }
            
        except Exception as e:
            logger.warning(f"Anthropic analysis failed for {domain}: {e}")
            return {}

    async def enhance_with_llm(self, results: List[AdInventoryResult]) -> List[AdInventoryResult]:
        """Enhance high-scoring results with LLM analysis for final recommendation"""
        if not results:
            return results
            
        # Only analyze domains with good base scores (efficiency)
        candidates = [r for r in results if r.overall_score >= 60]
        
        if not candidates:
            return results
            
        print(f"\n🤖 LLM ENHANCEMENT PHASE")
        print("-" * 40)
        print(f"Running AI analysis on {len(candidates)} high-scoring domains...")
        
        api_available = (HAS_OPENAI and os.getenv("OPENAI_API_KEY")) or (HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"))
        
        if not api_available:
            print("⚠️  No LLM API keys found - skipping AI enhancement")
            print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY for AI analysis")
            return results
            
        enhanced_count = 0
        
        for result in candidates:
            # Get content for LLM analysis
            content = await self.get_page_content_for_llm(result.domain)
            if not content:
                continue
                
            # Prepare metrics for LLM
            metrics = {
                'ad_slots': result.estimated_ad_slots,
                'direct_deals': result.direct_deals,
                'premium_dsps': result.premium_dsps
            }
            
            # Try OpenAI first, then Anthropic
            llm_result = {}
            if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                llm_result = await self.llm_analyze_with_openai(result.domain, content, metrics)
            elif HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
                llm_result = await self.llm_analyze_with_anthropic(result.domain, content, metrics)
                
            if llm_result:
                result.llm_recommendation = llm_result.get('recommendation', 'CAUTION')
                result.llm_score = llm_result.get('score', 50.0)
                result.llm_analysis = llm_result.get('analysis', 'No analysis available')
                
                # Calculate final score (weighted combination)
                result.final_score = (result.overall_score * 0.6) + ((result.llm_score or 50.0) * 0.4)
                enhanced_count += 1
                
                logger.info(f"LLM enhanced {result.domain}: {result.llm_recommendation} (Score: {result.llm_score:.1f})")
        
        print(f"✅ Enhanced {enhanced_count} domains with LLM analysis")
        
        # Re-sort by final score if available, otherwise overall score
        results.sort(key=lambda x: x.final_score if x.final_score else x.overall_score, reverse=True)
        
        return results

    def get_discovery_targets(self) -> List[str]:
        """Get list of domains to analyze"""
        # High-traffic domains known to have good ad inventory
        return [
            # Tech/Business News
            'techcrunch.com', 'venturebeat.com', 'theverge.com', 'engadget.com',
            'mashable.com', 'cnet.com', 'zdnet.com', 'computerworld.com',
            'infoworld.com', 'networkworld.com', 'pcworld.com', 'macworld.com',
            
            # Financial/Business
            'moneycontrol.com', 'livemint.com', 'economictimes.com',
            'business-standard.com', 'financialexpress.com', 'yourstory.com',
            'entrepreneur.com', 'inc.com', 'fastcompany.com',
            
            # Technology Focus
            'digitaltrends.com', 'tomsguide.com', 'tomshardware.com', 'anandtech.com'
        ]

    async def run_discovery(self, max_domains: int = 50) -> List[AdInventoryResult]:
        """Run complete high-ROI discovery process"""
        print("🎯 HIGH-ROI AD INVENTORY DISCOVERY")
        print("=" * 60)
        print("Finding domains with confirmed display ad opportunities...\n")
        
        # Get targets and analyze
        domains = self.get_discovery_targets()[:max_domains]
        print(f"🔍 Analyzing {len(domains)} high-traffic domains...")
        print("⏳ Filtering for ads.txt files and analyzing inventory...\n")
        
        results = []
        domains_with_ads = 0
        
        for domain in domains:
            result = await self.analyze_domain(domain)
            if result:
                domains_with_ads += 1
                results.append(result)
                self.save_result(result)
        
        # Sort by overall score
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        print(f"✅ Found {domains_with_ads} domains with ads.txt files")
        return results

    def save_result(self, result: AdInventoryResult):
        """Save result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO high_roi_publishers 
            (domain, has_ads_txt, ads_txt_entries, premium_dsps, direct_deals,
             reseller_deals, estimated_ad_slots, inventory_score, b2b_relevance,
             overall_score, discovered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.domain, result.has_ads_txt, result.ads_txt_entries,
            json.dumps(result.premium_dsps), result.direct_deals,
            result.reseller_deals, result.estimated_ad_slots,
            result.inventory_score, result.b2b_relevance,
            result.overall_score, datetime.now()
        ))
        
        conn.commit()
        conn.close()

    def export_results(self, results: List[AdInventoryResult], filename: str = "high_roi_publishers.csv"):
        """Export results to CSV"""
        data = []
        for result in results:
            data.append({
                'domain': result.domain,
                'overall_score': result.overall_score,
                'inventory_score': result.inventory_score,
                'b2b_relevance': result.b2b_relevance,
                'premium_dsps': len(result.premium_dsps),
                'direct_deals': result.direct_deals,
                'ad_slots': result.estimated_ad_slots,
                'ads_txt_entries': result.ads_txt_entries
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename

async def main():
    """Main execution function"""
    discovery = HighROIDiscovery()
    
    try:
        # Run discovery
        results = await discovery.run_discovery()
        
        # Enhance with LLM analysis for final recommendations
        results = await discovery.enhance_with_llm(results)
        
        # Categorize results
        premium = [r for r in results if r.overall_score >= 80]
        high_value = [r for r in results if 60 <= r.overall_score < 80]
        medium_value = [r for r in results if 40 <= r.overall_score < 60]
        
        # Display results
        print("🏆 DISCOVERY RESULTS")
        print("=" * 80)
        
        print(f"\n🥇 PREMIUM INVENTORY ({len(premium)} domains):")
        for result in premium:
            score_display = f"Final: {result.final_score:.1f}" if result.final_score else f"Score: {result.overall_score:.1f}"
            print(f"   {result.domain:<25} {score_display}")
            print(f"      Premium DSPs: {len(result.premium_dsps):3d} | "
                  f"Direct Deals: {result.direct_deals:3d} | "
                  f"Ad Slots: {result.estimated_ad_slots:3d}")
            if result.llm_recommendation:
                print(f"      🤖 AI: {result.llm_recommendation} | LLM Score: {result.llm_score:.1f}")
                print(f"      💭 {result.llm_analysis[:100]}..." if result.llm_analysis and len(result.llm_analysis) > 100 else f"      💭 {result.llm_analysis}")
        
        print(f"\n🥈 HIGH VALUE ({len(high_value)} domains):")
        for result in high_value:
            print(f"   {result.domain:<25} Score: {result.overall_score:.1f}")
        
        print(f"\n🥉 MEDIUM VALUE ({len(medium_value)} domains):")
        for result in medium_value:
            print(f"   {result.domain:<25} Score: {result.overall_score:.1f}")
        
        # Export results
        if results:
            filename = discovery.export_results(results)
            print(f"\n📁 Results exported to: {filename}")
        
        # Summary metrics
        total_slots = sum(r.estimated_ad_slots for r in premium + high_value)
        total_premium_dsps = sum(len(r.premium_dsps) for r in premium + high_value)
        total_direct_deals = sum(r.direct_deals for r in premium + high_value)
        
        print(f"\n💰 ROI SUMMARY:")
        print(f"   High-value domains found: {len(premium + high_value)}")
        print(f"   Total ad slots available: {total_slots}")
        print(f"   Premium DSP connections: {total_premium_dsps}")
        print(f"   Direct deal opportunities: {total_direct_deals}")
        
        if len(results) > 0:
            avg_score = sum(r.overall_score for r in results) / len(results)
            print(f"   Average quality score: {avg_score:.1f}/100")
        
    finally:
        await discovery.close_session()

if __name__ == "__main__":
    print("🚀 High-ROI Domain Discovery System")
    print("Targeting domains with confirmed display ad inventory\n")
    
    asyncio.run(main())
