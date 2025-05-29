#!/usr/bin/env python3
"""
API-Enhanced Domain Discovery System
Leverages multiple domain intelligence APIs for superior discovery and analysis
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
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedDomainData:
    """Enhanced domain data from multiple APIs"""
    domain: str
    # Traffic & Authority
    monthly_visitors: Optional[int] = None
    domain_authority: Optional[int] = None
    alexa_rank: Optional[int] = None
    
    # Company Data (Clearbit/Similar)
    company_name: Optional[str] = None
    company_size: Optional[str] = None
    industry: Optional[str] = None
    revenue_range: Optional[str] = None
    
    # Technology Stack (BuiltWith)
    ad_networks: List[str] = None
    cms_platform: Optional[str] = None
    analytics_tools: List[str] = None
    
    # SEO Data (Ahrefs/SEMrush)
    backlink_count: Optional[int] = None
    referring_domains: Optional[int] = None
    organic_keywords: Optional[int] = None
    
    # Domain Intelligence
    domain_age: Optional[int] = None
    ssl_score: Optional[int] = None
    safety_score: Optional[int] = None
    
    # Our Analysis
    has_ads_txt: bool = False
    overall_score: float = 0.0

class APIEnhancedDiscovery:
    """Domain discovery enhanced with multiple intelligence APIs"""
    
    def __init__(self):
        self.session = None
        
        # API Keys (set via environment variables)
        self.clearbit_key = os.getenv("CLEARBIT_API_KEY")
        self.builtwith_key = os.getenv("BUILTWITH_API_KEY") 
        self.similarweb_key = os.getenv("SIMILARWEB_API_KEY")
        self.whoisxml_key = os.getenv("WHOISXML_API_KEY")
        self.securitytrails_key = os.getenv("SECURITYTRAILS_API_KEY")
        
        # API Endpoints
        self.api_endpoints = {
            'clearbit': 'https://company.clearbit.com/v2/companies/find',
            'builtwith': 'https://api.builtwith.com/v21/api.json',
            'similarweb': 'https://api.similarweb.com/v1/similar-rank',
            'whoisxml': 'https://www.whoisxmlapi.com/whoisserver/WhoisService',
            'securitytrails': 'https://api.securitytrails.com/v1/domain'
        }

    async def get_session(self):
        """Get aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close_session(self):
        """Close session"""
        if self.session:
            await self.session.close()

    async def get_clearbit_data(self, domain: str) -> Dict:
        """Get company data from Clearbit API"""
        if not self.clearbit_key:
            return {}
            
        session = await self.get_session()
        url = f"{self.api_endpoints['clearbit']}?domain={domain}"
        headers = {'Authorization': f'Bearer {self.clearbit_key}'}
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'company_name': data.get('name'),
                        'industry': data.get('category', {}).get('industry'),
                        'company_size': data.get('metrics', {}).get('employees'),
                        'revenue_range': data.get('metrics', {}).get('annualRevenue')
                    }
        except Exception as e:
            logger.debug(f"Clearbit API error for {domain}: {e}")
        
        return {}

    async def get_builtwith_data(self, domain: str) -> Dict:
        """Get technology stack from BuiltWith API"""
        if not self.builtwith_key:
            return {}
            
        session = await self.get_session()
        url = f"{self.api_endpoints['builtwith']}?KEY={self.builtwith_key}&LOOKUP={domain}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    ad_networks = []
                    analytics_tools = []
                    cms_platform = None
                    
                    # Parse BuiltWith response for relevant tech
                    for tech_group in data.get('Results', []):
                        for tech in tech_group.get('Result', {}).get('Paths', []):
                            for tech_item in tech.get('Technologies', []):
                                name = tech_item.get('Name', '').lower()
                                
                                # Categorize technologies
                                if any(ad_term in name for ad_term in ['google ads', 'facebook', 'doubleclick', 'adsense']):
                                    ad_networks.append(tech_item.get('Name'))
                                elif any(cms_term in name for cms_term in ['wordpress', 'drupal', 'joomla']):
                                    cms_platform = tech_item.get('Name')
                                elif any(analytics_term in name for analytics_term in ['analytics', 'tracking']):
                                    analytics_tools.append(tech_item.get('Name'))
                    
                    return {
                        'ad_networks': ad_networks,
                        'cms_platform': cms_platform,
                        'analytics_tools': analytics_tools
                    }
        except Exception as e:
            logger.debug(f"BuiltWith API error for {domain}: {e}")
        
        return {}

    async def get_similarweb_data(self, domain: str) -> Dict:
        """Get traffic data from SimilarWeb API"""
        if not self.similarweb_key:
            return {}
            
        session = await self.get_session()
        url = f"{self.api_endpoints['similarweb']}/{domain}/all"
        headers = {'Api-Key': self.similarweb_key}
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'monthly_visitors': data.get('visits', {}).get('2024-01'),  # Latest month
                        'domain_authority': data.get('global_rank', {}).get('rank'),
                        'bounce_rate': data.get('engagement', {}).get('bounce_rate')
                    }
        except Exception as e:
            logger.debug(f"SimilarWeb API error for {domain}: {e}")
        
        return {}

    async def get_whois_data(self, domain: str) -> Dict:
        """Get domain registration data from WhoisXML API"""
        if not self.whoisxml_key:
            return {}
            
        session = await self.get_session()
        url = f"{self.api_endpoints['whoisxml']}?apiKey={self.whoisxml_key}&domainName={domain}&outputFormat=JSON"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    whois_record = data.get('WhoisRecord', {})
                    
                    # Calculate domain age
                    created_date = whois_record.get('createdDate')
                    domain_age = None
                    if created_date:
                        try:
                            created = datetime.strptime(created_date[:10], '%Y-%m-%d')
                            domain_age = (datetime.now() - created).days // 365
                        except:
                            pass
                    
                    return {
                        'domain_age': domain_age,
                        'registrar': whois_record.get('registrarName'),
                        'creation_date': created_date
                    }
        except Exception as e:
            logger.debug(f"WhoisXML API error for {domain}: {e}")
        
        return {}

    async def get_security_data(self, domain: str) -> Dict:
        """Get security/DNS data from SecurityTrails API"""
        if not self.securitytrails_key:
            return {}
            
        session = await self.get_session()
        url = f"{self.api_endpoints['securitytrails']}/{domain}"
        headers = {'APIKEY': self.securitytrails_key}
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'ssl_score': 85 if data.get('current_dns', {}).get('a') else 0,  # Simplified
                        'dns_records': len(data.get('current_dns', {})),
                        'subdomains_count': data.get('subdomain_count', 0)
                    }
        except Exception as e:
            logger.debug(f"SecurityTrails API error for {domain}: {e}")
        
        return {}

    async def analyze_domain_with_apis(self, domain: str) -> EnhancedDomainData:
        """Comprehensive domain analysis using multiple APIs"""
        logger.info(f"🔍 API Analysis: {domain}")
        
        # Run all API calls concurrently
        tasks = [
            self.get_clearbit_data(domain),
            self.get_builtwith_data(domain),
            self.get_similarweb_data(domain),
            self.get_whois_data(domain),
            self.get_security_data(domain)
        ]
        
        try:
            clearbit_data, builtwith_data, similarweb_data, whois_data, security_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for i, result in enumerate([clearbit_data, builtwith_data, similarweb_data, whois_data, security_data]):
                if isinstance(result, Exception):
                    logger.warning(f"API {i} failed for {domain}: {result}")
                    
            # Combine all data
            enhanced_data = EnhancedDomainData(domain=domain)
            
            # Clearbit data
            if isinstance(clearbit_data, dict):
                enhanced_data.company_name = clearbit_data.get('company_name')
                enhanced_data.industry = clearbit_data.get('industry')
                enhanced_data.company_size = clearbit_data.get('company_size')
                enhanced_data.revenue_range = clearbit_data.get('revenue_range')
            
            # BuiltWith data
            if isinstance(builtwith_data, dict):
                enhanced_data.ad_networks = builtwith_data.get('ad_networks', [])
                enhanced_data.cms_platform = builtwith_data.get('cms_platform')
                enhanced_data.analytics_tools = builtwith_data.get('analytics_tools', [])
            
            # SimilarWeb data
            if isinstance(similarweb_data, dict):
                enhanced_data.monthly_visitors = similarweb_data.get('monthly_visitors')
                enhanced_data.domain_authority = similarweb_data.get('domain_authority')
            
            # Whois data
            if isinstance(whois_data, dict):
                enhanced_data.domain_age = whois_data.get('domain_age')
            
            # Security data
            if isinstance(security_data, dict):
                enhanced_data.ssl_score = security_data.get('ssl_score', 0)
            
            # Check ads.txt
            enhanced_data.has_ads_txt = await self.check_ads_txt(domain)
            
            # Calculate overall score based on all factors
            enhanced_data.overall_score = self.calculate_enhanced_score(enhanced_data)
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"API analysis failed for {domain}: {e}")
            return EnhancedDomainData(domain=domain)

    async def check_ads_txt(self, domain: str) -> bool:
        """Quick ads.txt check"""
        session = await self.get_session()
        try:
            async with session.get(f"https://{domain}/ads.txt") as response:
                return response.status == 200
        except:
            return False

    def calculate_enhanced_score(self, data: EnhancedDomainData) -> float:
        """Calculate comprehensive domain score using all API data"""
        score = 0.0
        
        # Traffic Score (0-30 points)
        if data.monthly_visitors:
            if data.monthly_visitors > 1000000:  # 1M+ monthly visitors
                score += 30
            elif data.monthly_visitors > 100000:  # 100K+ monthly visitors
                score += 20
            elif data.monthly_visitors > 10000:   # 10K+ monthly visitors
                score += 10
        
        # Authority Score (0-25 points)
        if data.domain_authority:
            score += min(25, data.domain_authority / 4)  # Scale domain authority
        
        # Ad Network Integration (0-20 points)
        if data.ad_networks:
            score += min(20, len(data.ad_networks) * 5)
        
        # Ads.txt Score (0-15 points)
        if data.has_ads_txt:
            score += 15
        
        # Domain Age & Trust (0-10 points)
        if data.domain_age:
            if data.domain_age > 5:
                score += 10
            elif data.domain_age > 2:
                score += 5
        
        return min(100, score)

    async def discover_domains_with_apis(self, target_count: int = 1000) -> List[EnhancedDomainData]:
        """Discover domains using API intelligence"""
        print(f"🚀 API-ENHANCED DOMAIN DISCOVERY")
        print("=" * 80)
        print(f"Analyzing {target_count:,} domains with multiple intelligence APIs\n")
        
        # Load existing domains to avoid duplicates
        existing = self.load_existing_domains()
        
        # Generate high-quality target domains
        target_domains = self.get_high_value_targets(existing, target_count)
        
        print(f"📊 API ANALYSIS STATUS:")
        print(f"   Target domains: {len(target_domains):,}")
        print(f"   APIs available: {self.count_available_apis()}")
        print(f"   Processing...\n")
        
        # Analyze domains with APIs
        all_results = []
        batch_size = 50  # Smaller batches for API rate limits
        
        for i in range(0, len(target_domains), batch_size):
            batch = target_domains[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(target_domains) + batch_size - 1) // batch_size
            
            print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} domains)")
            
            # Process batch with rate limiting
            tasks = [self.analyze_domain_with_apis(domain) for domain in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            for result in batch_results:
                if isinstance(result, EnhancedDomainData) and result.overall_score > 40:
                    all_results.append(result)
            
            print(f"   ✅ Found {len([r for r in batch_results if isinstance(r, EnhancedDomainData) and r.overall_score > 40])} quality domains")
            
            # Rate limiting
            await asyncio.sleep(2)
        
        # Sort by score
        all_results.sort(key=lambda x: x.overall_score, reverse=True)
        
        print(f"\n🎯 API DISCOVERY COMPLETE")
        print(f"   High-quality domains found: {len(all_results):,}")
        print(f"   Average score: {sum(r.overall_score for r in all_results) / len(all_results):.1f}" if all_results else "   No results")
        
        return all_results

    def count_available_apis(self) -> int:
        """Count available API keys"""
        apis = [self.clearbit_key, self.builtwith_key, self.similarweb_key, self.whoisxml_key, self.securitytrails_key]
        return sum(1 for api in apis if api)

    def load_existing_domains(self) -> Set[str]:
        """Load existing domains"""
        existing = set()
        try:
            with open('existing_domains.txt', 'r') as f:
                for line in f:
                    domain = line.strip().replace('www.', '')
                    if domain:
                        existing.add(domain)
        except FileNotFoundError:
            pass
        return existing

    def get_high_value_targets(self, existing: Set[str], count: int) -> List[str]:
        """Get high-value domain targets for API analysis"""
        targets = [
            # Top-tier business publications
            'wsj.com', 'ft.com', 'economist.com', 'bloomberg.com', 'reuters.com',
            'forbes.com', 'fortune.com', 'businessinsider.com', 'cnbc.com',
            
            # Major news outlets
            'cnn.com', 'bbc.com', 'nytimes.com', 'washingtonpost.com',
            'theguardian.com', 'usatoday.com', 'npr.org', 'apnews.com',
            
            # Technology publications
            'techcrunch.com', 'wired.com', 'arstechnica.com', 'theverge.com',
            'engadget.com', 'gizmodo.com', 'mashable.com', 'venturebeat.com',
            
            # Industry-specific
            'adweek.com', 'marketingland.com', 'digiday.com', 'mediapost.com',
            'industryweek.com', 'constructiondive.com', 'retaildive.com',
            
            # Professional/Educational
            'hbr.org', 'mckinsey.com', 'mit.edu', 'stanford.edu', 'harvard.edu'
        ]
        
        # Filter out existing domains
        new_targets = [d for d in targets if d not in existing]
        return new_targets[:count]

# Demo usage
async def demo_api_discovery():
    """Demo API-enhanced discovery"""
    discovery = APIEnhancedDiscovery()
    
    try:
        # Check API availability
        api_count = discovery.count_available_apis()
        print(f"📡 Available APIs: {api_count}/5")
        
        if api_count == 0:
            print("⚠️  No API keys configured. Set environment variables:")
            print("   CLEARBIT_API_KEY, BUILTWITH_API_KEY, SIMILARWEB_API_KEY")
            print("   WHOISXML_API_KEY, SECURITYTRAILS_API_KEY")
            print("\n🔄 Running basic analysis without APIs...\n")
        
        # Run discovery
        results = await discovery.discover_domains_with_apis(target_count=100)
        
        # Display results
        if results:
            print(f"\n🏆 TOP API-ENHANCED RESULTS")
            print("=" * 80)
            
            for i, result in enumerate(results[:10], 1):
                print(f"{i:2d}. {result.domain:<25} Score: {result.overall_score:.1f}")
                if result.monthly_visitors:
                    print(f"    Monthly Visitors: {result.monthly_visitors:,}")
                if result.company_name:
                    print(f"    Company: {result.company_name}")
                if result.ad_networks:
                    print(f"    Ad Networks: {', '.join(result.ad_networks[:3])}")
                print()
            
            # Export results
            df = pd.DataFrame([asdict(r) for r in results])
            filename = f"api_enhanced_results_{int(time.time())}.csv"
            df.to_csv(filename, index=False)
            print(f"📁 Results exported to: {filename}")
        
    finally:
        await discovery.close_session()

if __name__ == "__main__":
    print("🚀 API-Enhanced Domain Discovery")
    print("Leverage multiple intelligence APIs for superior domain analysis\n")
    
    asyncio.run(demo_api_discovery())
