#!/usr/bin/env python3
"""
API-Enhanced Bulk Domain Scanner
High-scale domain discovery and analysis using external APIs
"""

import asyncio
import aiohttp
import sqlite3
import json
import logging
import os
import yaml
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import random

# Import core analysis functions from existing scanner
from fixed_enhanced_domain_scanner import (
    ScanResult, FixedEnhancedDomainScanner
)
from robust_domain_validator import RobustDomainValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BulkBatch:
    """Track bulk processing batches"""
    batch_id: str
    source_api: str
    total_domains: int
    processed_domains: int
    approved_domains: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = 'running'  # running, completed, failed, paused

class APIConfig:
    """API configuration management"""
    
    def __init__(self, config_file: str = "api_config.yaml"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load API configuration from YAML file"""
        default_config = {
            'common_crawl': {
                'enabled': False,
                'api_key': '',
                'base_url': 'http://index.commoncrawl.org',
                'rate_limit': 10  # requests per second
            },
            'semrush': {
                'enabled': False,
                'api_key': '',
                'base_url': 'https://api.semrush.com',
                'rate_limit': 1
            },
            'ahrefs': {
                'enabled': False,
                'api_key': '',
                'base_url': 'https://apiv2.ahrefs.com',
                'rate_limit': 1
            },
            'security_trails': {
                'enabled': False,
                'api_key': '',
                'base_url': 'https://api.securitytrails.com/v1',
                'rate_limit': 5
            },
            'bulk_processing': {
                'batch_size': 100,
                'max_concurrent': 50,
                'retry_attempts': 3,
                'timeout_seconds': 30
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Merge with defaults
                    for key, value in loaded_config.items():
                        if key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            else:
                # Create default config file
                self.save_config(default_config)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def save_config(self, config: Dict):
        """Save configuration to YAML file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_enabled_apis(self) -> List[str]:
        """Get list of enabled APIs"""
        return [api_name for api_name, config in self.config.items() 
                if isinstance(config, dict) and config.get('enabled', False)]

class BulkAPIManager:
    """Manage multiple APIs for bulk domain discovery"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = None
        self.rate_limiters = {}
        
        # Initialize rate limiters for each API
        for api_name, api_config in config.config.items():
            if isinstance(api_config, dict) and 'rate_limit' in api_config:
                self.rate_limiters[api_name] = AsyncRateLimiter(api_config['rate_limit'])
    
    async def get_session(self):
        """Get aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'User-Agent': 'API-Enhanced-Bulk-Scanner/1.0'}
            )
        return self.session
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def discover_domains_commoncrawl(self, target_count: int = 10000) -> List[str]:
        """Discover domains from Common Crawl API"""
        api_config = self.config.config.get('common_crawl', {})
        if not api_config.get('enabled', False):
            return []
        
        logger.info(f"🔍 Discovering {target_count} domains from Common Crawl...")
        
        try:
            # Common Crawl domain discovery logic
            # This is a placeholder - actual implementation depends on API structure
            session = await self.get_session()
            
            # Rate limiting
            await self.rate_limiters['common_crawl'].acquire()
            
            # Example API call structure
            url = f"{api_config['base_url']}/domain-search"
            params = {
                'query': 'ads.txt',
                'limit': target_count,
                'filter': 'domain_quality:high'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    domains = data.get('domains', [])
                    logger.info(f"✅ Common Crawl discovered {len(domains)} domains")
                    return domains[:target_count]
                else:
                    logger.warning(f"Common Crawl API error: {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Common Crawl discovery failed: {e}")
            return []
    
    async def discover_domains_semrush(self, target_count: int = 10000) -> List[str]:
        """Discover domains from SEMrush API"""
        api_config = self.config.config.get('semrush', {})
        if not api_config.get('enabled', False):
            return []
        
        logger.info(f"🔍 Discovering {target_count} domains from SEMrush...")
        
        try:
            session = await self.get_session()
            
            await self.rate_limiters['semrush'].acquire()
            
            # SEMrush API structure (placeholder)
            url = f"{api_config['base_url']}/domain/search"
            params = {
                'key': api_config['api_key'],
                'query': 'advertising_enabled:true',
                'limit': target_count,
                'database': 'us'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    domains = [item['domain'] for item in data.get('results', [])]
                    logger.info(f"✅ SEMrush discovered {len(domains)} domains")
                    return domains[:target_count]
                else:
                    logger.warning(f"SEMrush API error: {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"SEMrush discovery failed: {e}")
            return []
    
    async def discover_domains_bulk(self, target_count: int = 50000) -> List[str]:
        """Discover domains from all enabled APIs"""
        logger.info(f"🚀 Starting bulk domain discovery for {target_count} domains...")
        
        enabled_apis = self.config.get_enabled_apis()
        if not enabled_apis:
            logger.warning("No APIs enabled for bulk discovery")
            return []
        
        all_domains = []
        per_api_target = target_count // len(enabled_apis)
        
        # Discover from each enabled API
        for api_name in enabled_apis:
            if api_name == 'common_crawl':
                domains = await self.discover_domains_commoncrawl(per_api_target)
            elif api_name == 'semrush':
                domains = await self.discover_domains_semrush(per_api_target)
            # Add more API integrations here
            else:
                continue
            
            all_domains.extend(domains)
        
        # Remove duplicates and filter
        unique_domains = list(set(all_domains))
        logger.info(f"🎯 Total unique domains discovered: {len(unique_domains)}")
        
        return unique_domains[:target_count]

class AsyncRateLimiter:
    """Async rate limiter for API calls"""
    
    def __init__(self, rate: float):
        self.rate = rate  # requests per second
        self.last_called = 0.0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit token"""
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_called
            min_interval = 1.0 / self.rate
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_called = time.time()

class BulkDomainProcessor:
    """Process domains at massive scale"""
    
    def __init__(self, db_path: str = "bulk_domain_discovery.db"):
        self.db_path = db_path
        self.scanner = FixedEnhancedDomainScanner(db_path)
        self.validator = RobustDomainValidator()
        self.setup_bulk_database()
    
    def setup_bulk_database(self):
        """Setup database for bulk operations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bulk batch tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bulk_batches (
                batch_id TEXT PRIMARY KEY,
                source_api TEXT,
                total_domains INTEGER,
                processed_domains INTEGER DEFAULT 0,
                approved_domains INTEGER DEFAULT 0,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT DEFAULT 'running',
                config_snapshot TEXT
            )
        ''')
        
        # API usage tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name TEXT,
                endpoint TEXT,
                requests_made INTEGER,
                date DATE,
                success_rate REAL,
                avg_response_time REAL
            )
        ''')
        
        # Use same detailed domain analysis table from existing scanner
        # This ensures compatibility and reuses all the enhanced scoring fields
        
        conn.commit()
        conn.close()
        logger.info("✅ Bulk processing database initialized")
    
    def create_batch(self, source_api: str, total_domains: int) -> str:
        """Create a new processing batch"""
        batch_id = f"batch_{int(time.time())}_{source_api}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bulk_batches 
            (batch_id, source_api, total_domains, started_at, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (batch_id, source_api, total_domains, datetime.now(), 'running'))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Created batch {batch_id} for {total_domains} domains")
        return batch_id
    
    def update_batch_progress(self, batch_id: str, processed: int, approved: int):
        """Update batch processing progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE bulk_batches 
            SET processed_domains = ?, approved_domains = ?
            WHERE batch_id = ?
        ''', (processed, approved, batch_id))
        
        conn.commit()
        conn.close()
    
    def complete_batch(self, batch_id: str):
        """Mark batch as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE bulk_batches 
            SET status = 'completed', completed_at = ?
            WHERE batch_id = ?
        ''', (datetime.now(), batch_id))
        
        conn.commit()
        conn.close()
    
    async def process_domain_batch(self, domains: List[str], batch_size: int = 100, max_concurrent: int = 50) -> List[ScanResult]:
        """Process domains in parallel batches"""
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_single_domain(domain: str) -> Optional[ScanResult]:
            async with semaphore:
                try:
                    # Reuse the existing scanner's analysis logic
                    result = await self.scanner.enhanced_scan_domain(domain, strategy='flexible', discovery_source='api_bulk')
                    return result
                except Exception as e:
                    logger.debug(f"Failed to process {domain}: {e}")
                    return None
        
        # Process in batches to manage memory and progress tracking
        for i in range(0, len(domains), batch_size):
            batch = domains[i:i + batch_size]
            logger.info(f"🔄 Processing batch {i//batch_size + 1}/{(len(domains) + batch_size - 1)//batch_size} ({len(batch)} domains)")
            
            # Process batch in parallel
            tasks = [process_single_domain(domain) for domain in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            for result in batch_results:
                if isinstance(result, ScanResult):
                    results.append(result)
            
            # Brief pause between batches
            await asyncio.sleep(1)
        
        return results
    
    async def bulk_discover_and_analyze(self, target_count: int = 50000, api_source: str = 'mixed') -> Dict:
        """Complete bulk discovery and analysis pipeline"""
        logger.info(f"🚀 Starting bulk discovery and analysis for {target_count} domains")
        
        # Create processing batch
        batch_id = self.create_batch(api_source, target_count)
        
        try:
            # Step 1: API Bulk Discovery
            api_config = APIConfig()
            api_manager = BulkAPIManager(api_config)
            discovered_domains = await api_manager.discover_domains_bulk(target_count)
            
            if not discovered_domains:
                logger.warning("No domains discovered from APIs, using fallback")
                # Fallback to existing discovery methods
                discovered_domains = self.scanner.fallback_domain_discovery(min(target_count, 1000))
            
            logger.info(f"📊 Discovered {len(discovered_domains)} domains for analysis")
            
            # Step 2: Bulk Validation (faster pre-filtering)
            logger.info("⚡ Pre-validating domains...")
            validated_domains = await self.validator.batch_validate(discovered_domains, max_concurrent=20)
            logger.info(f"📊 Validation: {len(validated_domains)}/{len(discovered_domains)} domains reachable")
            
            # Step 3: Bulk Analysis
            logger.info("🔍 Starting bulk domain analysis...")
            start_time = time.time()
            
            # Get bulk processing config
            bulk_config = api_config.config.get('bulk_processing', {})
            
            analysis_results = await self.process_domain_batch(
                validated_domains,
                batch_size=bulk_config.get('batch_size', 100),
                max_concurrent=bulk_config.get('max_concurrent', 50)
            )
            
            # Count approvals
            approved_results = [r for r in analysis_results if r.status == 'approved']
            
            analysis_time = time.time() - start_time
            
            # Update batch progress
            self.update_batch_progress(batch_id, len(analysis_results), len(approved_results))
            self.complete_batch(batch_id)
            
            # Final statistics
            stats = {
                'batch_id': batch_id,
                'total_discovered': len(discovered_domains),
                'total_validated': len(validated_domains),
                'total_analyzed': len(analysis_results),
                'total_approved': len(approved_results),
                'analysis_time_minutes': analysis_time / 60,
                'approval_rate': (len(approved_results) / len(analysis_results) * 100) if analysis_results else 0,
                'domains_per_minute': len(analysis_results) / (analysis_time / 60) if analysis_time > 0 else 0
            }
            
            logger.info(f"📈 BULK ANALYSIS COMPLETE")
            logger.info(f"   Batch ID: {batch_id}")
            logger.info(f"   Analyzed: {stats['total_analyzed']} domains")
            logger.info(f"   Approved: {stats['total_approved']} domains")
            logger.info(f"   Success rate: {stats['approval_rate']:.1f}%")
            logger.info(f"   Processing rate: {stats['domains_per_minute']:.1f} domains/minute")
            
            await api_manager.close_session()
            await self.validator.close_session()
            await self.scanner.close_session()
            
            return stats
            
        except Exception as e:
            logger.error(f"Bulk processing failed: {e}")
            # Mark batch as failed
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE bulk_batches SET status = "failed" WHERE batch_id = ?', (batch_id,))
            conn.commit()
            conn.close()
            
            return {'error': str(e), 'batch_id': batch_id}

async def run_bulk_discovery_pipeline(target_count: int = 50000, api_source: str = 'mixed') -> Dict:
    """Run the complete bulk discovery pipeline"""
    processor = BulkDomainProcessor()
    
    print("🚀 API-ENHANCED BULK DOMAIN DISCOVERY")
    print("=" * 60)
    print(f"Target: {target_count} domains")
    print(f"API Source: {api_source}")
    print()
    
    return await processor.bulk_discover_and_analyze(target_count, api_source)

async def main():
    """Test the bulk system"""
    # Test with smaller numbers first
    result = await run_bulk_discovery_pipeline(target_count=1000)
    print(f"\n🎉 Bulk discovery result: {result}")

if __name__ == "__main__":
    print("🚀 API-Enhanced Bulk Domain Scanner")
    print("Designed for 50k+ domain analysis at scale\n")
    
    asyncio.run(main())
