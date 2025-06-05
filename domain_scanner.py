#!/usr/bin/env python3
"""
Optimized Domain Scanner - Production Ready
Fixed schema, proper rejection reasons, configurable thresholds, improved discovery
"""

import asyncio
import aiohttp
import sqlite3
import re
import json
import logging
import os
import random
import yaml
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from bs4 import BeautifulSoup

# Anthropic import with SSL bypass
try:
    import anthropic
    import httpx
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Import validators
from utilities.robust_domain_validator import RobustDomainValidator

# Configure AGGRESSIVE logging suppression for connection errors
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('aiohttp').setLevel(logging.CRITICAL)
logging.getLogger('aiohttp.client').setLevel(logging.CRITICAL)
logging.getLogger('aiohttp.connector').setLevel(logging.CRITICAL)
logging.getLogger('aiohttp.client_exceptions').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('urllib3.connectionpool').setLevel(logging.CRITICAL)
logging.getLogger('ssl').setLevel(logging.CRITICAL)

# Domain scanner logger at INFO level for important messages only
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Completely suppress asyncio warnings about unhandled futures
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class ScanResult:
    """Enhanced scan result with detailed breakdown"""
    domain: str
    strategy: str
    status: str
    score: float
    rejection_reason: str  # Specific reason, not generic
    scoring_breakdown: Dict
    validation_time: float
    analysis_duration: float
    discovery_source: str
    config_snapshot: Dict
    analyzed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = "config/scanner_config.yaml"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration with intelligent defaults"""
        default_config = {
            'scoring': {
                'approval_threshold': 35,  # Lowered based on analysis
                'content_weight': 0.7,
                'ads_txt_bonus': 25,
                'premium_platform_bonus': 3,
                'b2b_relevance_weight': 0.25,
                'quality_indicator_bonus': 5
            },
            'validation': {
                'timeout_seconds': 15,
                'max_retries': 2,
                'dns_cache_ttl': 3600
            },
            'discovery': {
                'target_validation_rate': 0.20,  # 20% of discovered domains should be valid
                'fallback_threshold': 0.10,      # Switch to fallback if <10% valid
                'quality_keywords': [
                    'business', 'finance', 'technology', 'industry',
                    'professional', 'enterprise', 'news', 'media'
                ]
            },
            'analysis': {
                'track_performance': True,
                'detailed_logging': True,
                'export_borderline': True,
                'auto_threshold_suggestions': True
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f) or {}
                    # Deep merge configurations
                    self._deep_merge(default_config, loaded_config)
            else:
                # Create default config file
                self.save_config(default_config)
                logger.info(f"Created default config: {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def save_config(self, config: Dict):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_threshold(self) -> float:
        """Get current approval threshold"""
        return self.config['scoring']['approval_threshold']
    
    def get_config_snapshot(self) -> Dict:
        """Get configuration snapshot for storage"""
        return {
            'threshold': self.get_threshold(),
            'content_weight': self.config['scoring']['content_weight'],
            'ads_txt_bonus': self.config['scoring']['ads_txt_bonus'],
            'timestamp': datetime.now().isoformat()
        }

class OptimizedDomainScanner:
    """Production-ready domain scanner with optimized schema and tracking"""
    
    def __init__(self, db_path: str = "domain_discovery.db"):
        self.db_path = db_path
        self.session = None
        self.config = ConfigManager()
        self.validator = RobustDomainValidator()
        
        # Setup optimized database
        self.setup_optimized_database()
        
        # Load existing domains
        self.existing_domains = self.load_existing_domains()
        
        # Setup LLM client
        self._setup_llm_client()
        
        # Premium platforms for ads.txt analysis
        self.premium_platforms = {
            'google.com', 'googlesyndication.com', 'doubleclick.net',
            'amazon-adsystem.com', 'rubiconproject.com', 'openx.com',
            'pubmatic.com', 'appnexus.com', 'criteo.com', 'medianet.com',
            'sovrn.com', 'indexexchange.com', 'sharethrough.com', 'triplelift.com',
            'adsystem.amazon.com', 'adsync.amazon.com'
        }
        
        # Performance tracking
        self.session_stats = {
            'discovered': 0,
            'validated': 0,
            'approved': 0,
            'validation_failures': 0,
            'start_time': time.time()
        }
    
    def setup_optimized_database(self):
        """Setup optimized database schema for production"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main domains table with proper rejection tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domains_analyzed (
                domain TEXT PRIMARY KEY,
                first_analyzed_at TIMESTAMP,
                last_analyzed_at TIMESTAMP,
                analysis_count INTEGER DEFAULT 1,
                current_status TEXT,
                current_score REAL,
                strategy_used TEXT,
                rejection_reason TEXT,
                discovery_source TEXT,
                
                -- Validation tracking
                validation_successful BOOLEAN DEFAULT FALSE,
                validation_time_seconds REAL DEFAULT 0,
                analysis_duration_seconds REAL DEFAULT 0,
                
                -- Detailed scoring breakdown
                content_score REAL DEFAULT 0,
                has_meaningful_content BOOLEAN DEFAULT FALSE,
                ad_slots_detected INTEGER DEFAULT 0,
                quality_indicators INTEGER DEFAULT 0,
                b2b_relevance_score REAL DEFAULT 0,
                
                -- Ads.txt analysis
                has_ads_txt BOOLEAN DEFAULT FALSE,
                ads_txt_entries_count INTEGER DEFAULT 0,
                premium_platforms_count INTEGER DEFAULT 0,
                direct_deals_count INTEGER DEFAULT 0,
                
                -- Configuration snapshot
                threshold_used REAL DEFAULT 0,
                config_snapshot TEXT DEFAULT '{}',
                
                -- Additional metadata
                final_score_breakdown TEXT DEFAULT '{}'
            )
        ''')
        
        # Analysis sessions tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                session_id TEXT PRIMARY KEY,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                strategy TEXT,
                discovery_source TEXT,
                target_count INTEGER,
                
                -- Results
                domains_discovered INTEGER DEFAULT 0,
                domains_validated INTEGER DEFAULT 0,
                domains_approved INTEGER DEFAULT 0,
                validation_rate REAL DEFAULT 0,
                approval_rate REAL DEFAULT 0,
                
                -- Performance
                avg_validation_time REAL DEFAULT 0,
                avg_analysis_time REAL DEFAULT 0,
                
                -- Configuration used
                threshold_used REAL DEFAULT 0,
                config_snapshot TEXT DEFAULT '{}'
            )
        ''')
        
        # Performance monitoring
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                metric_value REAL,
                session_id TEXT,
                additional_data TEXT DEFAULT '{}'
            )
        ''')
        
        # Discovery quality tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discovery_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discovery_source TEXT,
                domains_attempted INTEGER,
                domains_reachable INTEGER,
                reachability_rate REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                quality_score REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Optimized database schema initialized")
    
    def _setup_llm_client(self):
        """Setup LLM client for domain discovery"""
        self.llm_client = None
        
        if not HAS_ANTHROPIC:
            logger.warning("Anthropic not available")
            return
            
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set")
            return
        
        try:
            httpx_client = httpx.Client(verify=False, timeout=60.0)
            self.llm_client = anthropic.Anthropic(
                api_key=api_key,
                http_client=httpx_client,
                timeout=60.0
            )
            logger.info("✅ LLM client initialized")
        except Exception as e:
            logger.error(f"Failed to setup LLM client: {e}")
    
    def load_existing_domains(self) -> Set[str]:
        """Load ALL previously attempted domains (smart memory)"""
        attempted_domains = set()
        
        # Load from existing_domains.txt (original DSP domains)
        try:
            with open('existing_domains.txt', 'r') as f:
                for line in f:
                    domain = line.strip().replace('www.', '')
                    if domain and not line.startswith('#'):
                        attempted_domains.add(domain)
            logger.info(f"Loaded {len(attempted_domains)} domains from existing_domains.txt")
        except FileNotFoundError:
            logger.warning("existing_domains.txt not found")
        
        # Load from service_discovered_domains.txt (service discoveries)
        try:
            with open('service_discovered_domains.txt', 'r') as f:
                service_count = 0
                for line in f:
                    domain = line.strip().replace('www.', '')
                    if domain and not line.startswith('#'):
                        attempted_domains.add(domain)
                        service_count += 1
            logger.info(f"Loaded {service_count} service-discovered domains from service_discovered_domains.txt")
        except FileNotFoundError:
            logger.info("service_discovered_domains.txt not found - will create on first approval")
        
        # Load ALL previously analyzed domains from database (approved AND rejected)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT domain FROM domains_analyzed")
            db_domains = {row[0] for row in cursor.fetchall()}
            conn.close()
            
            original_count = len(attempted_domains)
            attempted_domains.update(db_domains)
            logger.info(f"Added {len(db_domains)} previously analyzed domains from database")
            logger.info(f"Total attempted domains: {len(attempted_domains)} (was {original_count})")
            
        except Exception as e:
            logger.debug(f"Could not load database domains: {e}")
        
        return attempted_domains
    
    async def get_session(self):
        """Get aiohttp session with robust error suppression"""
        if self.session is None:
            # Create SSL context that completely ignores errors
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Conservative timeout and connection settings
            timeout = aiohttp.ClientTimeout(total=self.config.config['validation']['timeout_seconds'])
            
            # Create connector with aggressive error suppression
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=5,
                limit_per_host=2,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; OptimizedDomainScanner/2.0)'}
            )
        return self.session
    
    async def close_session(self):
        """Close sessions"""
        if self.session:
            await self.session.close()
            self.session = None
        await self.validator.close_session()
    
    async def improved_domain_discovery(self, count: int = 50) -> List[str]:
        """TURBO MODE: Smart domain discovery with aggressive scaling"""
        # TURBO MODE: Check for turbo settings to increase discovery count
        turbo_config = self.config.config.get('turbo_mode', {})
        if turbo_config.get('enabled', False):
            llm_discovery_count = turbo_config.get('llm_discovery_count', 150)
            logger.info(f"🚀 TURBO: Discovering {llm_discovery_count} candidates for {count} targets...")
        else:
            llm_discovery_count = count * 2
            logger.info(f"🧠 Discovering {count} new high-quality domains...")
        
        discovered_domains = []
        
        if self.llm_client:
            try:
                # Get comprehensive samples from all existing domain sources
                recent_attempts = self._get_recent_attempts_sample(50)
                existing_samples = self._get_existing_domain_samples(100)
                service_samples = self._get_service_domain_samples(50)
                
                exclusion_context = ""
                all_exclusions = []
                
                if recent_attempts:
                    all_exclusions.extend(recent_attempts)
                if existing_samples:
                    all_exclusions.extend(existing_samples)
                if service_samples:
                    all_exclusions.extend(service_samples)
                
                if all_exclusions:
                    # Remove duplicates and limit to most relevant
                    unique_exclusions = list(set(all_exclusions))[:150]  # Cap for prompt size
                    exclusion_context = f"\n\nDO NOT suggest these domains - they are already in our lists:\n{chr(10).join(unique_exclusions)}\n\nAvoid domains similar to these and find NEW, different sites.\n"
                
                # TURBO MODE: Enhanced prompt to avoid obvious tier-1 domains
                prompt = f"""As an expert in programmatic advertising, discover {count} NEW high-quality but LESS OBVIOUS publisher domains for a 10K domain scaling project:

🚫 **AVOID OBVIOUS TIER-1 DOMAINS** (these are too well-known):
- Major networks: CNN, ESPN, Fox, NBC, ABC, CBS, etc.
- Big tech: Google, Facebook, Apple, Microsoft, Amazon, etc.  
- Major publications: Time, Newsweek, WSJ, NYT, Washington Post, etc.
- Gaming giants: IGN, GameSpot, Kotaku, etc.
- Entertainment majors: Variety, Entertainment Weekly, TMZ, etc.

🎯 **TARGET: Quality but less obvious domains**:

🗞️ **Regional/Niche News & Media**:
- State/city newspapers (not major metros)
- Industry-specific news sites
- Regional sports coverage
- Local TV station websites
- Niche magazine websites

🏙️ **Local & Regional Business**:
- Chamber of commerce sites
- Regional business journals
- Local event/tourism sites  
- City-specific lifestyle magazines
- State trade organization sites

🎮 **Gaming & Tech (avoid majors)**:
- Gaming hardware review sites
- Esports team/league sites
- Gaming community forums
- Tech review smaller sites
- Industry trade publications

🎬 **Entertainment & Lifestyle (niche)**:
- Independent movie/TV blogs
- Regional entertainment guides
- Hobby/interest communities
- Music scene publications
- Local arts/culture sites

💼 **Professional Services**:
- Industry associations
- Professional development sites
- Trade publication websites
- Certification/training sites
- B2B service company blogs

{exclusion_context}

🔍 **REQUIREMENTS**:
- REAL websites with actual content and traffic
- Professional appearance with ad potential
- NOT personal blogs or tiny sites
- NOT obvious Fortune 500 company sites
- Focus on "hidden gems" with quality content

Return ONLY domain names (like "example.com"), one per line.
Think second-tier quality sites that programmatics teams would want but aren't obvious."""

                response = self.llm_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                analysis = getattr(response.content[0], 'text', '') if response.content else ""
                
                # Parse domains with better validation
                for line in analysis.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#') and '.' in line:
                        # Clean domain
                        domain = re.sub(r'^[^a-zA-Z0-9]*', '', line)
                        domain = re.sub(r'[^a-zA-Z0-9\.-]*$', '', domain)
                        domain = domain.replace('www.', '').replace('http://', '').replace('https://', '')
                        
                        # Validate format
                        if (re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,}$', domain) 
                            and len(domain) > 4 and len(domain) < 50):
                            discovered_domains.append(domain.lower())
                
                logger.info(f"✅ LLM discovered {len(discovered_domains)} candidate domains")
                
            except Exception as e:
                logger.warning(f"LLM discovery failed: {e}")
        
        # Smart fallback with category rotation
        if len(discovered_domains) < count // 2:
            logger.info("🔄 Using smart fallback discovery...")
            fallback_domains = self._get_quality_fallback_domains()
            discovered_domains.extend(fallback_domains)
        
        # Remove duplicates and ALL previously attempted domains
        unique_domains = []
        seen = set()
        skipped_existing = 0
        
        for domain in discovered_domains:
            if domain not in seen and domain not in self.existing_domains:
                unique_domains.append(domain)
                seen.add(domain)
            elif domain in self.existing_domains:
                skipped_existing += 1
        
        if skipped_existing > 0:
            logger.info(f"🧠 Smart exclusion: Skipped {skipped_existing} already-attempted domains")
        
        # Track discovery quality
        self._track_discovery_quality(len(unique_domains), 'llm_smart_exclusion')
        
        return unique_domains[:count]
    
    def _get_recent_attempts_sample(self, limit: int = 20) -> List[str]:
        """Get sample of recently attempted domains for LLM context"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT domain FROM domains_analyzed 
                ORDER BY last_analyzed_at DESC 
                LIMIT ?
            """, (limit,))
            recent = [row[0] for row in cursor.fetchall()]
            conn.close()
            return recent
        except Exception:
            return []
    
    def _get_existing_domain_samples(self, limit: int = 100) -> List[str]:
        """Get sample of domains from existing_domains.txt"""
        try:
            with open('existing_domains.txt', 'r') as f:
                domains = []
                for line in f:
                    domain = line.strip().replace('www.', '')
                    if domain and not line.startswith('#'):
                        domains.append(domain)
                        if len(domains) >= limit:
                            break
                return domains
        except Exception:
            return []
    
    def _get_service_domain_samples(self, limit: int = 50) -> List[str]:
        """Get sample of domains from service_discovered_domains.txt"""
        try:
            with open('service_discovered_domains.txt', 'r') as f:
                domains = []
                for line in f:
                    domain = line.strip().replace('www.', '')
                    if domain and not line.startswith('#'):
                        domains.append(domain)
                        if len(domains) >= limit:
                            break
                return domains
        except Exception:
            return []
    
    async def analyze_direct_domains(self, domains: List[str], discovery_source: str = 'direct_input') -> List[ScanResult]:
        """Analyze directly provided domains (future unlock feature)"""
        logger.info(f"🎯 Analyzing {len(domains)} directly provided domains...")
        
        results = []
        
        # Validate domains first
        validated_domains = await self.enhanced_validate_domains(domains)
        logger.info(f"⚡ {len(validated_domains)}/{len(domains)} domains are reachable")
        
        # Analyze each domain
        for i, domain in enumerate(validated_domains, 1):
            result = await self.enhanced_scan_domain(domain, discovery_source=discovery_source)
            if result:
                results.append(result)
            
            if i % 10 == 0:
                approved_count = len([r for r in results if r.status == 'approved'])
                logger.info(f"   Progress: {i}/{len(validated_domains)} analyzed, {approved_count} approved")
        
        approved_results = [r for r in results if r.status == 'approved']
        logger.info(f"🎯 Direct analysis complete: {len(approved_results)}/{len(results)} approved")
        
        return results
    
    def _get_quality_fallback_domains(self) -> List[str]:
        """High-quality fallback domains based on known publishers"""
        quality_domains = [
            # Business/Finance - Tier 1
            'marketwatch.com', 'bloomberg.com', 'reuters.com', 'wsj.com',
            'financialtimes.com', 'businessinsider.com', 'cnbc.com',
            
            # Technology - Tier 1  
            'techcrunch.com', 'venturebeat.com', 'ars-technica.com',
            'zdnet.com', 'computerworld.com', 'infoworld.com',
            
            # Industry Trade
            'adweek.com', 'mediapost.com', 'digiday.com', 'marketingland.com',
            'industrydive.com', 'supplychainbrain.com', 'manufacturingtalk.com',
            
            # Regional Business
            'bizjournals.com', 'crainsnewyork.com', 'chicagobusiness.com',
            'djournal.com', 'tampabay.com', 'denverbusiness.com',
            
            # B2B/Enterprise
            'firstround.com', 'techstars.com', 'crunchbase.com',
            'salesforce.com', 'hubspot.com', 'zendesk.com'
        ]
        
        # Filter out existing and randomize
        available = [d for d in quality_domains if d not in self.existing_domains]
        random.shuffle(available)
        return available
    
    def _track_discovery_quality(self, domains_found: int, source: str):
        """Track discovery source quality"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO discovery_quality 
            (discovery_source, domains_attempted, domains_reachable, recorded_at)
            VALUES (?, ?, 0, ?)
        ''', (source, domains_found, datetime.now()))
        
        conn.commit()
        conn.close()
    
    async def enhanced_validate_domains(self, domains: List[str]) -> List[str]:
        """TURBO MODE: Enhanced validation with aggressive performance settings"""
        logger.info(f"⚡ TURBO: Validating {len(domains)} domains...")
        
        # TURBO MODE: Check for turbo settings
        turbo_config = self.config.config.get('turbo_mode', {})
        if turbo_config.get('enabled', False):
            max_concurrent = turbo_config.get('concurrent_validation', 75)
            timeout_override = turbo_config.get('validation_timeout', 5)
        else:
            max_concurrent = 15
            timeout_override = None
        
        start_time = time.time()
        
        # TURBO MODE: Use aggressive concurrency and faster timeouts
        if timeout_override:
            # Temporarily override validator timeout for turbo mode
            original_timeout = self.validator.session
            validated_domains = await self.validator.batch_validate(domains, max_concurrent=max_concurrent)
        else:
            validated_domains = await self.validator.batch_validate(domains, max_concurrent=max_concurrent)
            
        validation_duration = time.time() - start_time
        
        # Update session stats
        self.session_stats['discovered'] += len(domains)
        self.session_stats['validated'] += len(validated_domains)
        self.session_stats['validation_failures'] += len(domains) - len(validated_domains)
        
        # Track validation rate
        validation_rate = len(validated_domains) / len(domains) if domains else 0
        
        if turbo_config.get('enabled', False):
            logger.info(f"🚀 TURBO Validation: {len(validated_domains)}/{len(domains)} ({validation_rate:.1%}) in {validation_duration:.1f}s with {max_concurrent} concurrent")
        else:
            logger.info(f"📊 Validation: {len(validated_domains)}/{len(domains)} ({validation_rate:.1%}) in {validation_duration:.1f}s")
        
        # Store performance metrics
        self._store_performance_metric('validation_rate', validation_rate)
        self._store_performance_metric('avg_validation_time', validation_duration / len(domains) if domains else 0)
        
        return validated_domains
    
    def _store_performance_metric(self, metric_name: str, value: float):
        """Store performance metric for monitoring"""
        if not self.config.config['analysis']['track_performance']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (metric_name, metric_value, recorded_at)
                VALUES (?, ?, ?)
            ''', (metric_name, value, datetime.now()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to store metric {metric_name}: {e}")
    
    async def check_ads_txt(self, domain: str) -> Tuple[bool, Dict]:
        """Enhanced ads.txt analysis with complete error suppression"""
        session = await self.get_session()
        
        # Create task with error suppression
        async def _safe_ads_txt_request():
            try:
                async with session.get(f"https://{domain}/ads.txt") as response:
                    if response.status == 200:
                        content = await response.text()
                        return True, self._parse_ads_txt(content)
                    return False, {}
            except Exception:
                return False, {}
        
        try:
            task = asyncio.create_task(_safe_ads_txt_request())
            # Add done callback to prevent "Future exception was never retrieved"
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return await task
        except Exception:
            return False, {}
    
    def _parse_ads_txt(self, content: str) -> Dict:
        """Enhanced ads.txt parsing"""
        lines = content.strip().split('\n')
        analysis = {
            'total_entries': 0,
            'direct_deals': 0,
            'reseller_deals': 0,
            'premium_platforms': [],
            'quality_score': 0
        }
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 3:
                    platform = parts[0].strip().lower()
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
        
        # Calculate quality score
        analysis['quality_score'] = min(100, 
            (analysis['direct_deals'] * 10) + 
            (len(analysis['premium_platforms']) * 15) +
            (analysis['total_entries'] * 2)
        )
        
        return analysis
    
    async def analyze_content(self, domain: str) -> Tuple[float, Dict]:
        """Enhanced content analysis with IAB category detection and risk assessment"""
        session = await self.get_session()
        content_details = {
            'has_meaningful_content': False,
            'ad_slots_detected': 0,
            'quality_indicators': 0,
            'b2b_relevance_score': 0,
            'content_length': 0,
            'professional_indicators': 0,
            'iab_category': None,
            'iab_category_confidence': 0,
            'risk_content_detected': False,
            'risk_keywords_found': []
        }
        
        # Create task with error suppression
        async def _safe_content_request():
            try:
                async with session.get(f"https://{domain}", allow_redirects=True) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Clean text extraction
                        for script in soup(["script", "style", "nav", "footer", "header"]):
                            script.decompose()
                        
                        text_content = soup.get_text().lower()
                        content_details['content_length'] = len(text_content)
                        content_details['has_meaningful_content'] = len(text_content) > 1000
                        
                        # Enhanced ad slot detection
                        ad_selectors = [
                            {'class_': re.compile(r'ad|banner|advertisement|sponsor|adsense')},
                            {'id': re.compile(r'ad|banner|advertisement|sponsor')},
                            'iframe[src*="googlesyndication"]',
                            'script[src*="googlesyndication"]',
                            'script[src*="amazon-adsystem"]'
                        ]
                        
                        ad_count = 0
                        for selector in ad_selectors:
                            if isinstance(selector, dict):
                                ad_count += len(soup.find_all('div', **selector))
                            else:
                                ad_count += len(soup.select(selector))
                        
                        content_details['ad_slots_detected'] = min(10, ad_count)
                        
                        # Enhanced quality indicators
                        quality_terms = [
                            'subscribe', 'newsletter', 'premium', 'insights', 'analysis',
                            'whitepaper', 'report', 'research', 'industry', 'professional'
                        ]
                        
                        quality_count = sum(1 for term in quality_terms if term in text_content)
                        content_details['quality_indicators'] = min(10, quality_count)
                        
                        # Enhanced B2B relevance scoring
                        b2b_keywords = self.config.config['discovery']['quality_keywords']
                        b2b_count = sum(text_content.count(keyword) for keyword in b2b_keywords)
                        content_details['b2b_relevance_score'] = min(100, (b2b_count / len(b2b_keywords)) * 20)
                        
                        # Professional indicators
                        professional_terms = ['enterprise', 'solution', 'platform', 'service', 'technology']
                        professional_count = sum(1 for term in professional_terms if term in text_content)
                        content_details['professional_indicators'] = min(5, professional_count)
                        
                        # IAB CATEGORY DETECTION
                        iab_result = self._detect_iab_category(text_content, html_content)
                        content_details['iab_category'] = iab_result['category']
                        content_details['iab_category_confidence'] = iab_result['confidence']
                        
                        # RISK CONTENT DETECTION
                        risk_result = self._detect_risk_content(text_content)
                        content_details['risk_content_detected'] = risk_result['detected']
                        content_details['risk_keywords_found'] = risk_result['keywords']
                        
            except Exception:
                pass  # Silently ignore all content analysis errors
        
        try:
            task = asyncio.create_task(_safe_content_request())
            # Add done callback to prevent "Future exception was never retrieved"
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            await task
        except Exception:
            pass  # Silently ignore all errors
        
        # Enhanced scoring algorithm
        score = self._calculate_content_score(content_details)
        return score, content_details
    
    def _detect_iab_category(self, text_content: str, html_content: str) -> Dict:
        """IAB category detection using content analysis"""
        iab_config = self.config.config.get('iab_categories', {})
        target_categories = iab_config.get('target_categories', [])
        
        # IAB category mapping patterns
        category_patterns = {
            'IAB19': ['business', 'entrepreneur', 'startup', 'company', 'corporate', 'enterprise'],
            'IAB19-1': ['advertising', 'marketing', 'campaign', 'brand', 'promotion'],
            'IAB19-3': ['career', 'job', 'employment', 'hiring', 'recruitment', 'resume'],
            'IAB19-8': ['marketing', 'digital marketing', 'content marketing', 'seo', 'sem'],
            'IAB3': ['technology', 'tech', 'software', 'digital', 'innovation'],
            'IAB3-9': ['enterprise technology', 'saas', 'cloud', 'platform', 'solution'],
            'IAB3-13': ['programming', 'development', 'coding', 'developer', 'software engineer'],
            'IAB13': ['finance', 'financial', 'investment', 'banking', 'money'],
            'IAB15': ['news', 'breaking', 'report', 'journalist', 'media'],
            'IAB11': ['science', 'research', 'study', 'scientific', 'laboratory'],
            'IAB5': ['education', 'learning', 'training', 'course', 'university']
        }
        
        best_category = None
        best_confidence = 0
        
        for category, keywords in category_patterns.items():
            if category in target_categories:
                matches = sum(1 for keyword in keywords if keyword in text_content)
                confidence = min(100, (matches / len(keywords)) * 100)
                
                if confidence > best_confidence:
                    best_category = category
                    best_confidence = confidence
        
        return {
            'category': best_category,
            'confidence': best_confidence
        }
    
    def _detect_risk_content(self, text_content: str) -> Dict:
        """Risk content detection for brand safety"""
        iab_config = self.config.config.get('iab_categories', {})
        risky_categories = iab_config.get('risky_categories', [])
        
        # Risk keyword patterns
        risk_patterns = {
            'cannabis': ['cannabis', 'marijuana', 'weed', 'cbd', 'thc', 'hemp', 'dispensary'],
            'adult': ['adult', 'porn', 'xxx', 'sex', 'escort', 'nude'],
            'gambling': ['casino', 'poker', 'betting', 'gambling', 'lottery', 'jackpot'],
            'politics': ['political', 'election', 'vote', 'democrat', 'republican', 'partisan'],
            'controversial': ['controversial', 'scandal', 'protest', 'violence', 'hate'],
            'weapons': ['gun', 'weapon', 'firearm', 'rifle', 'ammunition']
        }
        
        detected_risks = []
        risk_keywords_found = []
        
        for risk_type, keywords in risk_patterns.items():
            if risk_type in risky_categories:
                found_keywords = [kw for kw in keywords if kw in text_content]
                if found_keywords:
                    detected_risks.append(risk_type)
                    risk_keywords_found.extend(found_keywords)
        
        return {
            'detected': len(detected_risks) > 0,
            'categories': detected_risks,
            'keywords': risk_keywords_found
        }
    
    def _calculate_content_score(self, content_details: Dict) -> float:
        """Enhanced content score calculation with IAB categories and risk assessment"""
        config = self.config.config['scoring']
        iab_config = self.config.config.get('iab_categories', {})
        
        score = 0
        
        # Base content score
        if content_details['has_meaningful_content']:
            score += 30
        
        # Ad slots (indicates monetization)
        score += min(20, content_details['ad_slots_detected'] * 3)
        
        # Quality indicators
        score += content_details['quality_indicators'] * config['quality_indicator_bonus']
        
        # B2B relevance
        score += content_details['b2b_relevance_score'] * config['b2b_relevance_weight']
        
        # Professional indicators
        score += content_details['professional_indicators'] * 3
        
        # IAB CATEGORY BONUSES
        iab_category = content_details.get('iab_category')
        iab_confidence = content_details.get('iab_category_confidence', 0)
        
        if iab_category and iab_confidence > 20:  # Minimum confidence threshold
            scoring_config = iab_config.get('scoring', {})
            
            # Perfect match categories (IAB19, IAB3, IAB13)
            if iab_category in ['IAB19', 'IAB3', 'IAB13'] or iab_category.startswith(('IAB19-', 'IAB3-')):
                bonus = scoring_config.get('perfect_match_bonus', 30)
                score += bonus * (iab_confidence / 100)
                logger.info(f"🎯 IAB Perfect Match: {iab_category} (+{bonus * (iab_confidence / 100):.1f})")
            
            # Good match categories
            elif iab_category in ['IAB15', 'IAB11', 'IAB5']:
                bonus = scoring_config.get('good_match_bonus', 20)
                score += bonus * (iab_confidence / 100)
                logger.info(f"✅ IAB Good Match: {iab_category} (+{bonus * (iab_confidence / 100):.1f})")
            
            # Moderate match categories
            elif iab_category in ['IAB20', 'IAB1']:
                bonus = scoring_config.get('moderate_match_bonus', 10)
                score += bonus * (iab_confidence / 100)
                logger.info(f"🟡 IAB Moderate Match: {iab_category} (+{bonus * (iab_confidence / 100):.1f})")
        
        # RISK CONTENT PENALTIES
        if content_details.get('risk_content_detected', False):
            penalty = iab_config.get('scoring', {}).get('risk_content_penalty', -50)
            score += penalty
            risk_keywords = content_details.get('risk_keywords_found', [])
            logger.warning(f"🚨 RISK CONTENT DETECTED: {risk_keywords} ({penalty} penalty)")
        
        return min(100, score)
    
    async def enhanced_scan_domain(self, domain: str, strategy: str = 'flexible', 
                                 discovery_source: str = 'llm_improved') -> Optional[ScanResult]:
        """Enhanced domain scanning with proper rejection tracking"""
        
        # Skip existing domains
        if domain in self.existing_domains:
            logger.debug(f"⏩ Skipping {domain} - already exists")
            return None
        
        analysis_start = time.time()
        
        # Validation phase
        validation_start = time.time()
        if not await self.validator.quick_validate(domain):
            validation_time = time.time() - validation_start
            
            return ScanResult(
                domain=domain,
                strategy=strategy,
                status='rejected',
                score=0,
                rejection_reason='domain_not_reachable',  # Specific reason
                scoring_breakdown={},
                validation_time=validation_time,
                analysis_duration=time.time() - analysis_start,
                discovery_source=discovery_source,
                config_snapshot=self.config.get_config_snapshot()
            )
        
        validation_time = time.time() - validation_start
        
        try:
            # Content analysis
            content_score, content_details = await self.analyze_content(domain)
            
            # Ads.txt analysis
            has_ads_txt, ads_analysis = await self.check_ads_txt(domain)
            
            # Calculate final score
            scoring_breakdown = {
                'content_score': content_score,
                'ads_txt_bonus': self.config.config['scoring']['ads_txt_bonus'] if has_ads_txt else 0,
                'premium_platform_bonus': len(ads_analysis.get('premium_platforms', [])) * self.config.config['scoring']['premium_platform_bonus'],
                'total_before_threshold': 0
            }
            
            # Weighted final score with proper cap at 100
            raw_final_score = (content_score * self.config.config['scoring']['content_weight'] + 
                              scoring_breakdown['ads_txt_bonus'] + 
                              scoring_breakdown['premium_platform_bonus'])
            
            final_score = min(100.0, raw_final_score)  # Cap at 100
            
            scoring_breakdown['total_before_threshold'] = raw_final_score
            scoring_breakdown['final_capped_score'] = final_score
            
            # IAB CATEGORY & RISK-BASED REJECTION LOGIC
            threshold = self.config.get_threshold()
            
            # Check for RISK CONTENT first (immediate rejection)
            if content_details.get('risk_content_detected', False):
                status = 'rejected'
                risk_keywords = content_details.get('risk_keywords_found', [])
                rejection_reason = f'risk_content_detected_{risk_keywords[0] if risk_keywords else "unknown"}'
                logger.warning(f"🚨 RISK REJECTION: {domain} - {risk_keywords}")
            
            elif final_score >= threshold:
                status = 'approved'
                rejection_reason = 'approved'
                
                # Log IAB category info for approved domains
                iab_category = content_details.get('iab_category')
                if iab_category:
                    logger.info(f"🎯 APPROVED with IAB: {domain} - {iab_category}")
            
            else:
                status = 'rejected'
                
                # Standard rejection reasons (no IAB category requirement)
                if content_score < 20:
                    rejection_reason = 'insufficient_content'
                elif content_score < 40 and not has_ads_txt:
                    rejection_reason = 'low_content_no_ads_txt'
                elif final_score < threshold - 10:
                    rejection_reason = 'significantly_below_threshold'
                else:
                    rejection_reason = 'below_threshold'
            
            result = ScanResult(
                domain=domain,
                strategy=strategy,
                status=status,
                score=final_score,
                rejection_reason=rejection_reason,
                scoring_breakdown=scoring_breakdown,
                validation_time=validation_time,
                analysis_duration=time.time() - analysis_start,
                discovery_source=discovery_source,
                config_snapshot=self.config.get_config_snapshot()
            )
            
            # Save to database
            self._save_enhanced_result(result, content_details, ads_analysis)
            
            # Log result
            if status == 'approved':
                logger.info(f"✅ APPROVED: {domain} (Score: {final_score:.1f})")
                self.session_stats['approved'] += 1
            else:
                logger.info(f"❌ REJECTED: {domain} (Score: {final_score:.1f}, Reason: {rejection_reason})")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced scan failed for {domain}: {e}")
            return ScanResult(
                domain=domain,
                strategy=strategy,
                status='rejected',
                score=0,
                rejection_reason='analysis_error',
                scoring_breakdown={},
                validation_time=validation_time,
                analysis_duration=time.time() - analysis_start,
                discovery_source=discovery_source,
                config_snapshot=self.config.get_config_snapshot()
            )
    
    def _save_approved_domain_live(self, domain: str, score: float):
        """Immediately save approved domain to live service file"""
        try:
            with open('service_discovered_domains.txt', 'a') as f:
                f.write(f"{domain}\n")
            logger.info(f"💾 Added {domain} to service_discovered_domains.txt")
        except Exception as e:
            logger.error(f"Failed to save {domain} to live file: {e}")
    
    def _save_enhanced_result(self, result: ScanResult, content_details: Dict, ads_analysis: Dict):
        """Save enhanced result to optimized database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO domains_analyzed (
                domain, first_analyzed_at, last_analyzed_at, analysis_count,
                current_status, current_score, strategy_used, rejection_reason, discovery_source,
                validation_successful, validation_time_seconds, analysis_duration_seconds,
                content_score, has_meaningful_content, ad_slots_detected, quality_indicators, b2b_relevance_score,
                has_ads_txt, ads_txt_entries_count, premium_platforms_count, direct_deals_count,
                threshold_used, config_snapshot, final_score_breakdown
            ) VALUES (
                ?, 
                COALESCE((SELECT first_analyzed_at FROM domains_analyzed WHERE domain = ?), ?),
                ?, 
                COALESCE((SELECT analysis_count FROM domains_analyzed WHERE domain = ?) + 1, 1),
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?
            )
        ''', (
            result.domain, result.domain, result.analyzed_at, result.analyzed_at,
            result.domain, result.status, result.score, result.strategy, result.rejection_reason, result.discovery_source,
            result.status != 'rejected' or result.rejection_reason != 'domain_not_reachable',
            result.validation_time, result.analysis_duration,
            result.scoring_breakdown.get('content_score', 0),
            content_details.get('has_meaningful_content', False),
            content_details.get('ad_slots_detected', 0),
            content_details.get('quality_indicators', 0),
            content_details.get('b2b_relevance_score', 0),
            ads_analysis.get('total_entries', 0) > 0,
            ads_analysis.get('total_entries', 0),
            len(ads_analysis.get('premium_platforms', [])),
            ads_analysis.get('direct_deals', 0),
            self.config.get_threshold(),
            json.dumps(result.config_snapshot),
            json.dumps(result.scoring_breakdown)
        ))
        
        conn.commit()
        conn.close()
        
        # If domain is approved, immediately save to live file
        if result.status == 'approved':
            self._save_approved_domain_live(result.domain, result.score)
    
    async def run_optimized_discovery_pipeline(self, target_count: int = 30) -> Dict:
        """Run optimized discovery pipeline"""
        session_id = f"session_{int(time.time())}"
        
        print("🚀 OPTIMIZED DOMAIN DISCOVERY PIPELINE")
        print("=" * 50)
        print(f"Target: {target_count} domains")
        print(f"Threshold: {self.config.get_threshold()}")
        print(f"Session ID: {session_id}")
        print()
        
        # Store session start
        self._start_session(session_id, target_count)
        
        try:
            # Step 1: Improved domain discovery
            discovered_domains = await self.improved_domain_discovery(target_count * 2)
            print(f"🧠 Discovered: {len(discovered_domains)} candidate domains")
            
            # Step 2: Enhanced validation
            validated_domains = await self.enhanced_validate_domains(discovered_domains)
            print(f"⚡ Validated: {len(validated_domains)} reachable domains")
            
            # Step 3: Enhanced analysis
            approved_results = []
            for i, domain in enumerate(validated_domains[:target_count], 1):
                result = await self.enhanced_scan_domain(domain)
                if result and result.status == 'approved':
                    approved_results.append(result)
                
                if i % 5 == 0:
                    print(f"   Progress: {i}/{min(len(validated_domains), target_count)} analyzed, {len(approved_results)} approved")
            
            # Update session completion
            self._complete_session(session_id, discovered_domains, validated_domains, approved_results)
            
            # Final stats with safe division
            runtime = time.time() - self.session_stats['start_time']
            
            # Safe rate calculations to prevent division by zero
            discovery_rate = (len(validated_domains)/max(1, len(discovered_domains))*100) if discovered_domains else 0.0
            approval_rate = (len(approved_results)/max(1, len(validated_domains))*100) if validated_domains else 0.0
            processing_rate = (len(validated_domains)/max(0.1, runtime/60)) if runtime > 0 else 0.0
            
            print(f"\n📈 OPTIMIZED PIPELINE COMPLETE")
            print(f"   Discovery rate: {discovery_rate:.1f}%")
            print(f"   Approval rate: {approval_rate:.1f}% (of validated)")
            print(f"   Processing rate: {processing_rate:.1f} domains/minute")
            print(f"   New approvals: {len(approved_results)}")
            
            return {
                'session_id': session_id,
                'discovered': len(discovered_domains),
                'validated': len(validated_domains),
                'approved': len(approved_results),
                'discovery_rate': len(validated_domains)/len(discovered_domains)*100 if discovered_domains else 0,
                'approval_rate': len(approved_results)/len(validated_domains)*100 if validated_domains else 0,
                'runtime_minutes': runtime/60
            }
            
        finally:
            await self.close_session()
    
    def _start_session(self, session_id: str, target_count: int):
        """Record session start"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_sessions 
            (session_id, started_at, strategy, discovery_source, target_count, threshold_used, config_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, datetime.now(), 'flexible', 'llm_improved', target_count,
            self.config.get_threshold(), json.dumps(self.config.get_config_snapshot())
        ))
        
        conn.commit()
        conn.close()
    
    def _complete_session(self, session_id: str, discovered: List[str], validated: List[str], approved: List):
        """Record session completion"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        validation_rate = len(validated) / len(discovered) if discovered else 0
        approval_rate = len(approved) / len(validated) if validated else 0
        
        cursor.execute('''
            UPDATE analysis_sessions SET
                completed_at = ?,
                domains_discovered = ?,
                domains_validated = ?,
                domains_approved = ?,
                validation_rate = ?,
                approval_rate = ?
            WHERE session_id = ?
        ''', (
            datetime.now(), len(discovered), len(validated), len(approved),
            validation_rate, approval_rate, session_id
        ))
        
        conn.commit()
        conn.close()

# Pipeline runner function
async def run_optimized_discovery_pipeline(target_count: int = 30) -> Dict:
    """Run the optimized discovery pipeline"""
    scanner = OptimizedDomainScanner()
    return await scanner.run_optimized_discovery_pipeline(target_count)

async def main():
    """Test the optimized system"""
    result = await run_optimized_discovery_pipeline(target_count=20)
    print(f"\n🎉 Optimized pipeline result: {result}")

if __name__ == "__main__":
    print("🚀 Optimized Domain Scanner - Production Ready")
    print("Features: Proper rejection tracking, configurable thresholds, improved discovery")
    print()
    
    asyncio.run(main())
