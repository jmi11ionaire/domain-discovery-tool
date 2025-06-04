#!/usr/bin/env python3
"""
Robust Domain Validator with SSL Error Handling
Handles SSL connection errors gracefully during domain validation
"""

import asyncio
import aiohttp
import socket
import ssl
import logging
from typing import List, Optional
import re

# Configure AGGRESSIVE logging suppression for ALL connection issues
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('aiohttp').setLevel(logging.CRITICAL)
logging.getLogger('aiohttp.client').setLevel(logging.CRITICAL)
logging.getLogger('aiohttp.connector').setLevel(logging.CRITICAL)
logging.getLogger('aiohttp.client_exceptions').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('ssl').setLevel(logging.CRITICAL)

# Domain validator logger - only show important messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Completely suppress asyncio warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Suppress specific aiohttp future warnings
warnings.filterwarnings("ignore", message=".*Future exception was never retrieved.*")

class RobustDomainValidator:
    """Domain validator with robust SSL error handling"""
    
    def __init__(self):
        self.session = None
        self.dns_cache = {}
        
    async def get_session(self):
        """Get validation session with robust SSL settings"""
        if self.session is None:
            # Create SSL context that's more forgiving
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Conservative timeout and connection settings
            timeout = aiohttp.ClientTimeout(total=15, connect=5, sock_read=5)
            
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=10,
                limit_per_host=3,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; DomainValidator/2.0)'}
            )
        return self.session
    
    async def close_session(self):
        """Close validation session"""
        if self.session:
            try:
                await self.session.close()
            except Exception:
                pass  # Ignore close errors
            self.session = None
    
    def dns_resolves(self, domain: str) -> bool:
        """Quick DNS resolution check with caching"""
        if domain in self.dns_cache:
            return self.dns_cache[domain]
        
        try:
            socket.gethostbyname(domain)
            self.dns_cache[domain] = True
            return True
        except (socket.gaierror, socket.timeout):
            self.dns_cache[domain] = False
            return False
    
    async def safe_head_request(self, url: str) -> bool:
        """Make a HEAD request with comprehensive error handling"""
        session = await self.get_session()
        
        try:
            async with session.head(url, allow_redirects=True) as response:
                # Accept any reasonable HTTP status
                return response.status in [200, 301, 302, 403, 401, 405]
        except (
            aiohttp.ClientConnectionError,
            aiohttp.ClientSSLError,
            aiohttp.ServerTimeoutError,
            aiohttp.ClientPayloadError,
            ssl.SSLError,
            asyncio.TimeoutError,
            OSError
        ) as e:
            logger.debug(f"Connection error for {url}: {type(e).__name__}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error for {url}: {e}")
            return False
    
    async def is_reachable(self, domain: str) -> bool:
        """Check if domain is reachable with robust error handling"""
        if not self.dns_resolves(domain):
            return False
        
        # Try both HTTPS and HTTP
        for protocol in ['https', 'http']:
            url = f"{protocol}://{domain}"
            if await self.safe_head_request(url):
                return True
        
        return False
    
    async def quick_validate(self, domain: str) -> bool:
        """Fast validation with comprehensive error handling"""
        logger.debug(f"🔍 Validating: {domain}")
        
        # Basic format check
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,}$', domain):
            return False
        
        # DNS + reachability with error handling
        try:
            is_reachable = await self.is_reachable(domain)
            if is_reachable:
                logger.debug(f"✅ {domain} validated")
            else:
                logger.debug(f"❌ {domain} not reachable")
            return is_reachable
        except Exception as e:
            logger.debug(f"❌ {domain} validation error: {e}")
            return False
    
    async def batch_validate(self, domains: List[str], max_concurrent: int = 3) -> List[str]:
        """Validate multiple domains with enhanced error handling"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_single(domain: str) -> Optional[str]:
            """Validate single domain with complete exception suppression"""
            async with semaphore:
                try:
                    await asyncio.sleep(0.1)  # Small delay
                    result = await self.quick_validate(domain)
                    return domain if result else None
                except Exception:
                    return None
        
        # Process all domains with proper task handling
        valid_domains = []
        tasks = [asyncio.create_task(validate_single(domain)) for domain in domains]
        
        try:
            # Use as_completed to process results as they finish
            for completed_task in asyncio.as_completed(tasks, timeout=120):
                try:
                    result = await completed_task
                    if result:
                        valid_domains.append(result)
                except Exception:
                    # Silently ignore individual validation failures
                    pass
                    
        except asyncio.TimeoutError:
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
        except Exception:
            # Cancel all tasks on any other error
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        logger.info(f"📊 Validation: {len(valid_domains)}/{len(domains)} domains reachable")
        return valid_domains

# Test the robust validator
async def test_validator():
    """Test the robust validator"""
    validator = RobustDomainValidator()
    
    test_domains = [
        'google.com',
        'nonexistent-domain-12345.com',
        'example.com',
        'badssl-expired.com',  # Domain with SSL issues
        'spotify.com'
    ]
    
    print("🧪 Testing Robust Domain Validator")
    print("=" * 40)
    
    valid_domains = await validator.batch_validate(test_domains)
    
    print(f"✅ Validated domains: {valid_domains}")
    print(f"📊 Success rate: {len(valid_domains)}/{len(test_domains)}")
    
    await validator.close_session()

if __name__ == "__main__":
    asyncio.run(test_validator())
