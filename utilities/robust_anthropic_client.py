#!/usr/bin/env python3
"""
Robust Anthropic Client with SSL Fix
Handles SSL certificate issues and provides reliable API connectivity
"""

import os
import ssl
import certifi
import asyncio
import time
import logging
from typing import Optional, Dict, List
import httpx

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustAnthropicClient:
    """Robust Anthropic client with SSL fixes and retry logic"""
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.max_retries = max_retries
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup Anthropic client with SSL fixes"""
        if not HAS_ANTHROPIC or not self.api_key:
            logger.warning("Anthropic not available or API key not set")
            return
        
        try:
            # Create SSL context with certifi certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Create custom httpx client with proper SSL configuration
            httpx_client = httpx.Client(
                verify=ssl_context,
                timeout=60.0,  # Generous timeout
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10
                )
            )
            
            # Create Anthropic client with custom httpx client
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                http_client=httpx_client,
                max_retries=2,  # Let us handle retries
                timeout=60.0
            )
            
            logger.info("✅ Robust Anthropic client initialized with SSL fixes")
            
        except Exception as e:
            logger.error(f"Failed to setup Anthropic client: {e}")
            self.client = None
    
    async def test_connection(self) -> Dict:
        """Test the connection with the robust client"""
        if not self.client:
            return {'success': False, 'error': 'Client not initialized'}
        
        try:
            logger.info("🔗 Testing robust Anthropic connection...")
            
            start_time = time.time()
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                messages=[{
                    "role": "user", 
                    "content": "Hello! Please respond with: 'Robust connection successful'"
                }]
            )
            duration = time.time() - start_time
            
            if response and response.content:
                content = getattr(response.content[0], 'text', '')
                return {
                    'success': True,
                    'response_time': duration,
                    'content': content
                }
            else:
                return {'success': False, 'error': 'Empty response'}
                
        except Exception as e:
            logger.error(f"Robust connection test failed: {e}")
            return {'success': False, 'error': str(e), 'error_type': type(e).__name__}
    
    async def discover_publishers(self, category: str, count: int = 20) -> List[str]:
        """Discover publishers using the robust client"""
        if not self.client:
            logger.warning("Client not available, returning empty list")
            return []
        
        category_prompts = {
            'business_finance': 'business, finance, economics publications',
            'technology': 'technology and software media sites',
            'industry_trade': 'professional trade publications',
            'marketing_advertising': 'marketing and advertising industry sites'
        }
        
        prompt = f"""List {count} high-quality {category_prompts.get(category, category)} domains for B2B advertising.

Requirements:
- Only domain names (e.g., "forbes.com")
- One per line
- Real domains that exist
- Focus on established publishers

Category: {category_prompts.get(category, category)}"""
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔍 Discovering {category} publishers (attempt {attempt + 1})")
                
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=800,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                if response and response.content:
                    content = getattr(response.content[0], 'text', '')
                    
                    # Parse domains from response
                    domains = []
                    for line in content.strip().split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Clean up domain format
                            domain = line.replace('www.', '').replace('http://', '').replace('https://', '')
                            domain = domain.split()[0]  # Take first word
                            
                            # Basic validation
                            if '.' in domain and len(domain) > 4 and len(domain) < 50:
                                domains.append(domain.lower())
                    
                    logger.info(f"✅ Discovered {len(domains)} domains for {category}")
                    return domains[:count]
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        logger.error(f"All attempts failed for {category}")
        return []
    
    async def validate_with_llm(self, domain: str, metrics: Dict) -> Dict:
        """Validate domain using LLM with robust client"""
        if not self.client:
            return {}
        
        prompt = f"""Analyze this website for B2B advertising quality:

Domain: {domain}
- Ads.txt: {metrics.get('ads_txt_found', False)}
- Premium DSPs: {metrics.get('premium_dsps', 0)}
- Content quality: {metrics.get('content_score', 0):.1f}/100

Provide:
1. RECOMMENDATION: APPROVE/REJECT/CAUTION
2. SCORE: 0-100
3. BRIEF reasoning

Focus on advertising value and brand safety."""
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                if response and response.content:
                    analysis = getattr(response.content[0], 'text', '')
                    
                    # Parse response
                    recommendation = "CAUTION"
                    score = 50.0
                    
                    if "APPROVE" in analysis.upper():
                        recommendation = "APPROVE"
                    elif "REJECT" in analysis.upper():
                        recommendation = "REJECT"
                    
                    # Extract score
                    import re
                    score_match = re.search(r'SCORE[:\s]*(\d+)', analysis.upper())
                    if score_match:
                        score = float(score_match.group(1))
                    
                    return {
                        'recommendation': recommendation,
                        'score': score,
                        'reasoning': analysis,
                        'success': True
                    }
                    
            except Exception as e:
                logger.warning(f"LLM validation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
        
        return {'success': False, 'error': 'All validation attempts failed'}

class SSLFixer:
    """SSL certificate management utilities"""
    
    @staticmethod
    def update_certificates():
        """Update SSL certificates using various methods"""
        import subprocess
        import sys
        
        print("🔧 ATTEMPTING SSL CERTIFICATE FIXES...")
        
        methods = []
        
        # Method 1: Install Certificates command (macOS Python)
        try:
            python_path = sys.executable
            python_dir = os.path.dirname(os.path.dirname(python_path))
            cert_command = os.path.join(python_dir, "Install Certificates.command")
            
            if os.path.exists(cert_command):
                methods.append(("Install Certificates.command", cert_command))
        except:
            pass
        
        # Method 2: Update certificates via pip
        methods.append(("Update certifi", [sys.executable, "-m", "pip", "install", "--upgrade", "certifi"]))
        
        # Method 3: macOS certificate update
        methods.append(("Update macOS certificates", ["brew", "install", "ca-certificates"]))
        
        for name, command in methods:
            try:
                print(f"   🔄 Trying: {name}")
                if isinstance(command, str):
                    subprocess.run([command], shell=True, capture_output=True)
                else:
                    subprocess.run(command, capture_output=True)
                print(f"   ✅ {name} completed")
            except Exception as e:
                print(f"   ⚠️  {name} failed: {e}")
    
    @staticmethod 
    def check_ssl_setup() -> Dict:
        """Check current SSL configuration"""
        results = {
            'certifi_available': False,
            'certifi_path': None,
            'ssl_context_works': False
        }
        
        try:
            import certifi
            results['certifi_available'] = True
            results['certifi_path'] = certifi.where()
        except ImportError:
            pass
        
        try:
            ssl_context = ssl.create_default_context()
            results['ssl_context_works'] = True
        except Exception:
            pass
        
        return results

async def main():
    """Test the robust client"""
    print("🚀 ROBUST ANTHROPIC CLIENT TEST")
    print("=" * 40)
    
    # Check SSL setup
    print("\n1️⃣ SSL Configuration Check:")
    ssl_info = SSLFixer.check_ssl_setup()
    for key, value in ssl_info.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key}: {value}")
    
    # Test robust client
    print("\n2️⃣ Robust Client Test:")
    client = RobustAnthropicClient()
    
    if client.client:
        # Test connection
        connection_result = await client.test_connection()
        if connection_result['success']:
            print(f"   ✅ Connection successful ({connection_result['response_time']:.2f}s)")
            print(f"   📝 Response: {connection_result['content']}")
            
            # Test publisher discovery
            print("\n3️⃣ Publisher Discovery Test:")
            domains = await client.discover_publishers('technology', 5)
            if domains:
                print(f"   ✅ Discovered {len(domains)} domains:")
                for domain in domains[:3]:
                    print(f"      - {domain}")
            else:
                print("   ❌ No domains discovered")
        else:
            print(f"   ❌ Connection failed: {connection_result['error']}")
            
            print("\n🔧 ATTEMPTING CERTIFICATE FIXES:")
            SSLFixer.update_certificates()
            
            print("\n   Retry connection after fixes...")
            client._setup_client()
            retry_result = await client.test_connection()
            if retry_result['success']:
                print("   ✅ Connection successful after fixes!")
            else:
                print(f"   ❌ Still failing: {retry_result['error']}")
    else:
        print("   ❌ Client initialization failed")

if __name__ == "__main__":
    asyncio.run(main())
