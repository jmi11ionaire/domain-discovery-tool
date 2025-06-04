#!/usr/bin/env python3
"""
Anthropic API Diagnostics and Testing
Diagnose and fix connection issues with Anthropic API
"""

import os
import asyncio
import time
import logging
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AnthropicDiagnostics:
    """Comprehensive Anthropic API diagnostics"""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
    def check_environment(self) -> dict:
        """Check environment setup"""
        results = {
            'anthropic_installed': HAS_ANTHROPIC,
            'api_key_set': bool(self.api_key),
            'api_key_format': None,
            'environment_ok': False
        }
        
        if self.api_key:
            # Check API key format (should start with sk-ant-)
            if self.api_key.startswith('sk-ant-'):
                results['api_key_format'] = 'valid_format'
            else:
                results['api_key_format'] = 'invalid_format'
                
        results['environment_ok'] = (
            results['anthropic_installed'] and 
            results['api_key_set'] and 
            results['api_key_format'] == 'valid_format'
        )
        
        return results
    
    async def test_basic_connection(self) -> dict:
        """Test basic API connection"""
        if not HAS_ANTHROPIC or not self.api_key:
            return {'success': False, 'error': 'Environment not setup'}
        
        try:
            # Create client with custom timeout
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=30.0  # 30 second timeout
            )
            
            logger.info("Testing basic Anthropic API connection...")
            
            start_time = time.time()
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                messages=[{
                    "role": "user", 
                    "content": "Hello! Please respond with exactly: 'API connection test successful'"
                }]
            )
            duration = time.time() - start_time
            
            if response and response.content:
                content = getattr(response.content[0], 'text', '')
                return {
                    'success': True,
                    'response_time': duration,
                    'response_content': content,
                    'model_used': 'claude-3-haiku-20240307'
                }
            else:
                return {
                    'success': False,
                    'error': 'Empty response from API'
                }
                
        except Exception as e:
            logger.error(f"Basic connection test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    async def test_concurrent_requests(self, count: int = 3) -> dict:
        """Test multiple concurrent requests"""
        if not self.client:
            return {'success': False, 'error': 'Client not initialized'}
        
        async def single_request(request_id: int):
            try:
                start_time = time.time()
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=30,
                    messages=[{
                        "role": "user", 
                        "content": f"Request {request_id}: Say 'Success {request_id}'"
                    }]
                )
                duration = time.time() - start_time
                
                content = getattr(response.content[0], 'text', '') if response.content else ''
                return {
                    'request_id': request_id,
                    'success': True,
                    'duration': duration,
                    'content': content
                }
            except Exception as e:
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
        
        logger.info(f"Testing {count} concurrent requests...")
        
        # Run concurrent requests
        tasks = [single_request(i) for i in range(count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = count - successful
        
        return {
            'total_requests': count,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / count) * 100,
            'individual_results': results
        }
    
    async def test_different_timeouts(self) -> dict:
        """Test with different timeout configurations"""
        if not HAS_ANTHROPIC or not self.api_key:
            return {'success': False, 'error': 'Environment not setup'}
        
        timeout_configs = [5, 15, 30, 60]  # Different timeout values
        results = {}
        
        for timeout in timeout_configs:
            logger.info(f"Testing with {timeout}s timeout...")
            
            try:
                client = anthropic.Anthropic(
                    api_key=self.api_key,
                    timeout=timeout
                )
                
                start_time = time.time()
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=100,
                    messages=[{
                        "role": "user", 
                        "content": "Generate a list of 5 business publication domains (just domain names, one per line)"
                    }]
                )
                duration = time.time() - start_time
                
                content = getattr(response.content[0], 'text', '') if response.content else ''
                
                results[f'timeout_{timeout}s'] = {
                    'success': True,
                    'duration': duration,
                    'content_length': len(content),
                    'timed_out': duration >= timeout
                }
                
            except Exception as e:
                results[f'timeout_{timeout}s'] = {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
        
        return results
    
    async def run_full_diagnostics(self) -> dict:
        """Run complete diagnostic suite"""
        print("🔍 ANTHROPIC API DIAGNOSTICS")
        print("=" * 40)
        
        # 1. Environment check
        print("\n1️⃣ Environment Check:")
        env_results = self.check_environment()
        for key, value in env_results.items():
            status = "✅" if value else "❌"
            print(f"   {status} {key}: {value}")
        
        if not env_results['environment_ok']:
            print("\n❌ Environment issues detected. Please fix before proceeding.")
            return {'overall_status': 'failed', 'stage': 'environment'}
        
        # 2. Basic connection test
        print("\n2️⃣ Basic Connection Test:")
        basic_test = await self.test_basic_connection()
        if basic_test['success']:
            print(f"   ✅ Connection successful ({basic_test['response_time']:.2f}s)")
            print(f"   📝 Response: {basic_test['response_content'][:100]}...")
        else:
            print(f"   ❌ Connection failed: {basic_test['error']}")
            return {'overall_status': 'failed', 'stage': 'basic_connection', 'details': basic_test}
        
        # 3. Timeout testing
        print("\n3️⃣ Timeout Configuration Test:")
        timeout_results = await self.test_different_timeouts()
        for config, result in timeout_results.items():
            if result['success']:
                print(f"   ✅ {config}: {result['duration']:.2f}s")
            else:
                print(f"   ❌ {config}: {result['error']}")
        
        # 4. Concurrent requests test
        print("\n4️⃣ Concurrent Requests Test:")
        concurrent_results = await self.test_concurrent_requests(3)
        print(f"   📊 Success rate: {concurrent_results['success_rate']:.1f}% ({concurrent_results['successful']}/{concurrent_results['total_requests']})")
        
        # Overall assessment
        print("\n📈 OVERALL ASSESSMENT:")
        if basic_test['success'] and concurrent_results['success_rate'] > 80:
            print("   ✅ Anthropic API is working reliably")
            status = 'healthy'
        elif basic_test['success']:
            print("   ⚠️  Basic API works but reliability issues detected")
            status = 'partial'
        else:
            print("   ❌ Serious API connection issues")
            status = 'failed'
        
        return {
            'overall_status': status,
            'environment': env_results,
            'basic_test': basic_test,
            'timeout_tests': timeout_results,
            'concurrent_tests': concurrent_results
        }

async def main():
    """Run diagnostics"""
    diagnostics = AnthropicDiagnostics()
    results = await diagnostics.run_full_diagnostics()
    
    print(f"\n🎯 Final Status: {results['overall_status'].upper()}")
    
    if results['overall_status'] == 'failed':
        print("\n🔧 RECOMMENDED FIXES:")
        
        if 'environment' in results:
            env = results['environment']
            if not env['anthropic_installed']:
                print("   - Install anthropic: pip install anthropic")
            if not env['api_key_set']:
                print("   - Set ANTHROPIC_API_KEY environment variable")
            if env['api_key_format'] == 'invalid_format':
                print("   - Check API key format (should start with sk-ant-)")
        
        if 'basic_test' in results and not results['basic_test']['success']:
            error_type = results['basic_test'].get('error_type', '')
            if 'timeout' in error_type.lower():
                print("   - Increase timeout settings")
            elif 'connection' in error_type.lower():
                print("   - Check network connectivity")
                print("   - Verify firewall/proxy settings")
            elif 'auth' in error_type.lower():
                print("   - Verify API key validity")

if __name__ == "__main__":
    asyncio.run(main())
