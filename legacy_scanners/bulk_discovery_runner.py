#!/usr/bin/env python3
"""
Bulk Discovery Runner
Easy-to-use runner for the API-Enhanced Bulk Scanner
"""

import asyncio
import argparse
import logging
import sys
from api_enhanced_bulk_scanner import run_bulk_discovery_pipeline

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/bulk_discovery.log', mode='a')
        ]
    )

async def main():
    """Main entry point for bulk discovery"""
    parser = argparse.ArgumentParser(
        description='API-Enhanced Bulk Domain Discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run with 1000 domains
  python bulk_discovery_runner.py --count 1000

  # Full scale run with 50k domains
  python bulk_discovery_runner.py --count 50000

  # Specific API source
  python bulk_discovery_runner.py --count 10000 --source semrush

  # Enable verbose logging
  python bulk_discovery_runner.py --count 5000 --verbose

Prerequisites:
  1. Configure API keys in api_config.yaml
  2. Set enabled: true for available APIs
  3. Ensure PyYAML is installed: pip install PyYAML
        """
    )
    
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=1000,
        help='Number of domains to discover and analyze (default: 1000)'
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='mixed',
        choices=['mixed', 'common_crawl', 'semrush', 'ahrefs', 'security_trails'],
        help='API source for domain discovery (default: mixed)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run a small test with 50 domains'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Override count for test mode
    if args.test:
        args.count = 50
        print("🧪 Test mode: Running with 50 domains")
    
    print("🚀 API-Enhanced Bulk Domain Discovery Runner")
    print("=" * 60)
    print(f"Target domains: {args.count:,}")
    print(f"API source: {args.source}")
    print(f"Verbose logging: {args.verbose}")
    print()
    
    # Check if config file exists
    try:
        with open('api_config.yaml', 'r') as f:
            pass
    except FileNotFoundError:
        print("❌ ERROR: api_config.yaml not found!")
        print("Please ensure api_config.yaml exists and is configured.")
        return 1
    
    try:
        # Run the bulk discovery pipeline
        result = await run_bulk_discovery_pipeline(
            target_count=args.count,
            api_source=args.source
        )
        
        if 'error' in result:
            print(f"❌ Pipeline failed: {result['error']}")
            return 1
        
        print("\n🎉 BULK DISCOVERY COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Batch ID: {result.get('batch_id', 'N/A')}")
        print(f"Total discovered: {result.get('total_discovered', 0):,} domains")
        print(f"Total validated: {result.get('total_validated', 0):,} domains")
        print(f"Total analyzed: {result.get('total_analyzed', 0):,} domains")
        print(f"Total approved: {result.get('total_approved', 0):,} domains")
        print(f"Approval rate: {result.get('approval_rate', 0):.1f}%")
        print(f"Processing rate: {result.get('domains_per_minute', 0):.1f} domains/minute")
        print(f"Analysis time: {result.get('analysis_time_minutes', 0):.1f} minutes")
        
        # Success/failure summary
        if result.get('total_approved', 0) > 0:
            print(f"\n✅ SUCCESS: Found {result['total_approved']} new high-quality domains!")
        else:
            print(f"\n⚠️  No domains met approval criteria. Consider:")
            print("   - Adjusting scoring thresholds")
            print("   - Trying different API sources")
            print("   - Checking API configuration")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Discovery interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logging.exception("Bulk discovery failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
