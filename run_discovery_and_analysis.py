#!/usr/bin/env python3
"""
Integrated Discovery and Analysis Runner
Runs optimized domain discovery and provides immediate analysis
"""

import asyncio
import sys
import os
from optimized_domain_scanner import run_optimized_discovery_pipeline

def run_analysis_tools():
    """Run analysis tools on the results"""
    print("\n🔍 RUNNING POST-DISCOVERY ANALYSIS")
    print("=" * 50)
    
    try:
        # Import and run comprehensive analysis
        from analysis_tools.comprehensive_rejection_analysis import analyze_complete_rejection_patterns
        
        print("📊 Running comprehensive rejection analysis...")
        analyze_complete_rejection_patterns()
        
    except ImportError as e:
        print(f"⚠️  Analysis tools not available: {e}")
        print("   Run: python analysis_tools/comprehensive_rejection_analysis.py")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

async def main():
    """Run discovery pipeline followed by analysis"""
    
    # Parse command line arguments
    target_count = 30
    if len(sys.argv) > 1:
        try:
            target_count = int(sys.argv[1])
        except ValueError:
            print("Usage: python run_discovery_and_analysis.py [target_count]")
            return
    
    print("🚀 INTEGRATED DISCOVERY & ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Target domains: {target_count}")
    print(f"Database: optimized_domain_discovery.db")
    print()
    
    # Phase 1: Run optimized discovery
    try:
        print("📡 PHASE 1: DOMAIN DISCOVERY")
        print("-" * 30)
        
        result = await run_optimized_discovery_pipeline(target_count)
        
        print(f"\n✅ Discovery Complete:")
        print(f"   Session: {result['session_id']}")
        print(f"   Discovered: {result['discovered']} domains")
        print(f"   Validated: {result['validated']} domains ({result['discovery_rate']:.1f}%)")
        print(f"   Approved: {result['approved']} domains ({result['approval_rate']:.1f}%)")
        print(f"   Runtime: {result['runtime_minutes']:.2f} minutes")
        
    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        return
    
    # Phase 2: Run analysis
    print("\n📈 PHASE 2: ANALYSIS & INSIGHTS")
    print("-" * 30)
    
    run_analysis_tools()
    
    # Summary
    print(f"\n🎉 PIPELINE COMPLETE")
    print(f"   New approved domains: {result['approved']}")
    print(f"   Check analysis/ directory for detailed analysis")
    
    # Quick recommendations
    if result['discovery_rate'] < 15:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   🔍 Discovery rate ({result['discovery_rate']:.1f}%) is low - improve domain quality")
        
    if result['approval_rate'] > 70:
        print(f"   ✅ High approval rate ({result['approval_rate']:.1f}%) - system working well")
    elif result['approval_rate'] < 30:
        print(f"   📉 Low approval rate ({result['approval_rate']:.1f}%) - consider threshold adjustment")

if __name__ == "__main__":
    print("🚀 Integrated Domain Discovery & Analysis")
    print("Production-ready pipeline with immediate insights")
    print()
    
    asyncio.run(main())
