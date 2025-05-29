#!/usr/bin/env python3
"""
Demo script for B2B Publisher Discovery System
Shows how to use the system to discover and vet publishers
"""

import asyncio
import pandas as pd
from publisher_discovery import PublisherDiscovery, PublisherScore

async def run_demo():
    """Demonstrate the publisher discovery system"""
    print("🚀 B2B Publisher Discovery System Demo")
    print("=" * 50)
    
    # Initialize the system
    discovery = PublisherDiscovery()
    
    try:
        # Phase 1: Discovery
        print("\n📊 Phase 1: Publisher Discovery")
        print("-" * 30)
        
        # Discover from various sources
        competitor_domains = discovery.discover_from_competitors(['salesforce.com'])
        print(f"✅ Discovered {len(competitor_domains)} domains from competitor analysis")
        
        industry_domains = discovery.discover_from_industry_directories()
        print(f"✅ Discovered {len(industry_domains)} domains from industry directories")
        
        # Additional high-quality test domains
        test_domains = [
            'techcrunch.com',      # Tech/Business news
            'forbes.com',          # Business authority
            'businessinsider.com', # Business content
            'hbr.org',            # Harvard Business Review
            'wired.com',          # Technology
            'arstechnica.com',    # Tech authority
            'fastcompany.com',    # Business innovation
            'inc.com',            # Entrepreneurship
            'venturebeat.com',    # Tech/Business
            'zdnet.com'           # Technology news
        ]
        
        print(f"✅ Added {len(test_domains)} high-quality test domains")
        
        # Phase 2: Analysis
        print(f"\n🔍 Phase 2: Domain Analysis")
        print("-" * 30)
        print(f"Analyzing {len(test_domains)} domains (this may take a moment)...")
        
        # Analyze domains concurrently
        scores = await discovery.batch_analyze(test_domains, max_concurrent=5)
        successful_analyses = [s for s in scores if s.overall_score > 0]
        
        print(f"✅ Successfully analyzed {len(successful_analyses)} domains")
        
        # Phase 3: Results
        print(f"\n📈 Phase 3: Results & Scoring")
        print("-" * 30)
        
        # Display individual scores
        print("\nDetailed Scoring Results:")
        print(f"{'Domain':<20} {'Overall':<8} {'Safety':<8} {'B2B':<8} {'Authority':<8} {'Status'}")
        print("-" * 80)
        
        for score in sorted(successful_analyses, key=lambda x: x.overall_score, reverse=True):
            status = "✅ APPROVED" if score.overall_score >= 70 else "❌ REJECTED"
            print(f"{score.domain:<20} {score.overall_score:<8.1f} {score.brand_safety_score:<8.1f} "
                  f"{score.b2b_relevance_score:<8.1f} {score.authority_score:<8.1f} {status}")
        
        # Get approved publishers from database
        approved_publishers = discovery.get_top_publishers(limit=20)
        
        print(f"\n🎯 Approved Publishers Summary")
        print("-" * 30)
        print(f"Total approved publishers: {len(approved_publishers)}")
        
        if len(approved_publishers) > 0:
            avg_score = approved_publishers['overall_score'].mean()
            print(f"Average overall score: {avg_score:.1f}")
            print(f"Score range: {approved_publishers['overall_score'].min():.1f} - {approved_publishers['overall_score'].max():.1f}")
            
            # Show top 5
            print(f"\nTop 5 Publishers:")
            for idx, row in approved_publishers.head().iterrows():
                print(f"  {idx+1}. {row['domain']} (Score: {row['overall_score']:.1f})")
        
        # Phase 4: Quality Metrics
        print(f"\n📊 Phase 4: Quality Metrics")
        print("-" * 30)
        
        total_analyzed = len(successful_analyses)
        approved_count = len([s for s in successful_analyses if s.overall_score >= 70])
        approval_rate = (approved_count / total_analyzed * 100) if total_analyzed > 0 else 0
        
        print(f"Approval rate: {approval_rate:.1f}% ({approved_count}/{total_analyzed})")
        
        # Safety metrics
        high_safety = len([s for s in successful_analyses if s.brand_safety_score >= 85])
        safety_rate = (high_safety / total_analyzed * 100) if total_analyzed > 0 else 0
        print(f"Brand safety compliance: {safety_rate:.1f}% ({high_safety}/{total_analyzed})")
        
        # B2B relevance
        b2b_relevant = len([s for s in successful_analyses if s.b2b_relevance_score >= 50])
        b2b_rate = (b2b_relevant / total_analyzed * 100) if total_analyzed > 0 else 0
        print(f"B2B relevance rate: {b2b_rate:.1f}% ({b2b_relevant}/{total_analyzed})")
        
        # Technical quality
        ssl_enabled = len([s for s in successful_analyses if s.ssl_enabled])
        ssl_rate = (ssl_enabled / total_analyzed * 100) if total_analyzed > 0 else 0
        print(f"SSL compliance: {ssl_rate:.1f}% ({ssl_enabled}/{total_analyzed})")
        
        # Phase 5: Recommendations
        print(f"\n🎯 Phase 5: Recommendations")
        print("-" * 30)
        
        if approval_rate >= 70:
            print("✅ EXCELLENT: High approval rate indicates good discovery sources")
        elif approval_rate >= 50:
            print("⚠️  GOOD: Moderate approval rate, consider refining discovery criteria")
        else:
            print("❌ NEEDS IMPROVEMENT: Low approval rate, review discovery sources")
        
        print(f"\nTo reach 10,000 publishers from current {len(approved_publishers)}:")
        remaining = 10000 - len(approved_publishers)
        if approval_rate > 0:
            domains_needed = int(remaining / (approval_rate / 100))
            print(f"- Need to analyze approximately {domains_needed:,} more domains")
            print(f"- At current approval rate ({approval_rate:.1f}%), this is achievable")
        
        print(f"\nNext steps:")
        print(f"1. Integrate with traffic APIs (SimilarWeb, SEMrush) for better authority scoring")
        print(f"2. Add competitive intelligence for B2B ad placement discovery") 
        print(f"3. Implement continuous monitoring for approved publishers")
        print(f"4. Set up automated daily discovery pipeline")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        
    finally:
        await discovery.close_session()
        print(f"\n🎉 Demo completed! Check publishers.db for stored results.")

if __name__ == "__main__":
    asyncio.run(run_demo())
