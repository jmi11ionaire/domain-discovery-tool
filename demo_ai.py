"""
Demo script for AI-Enhanced B2B Publisher Discovery System
Demonstrates key features and AI capabilities
"""

import asyncio
import os
import json
from ai_enhanced_publisher_discovery import AIPublisherDiscovery

# Demo configuration
DEMO_DOMAINS = [
    'techcrunch.com',
    'forbes.com',
    'businessinsider.com',
    'hbr.org',
    'wired.com',
    'fastcompany.com',
    'arstechnica.com',
    'venturebeat.com'
]

async def demo_basic_ai_analysis():
    """Demonstrate basic AI analysis capabilities"""
    print("🚀 AI-Enhanced Publisher Discovery Demo")
    print("=" * 60)
    
    # Initialize the AI discovery system
    # Note: You'll need to set your API keys
    discovery = AIPublisherDiscovery(
        db_path="demo_ai_publishers.db",
        openai_api_key=os.getenv("OPENAI_API_KEY"),  # Set this in your environment
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")  # Optional alternative
    )
    
    try:
        print(f"\n📊 Analyzing {len(DEMO_DOMAINS)} domains with AI...")
        print("-" * 60)
        
        # Analyze domains with AI enhancement
        scores = await discovery.batch_analyze_ai(DEMO_DOMAINS[:4], max_concurrent=2)
        
        print(f"\n✅ Analysis Complete! Results for {len(scores)} domains:")
        print("=" * 80)
        
        # Display detailed results
        for i, score in enumerate(scores, 1):
            print(f"\n🌐 Domain #{i}: {score.domain}")
            print(f"   Overall Score: {score.overall_score:.1f}/100")
            print(f"   AI Content Quality: {score.content_quality_ai:.1f}/100")
            print(f"   Professional Tone: {score.professional_tone_score:.1f}/100")
            print(f"   B2B Relevance: {score.b2b_relevance_score:.1f}/100")
            print(f"   Brand Safety: {score.brand_safety_score:.1f}/100")
            print(f"   Fraud Risk: {score.fraud_risk_score:.1f}/100")
            print(f"   🤖 AI Recommendation: {score.recommendation}")
            print(f"   📝 AI Summary: {score.content_summary}")
            
            if score.risk_factors:
                print(f"   ⚠️  Risk Factors: {', '.join(score.risk_factors[:2])}")
            
            if score.opportunities:
                print(f"   💡 Opportunities: {', '.join(score.opportunities[:2])}")
            
            print(f"   🔒 SSL: {'✅' if score.ssl_enabled else '❌'}")
            print(f"   📞 Contact Page: {'✅' if score.has_contact_page else '❌'}")
        
        return scores
        
    finally:
        await discovery.close_session()

async def demo_ai_insights_report(discovery_instance=None):
    """Demonstrate AI insights and reporting capabilities"""
    if not discovery_instance:
        discovery_instance = AIPublisherDiscovery(
            db_path="demo_ai_publishers.db",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    print("\n📈 AI INSIGHTS REPORT")
    print("=" * 80)
    
    try:
        # Generate comprehensive insights
        insights = discovery_instance.get_ai_insights_report(limit=20)
        
        # Display analytics
        analytics = insights['analytics']
        print(f"\n📊 ANALYTICS SUMMARY:")
        print(f"   Total Domains Analyzed: {analytics['total_analyzed']}")
        print(f"   Average Overall Score: {analytics['average_overall_score']:.1f}/100")
        print(f"   Average AI Content Quality: {analytics['average_content_quality']:.1f}/100")
        print(f"   Average Professional Tone: {analytics['average_professional_tone']:.1f}/100")
        print(f"   Average Fraud Risk: {analytics['average_fraud_risk']:.1f}/100")
        
        # Display top topics
        if insights['topic_distribution']:
            print(f"\n🎯 TOP BUSINESS TOPICS DETECTED:")
            for topic, data in list(insights['topic_distribution'].items())[:5]:
                print(f"   {topic}: {data['count']} domains (avg score: {data['avg_score']:.1f})")
        
        # Display top publishers
        top_publishers = insights['top_publishers']
        if top_publishers:
            print(f"\n🏆 TOP {min(5, len(top_publishers))} APPROVED PUBLISHERS:")
            for i, pub in enumerate(top_publishers[:5], 1):
                print(f"   {i}. {pub['domain']} - Score: {pub['overall_score']:.1f}")
                print(f"      Recommendation: {pub['recommendation']}")
                print(f"      Summary: {pub['summary'][:100]}...")
    
    except Exception as e:
        print(f"⚠️  Error generating insights: {e}")

def demo_fallback_analysis():
    """Demonstrate fallback analysis when AI is unavailable"""
    print("\n🔄 FALLBACK ANALYSIS DEMO (No AI APIs)")
    print("=" * 60)
    
    # Initialize without API keys to test fallback
    discovery = AIPublisherDiscovery(db_path="demo_fallback.db")
    
    # Test fallback content analysis
    sample_content = """
    <html>
    <head><title>Business Technology News</title></head>
    <body>
    <h1>Enterprise Software Solutions</h1>
    <p>Latest news on cloud computing, SaaS platforms, and digital transformation 
    for business professionals. Our coverage includes strategy, leadership, and 
    technology innovation.</p>
    <p>Contact us at info@example.com</p>
    </body>
    </html>
    """
    
    fallback_result = discovery.fallback_content_analysis(sample_content)
    
    print("📋 Fallback Analysis Results:")
    print(f"   Content Quality: {fallback_result['content_quality']}/100")
    print(f"   B2B Relevance: {fallback_result['b2b_relevance']}/100")
    print(f"   Brand Safety: {fallback_result['brand_safety']}/100")
    print(f"   Recommendation: {fallback_result['recommendation']}")
    print(f"   Summary: {fallback_result['summary']}")

async def main():
    """Main demo function"""
    print("🤖 AI-Enhanced B2B Publisher Discovery System")
    print("=" * 70)
    print("This demo showcases the AI and LLM integration capabilities.")
    print("\n⚠️  NOTE: You need OpenAI or Anthropic API keys for full AI features.")
    print("Set environment variables: OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    print(f"\n🔑 API Key Status:")
    print(f"   OpenAI: {'✅ Available' if has_openai else '❌ Not set'}")
    print(f"   Anthropic: {'✅ Available' if has_anthropic else '❌ Not set'}")
    
    if has_openai or has_anthropic:
        print("\n🎯 Running FULL AI ANALYSIS...")
        scores = await demo_basic_ai_analysis()
        
        if scores:
            # Generate insights report
            discovery = AIPublisherDiscovery(
                db_path="demo_ai_publishers.db",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            await demo_ai_insights_report(discovery)
            await discovery.close_session()
    else:
        print("\n⚠️  No API keys found. Running fallback demo...")
        demo_fallback_analysis()
    
    print("\n" + "=" * 70)
    print("📚 For more information, see:")
    print("   - ai_integration_overview.md: Complete AI integration guide")
    print("   - ai_enhanced_publisher_discovery.py: Full implementation")
    print("   - requirements_ai.txt: Required dependencies")
    print("\n🚀 Ready to scale your B2B publisher network with AI!")

if __name__ == "__main__":
    asyncio.run(main())
