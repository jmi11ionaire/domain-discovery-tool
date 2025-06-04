#!/usr/bin/env python3
"""
Comprehensive Rejection Analysis
Analyzes both validation failures and scoring issues
"""

import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np

def analyze_complete_rejection_patterns():
    """Complete analysis of all rejection patterns"""
    
    print("🔍 COMPREHENSIVE REJECTION ANALYSIS")
    print("=" * 50)
    
    conn = sqlite3.connect('fixed_domain_discovery.db')
    
    # Get complete data
    query = """
        SELECT domain, current_status, current_score, strategy_used, 
               first_analyzed_at, discovery_source
        FROM domains_analyzed
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Categorize rejection types
    validation_failures = df[df['current_score'] <= 1]  # Score 0-1 = validation failures
    analyzed_domains = df[df['current_score'] > 1]      # Score >1 = actually analyzed
    
    print(f"\n📊 OVERALL BREAKDOWN")
    print(f"   Total domains: {len(df):,}")
    print(f"   Validation failures: {len(validation_failures):,} ({len(validation_failures)/len(df)*100:.1f}%)")
    print(f"   Actually analyzed: {len(analyzed_domains):,} ({len(analyzed_domains)/len(df)*100:.1f}%)")
    
    # Validation failure analysis
    print(f"\n❌ VALIDATION FAILURES ({len(validation_failures):,} domains)")
    print(f"   These domains scored ≤1, indicating:")
    print(f"   - Domain not reachable (DNS/HTTP failures)")
    print(f"   - No content found")
    print(f"   - Basic validation failed")
    print(f"   Average score: {validation_failures['current_score'].mean():.2f}")
    
    # Analysis of domains that were actually scored
    if len(analyzed_domains) > 0:
        approved_analyzed = analyzed_domains[analyzed_domains['current_status'] == 'approved']
        rejected_analyzed = analyzed_domains[analyzed_domains['current_status'] == 'rejected']
        
        print(f"\n✅ DOMAINS THAT PASSED VALIDATION ({len(analyzed_domains):,} domains)")
        print(f"   Approved: {len(approved_analyzed):,} ({len(approved_analyzed)/len(analyzed_domains)*100:.1f}%)")
        print(f"   Rejected: {len(rejected_analyzed):,} ({len(rejected_analyzed)/len(analyzed_domains)*100:.1f}%)")
        print(f"   Approved avg score: {approved_analyzed['current_score'].mean():.1f}")
        print(f"   Rejected avg score: {rejected_analyzed['current_score'].mean():.1f}")
        
        # Borderline analysis for actually analyzed domains
        borderline = rejected_analyzed[
            (rejected_analyzed['current_score'] >= 30) & 
            (rejected_analyzed['current_score'] < 40)
        ]
        
        print(f"\n🎯 BORDERLINE DOMAINS (30-39 scores)")
        print(f"   Count: {len(borderline):,}")
        print(f"   These are quality domains close to approval")
        
        if len(borderline) > 0:
            print(f"   Sample borderline domains:")
            for _, domain in borderline.head(5).iterrows():
                print(f"     {domain['domain']} ({domain['current_score']:.1f})")
    
    # Create visualization
    create_comprehensive_chart(df, validation_failures, analyzed_domains)
    
    # Recommendations
    print(f"\n💡 KEY INSIGHTS & RECOMMENDATIONS")
    
    validation_rate = len(validation_failures) / len(df) * 100
    if validation_rate > 90:
        print(f"   🚨 PRIMARY ISSUE: {validation_rate:.1f}% domains fail basic validation")
        print(f"   📡 Focus on improving domain discovery quality")
        print(f"   🔍 Many discovered domains may not exist or be unreachable")
    
    if len(analyzed_domains) > 0:
        approval_rate = len(approved_analyzed) / len(analyzed_domains) * 100
        if approval_rate > 50:
            print(f"   ✅ GOOD: {approval_rate:.1f}% of reachable domains get approved")
            print(f"   📈 Scoring algorithm works well for valid domains")
        
        if len(borderline) > 10:
            print(f"   🎯 OPPORTUNITY: {len(borderline)} borderline domains (30-39 scores)")
            print(f"   📉 Consider threshold adjustment from 40 to 35")
            print(f"   🔄 Or improve scoring for these quality domains")
    
    return {
        'total_domains': len(df),
        'validation_failures': len(validation_failures),
        'analyzed_domains': len(analyzed_domains),
        'approved': len(approved_analyzed) if len(analyzed_domains) > 0 else 0,
        'borderline': len(borderline) if len(analyzed_domains) > 0 else 0
    }

def create_comprehensive_chart(df, validation_failures, analyzed_domains):
    """Create comprehensive visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Domain Discovery Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall breakdown pie chart
    sizes = [len(validation_failures), len(analyzed_domains)]
    labels = [f'Validation Failures\n({len(validation_failures):,})', 
              f'Successfully Analyzed\n({len(analyzed_domains):,})']
    colors = ['lightcoral', 'lightgreen']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Domain Processing Breakdown')
    
    # 2. Score distribution for analyzed domains only
    if len(analyzed_domains) > 0:
        approved_scores = analyzed_domains[analyzed_domains['current_status'] == 'approved']['current_score']
        rejected_scores = analyzed_domains[analyzed_domains['current_status'] == 'rejected']['current_score']
        
        ax2.hist(rejected_scores, bins=15, alpha=0.7, label=f'Rejected ({len(rejected_scores)})', color='red')
        ax2.hist(approved_scores, bins=15, alpha=0.7, label=f'Approved ({len(approved_scores)})', color='green')
        ax2.axvline(x=40, color='orange', linestyle='--', linewidth=2, label='Threshold (40)')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Number of Domains')
        ax2.set_title('Score Distribution (Analyzed Domains Only)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Validation failure rate over time (if we have timestamps)
    try:
        df['date'] = pd.to_datetime(df['first_analyzed_at']).dt.date
        daily_stats = df.groupby('date').agg({
            'domain': 'count',
            'current_score': lambda x: (x <= 1).sum()
        }).rename(columns={'domain': 'total', 'current_score': 'failures'})
        
        daily_stats['failure_rate'] = daily_stats['failures'] / daily_stats['total'] * 100
        
        ax3.plot(daily_stats.index, daily_stats['failure_rate'], 'o-', color='red')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Validation Failure Rate (%)')
        ax3.set_title('Validation Failure Rate Over Time')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    except:
        ax3.text(0.5, 0.5, 'Timeline data unavailable', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Timeline Analysis')
    
    # 4. Threshold sensitivity for analyzed domains
    if len(analyzed_domains) > 0:
        thresholds = range(25, 65, 5)
        approval_counts = []
        
        for threshold in thresholds:
            would_approve = len(analyzed_domains[analyzed_domains['current_score'] >= threshold])
            approval_counts.append(would_approve)
        
        ax4.plot(thresholds, approval_counts, 'b-o', linewidth=2, markersize=8)
        ax4.axvline(x=40, color='orange', linestyle='--', linewidth=2, label='Current Threshold')
        ax4.set_xlabel('Threshold Score')
        ax4.set_ylabel('Number of Approved Domains')
        ax4.set_title('Threshold Sensitivity (Analyzed Domains)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chart_file = f'analysis/comprehensive_analysis_{timestamp}.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Comprehensive chart saved: {chart_file}")
    return chart_file

if __name__ == "__main__":
    analyze_complete_rejection_patterns()
