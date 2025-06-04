#!/usr/bin/env python3
"""
Improved Domain Analysis with Charts and Bug Fixes
Corrects the rejection reason analysis and creates score distribution visualization
"""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import seaborn as sns

class ImprovedDomainAnalysis:
    """Fixed domain analysis with proper rejection tracking and visualization"""
    
    def __init__(self, db_path: str = "fixed_domain_discovery.db"):
        self.db_path = db_path
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def analyze_score_distribution(self) -> Dict:
        """Comprehensive score distribution analysis"""
        print("📊 Analyzing score distribution...")
        
        conn = self.get_connection()
        
        # Get all scores with status
        query = "SELECT current_score, current_status FROM domains_analyzed WHERE current_score > 0"
        df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        # Basic statistics
        approved_scores = df[df['current_status'] == 'approved']['current_score']
        rejected_scores = df[df['current_status'] == 'rejected']['current_score']
        
        analysis = {
            'total_domains': len(df),
            'approved_count': len(approved_scores),
            'rejected_count': len(rejected_scores),
            'approved_stats': {
                'mean': approved_scores.mean(),
                'median': approved_scores.median(),
                'min': approved_scores.min(),
                'max': approved_scores.max(),
                'std': approved_scores.std()
            },
            'rejected_stats': {
                'mean': rejected_scores.mean(),
                'median': rejected_scores.median(),
                'min': rejected_scores.min(),
                'max': rejected_scores.max(),
                'std': rejected_scores.std()
            }
        }
        
        # Score range distribution
        score_bins = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
        range_distribution = {}
        
        for min_score, max_score in score_bins:
            range_key = f"{min_score}-{max_score}"
            range_data = df[(df['current_score'] >= min_score) & (df['current_score'] < max_score)]
            range_distribution[range_key] = {
                'total': len(range_data),
                'approved': len(range_data[range_data['current_status'] == 'approved']),
                'rejected': len(range_data[range_data['current_status'] == 'rejected'])
            }
        
        analysis['score_ranges'] = range_distribution
        analysis['dataframe'] = df  # For plotting
        
        return analysis
    
    def create_score_distribution_chart(self, analysis: Dict):
        """Create comprehensive score distribution visualization"""
        print("📈 Creating score distribution charts...")
        
        df = analysis['dataframe']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Domain Discovery Score Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram of all scores
        approved_scores = df[df['current_status'] == 'approved']['current_score']
        rejected_scores = df[df['current_status'] == 'rejected']['current_score']
        
        ax1.hist(rejected_scores, bins=20, alpha=0.7, label=f'Rejected ({len(rejected_scores)})', color='red')
        ax1.hist(approved_scores, bins=20, alpha=0.7, label=f'Approved ({len(approved_scores)})', color='green')
        ax1.axvline(x=40, color='orange', linestyle='--', linewidth=2, label='Current Threshold (40)')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Number of Domains')
        ax1.set_title('Score Distribution by Status')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot comparison
        data_for_box = [approved_scores.values, rejected_scores.values]
        labels = ['Approved', 'Rejected']
        box_plot = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        ax2.axhline(y=40, color='orange', linestyle='--', linewidth=2, label='Threshold (40)')
        ax2.set_ylabel('Score')
        ax2.set_title('Score Distribution Summary')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Score range breakdown
        ranges = []
        total_counts = []
        approved_counts = []
        rejected_counts = []
        
        for range_key, data in analysis['score_ranges'].items():
            ranges.append(range_key)
            total_counts.append(data['total'])
            approved_counts.append(data['approved'])
            rejected_counts.append(data['rejected'])
        
        x = np.arange(len(ranges))
        width = 0.35
        
        ax3.bar(x - width/2, approved_counts, width, label='Approved', color='green', alpha=0.7)
        ax3.bar(x + width/2, rejected_counts, width, label='Rejected', color='red', alpha=0.7)
        ax3.set_xlabel('Score Range')
        ax3.set_ylabel('Number of Domains')
        ax3.set_title('Domains by Score Range')
        ax3.set_xticks(x)
        ax3.set_xticklabels(ranges, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Threshold sensitivity analysis
        thresholds = range(25, 65, 5)
        approval_rates = []
        would_approve_counts = []
        
        for threshold in thresholds:
            would_approve = len(df[df['current_score'] >= threshold])
            approval_rate = (would_approve / len(df)) * 100
            approval_rates.append(approval_rate)
            would_approve_counts.append(would_approve)
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(thresholds, approval_rates, 'b-o', label='Approval Rate %')
        line2 = ax4_twin.plot(thresholds, would_approve_counts, 'r-s', label='Total Approved')
        
        ax4.axvline(x=40, color='orange', linestyle='--', linewidth=2, label='Current Threshold')
        ax4.set_xlabel('Threshold Score')
        ax4.set_ylabel('Approval Rate (%)', color='b')
        ax4_twin.set_ylabel('Number of Domains', color='r')
        ax4.set_title('Threshold Sensitivity Analysis')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        chart_file = f'analysis/score_distribution_analysis_{timestamp}.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Chart saved: {chart_file}")
        return chart_file
    
    def investigate_borderline_domains(self) -> Dict:
        """Deep dive into borderline domains to understand rejection patterns"""
        print("🔎 Investigating borderline domains (30-45 range)...")
        
        conn = self.get_connection()
        
        # Get borderline domains with more details
        query = """
            SELECT domain, current_score, current_status
            FROM domains_analyzed 
            WHERE current_score BETWEEN 30 AND 45
            ORDER BY current_score DESC
        """
        
        borderline_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Analyze the borderline domains
        analysis = {
            'total_borderline': len(borderline_df),
            'borderline_approved': len(borderline_df[borderline_df['current_status'] == 'approved']),
            'borderline_rejected': len(borderline_df[borderline_df['current_status'] == 'rejected']),
            'threshold_impact': {}
        }
        
        # Test different thresholds
        for threshold in [30, 32, 35, 37, 40]:
            would_approve = len(borderline_df[borderline_df['current_score'] >= threshold])
            analysis['threshold_impact'][threshold] = would_approve
        
        # Get specific examples
        high_rejected = borderline_df[
            (borderline_df['current_score'] >= 35) & 
            (borderline_df['current_status'] == 'rejected')
        ].head(10)
        
        analysis['high_scoring_rejected'] = high_rejected.to_dict('records')
        
        return analysis
    
    def generate_recommendations(self, score_analysis: Dict, borderline_analysis: Dict) -> List[str]:
        """Generate specific recommendations based on analysis"""
        recommendations = []
        
        # Approval rate analysis
        approval_rate = (score_analysis['approved_count'] / score_analysis['total_domains']) * 100
        
        if approval_rate < 5:
            recommendations.append("🚨 CRITICAL: <5% approval rate is extremely low for quality discovery")
        elif approval_rate < 10:
            recommendations.append("⚠️ ALERT: <10% approval rate suggests overly strict criteria")
        
        # Score distribution analysis
        rejected_mean = score_analysis['rejected_stats']['mean']
        approved_mean = score_analysis['approved_stats']['mean']
        
        if rejected_mean > 30:
            recommendations.append(f"📊 Rejected domains average {rejected_mean:.1f} - many may be quality sites")
        
        # Threshold analysis
        if borderline_analysis['borderline_rejected'] > 20:
            recommendations.append(f"🎯 {borderline_analysis['borderline_rejected']} domains scoring 30-45 rejected - consider threshold adjustment")
        
        # Specific threshold recommendations
        if borderline_analysis['threshold_impact'].get(35, 0) > borderline_analysis['threshold_impact'].get(40, 0) * 1.5:
            diff = borderline_analysis['threshold_impact'][35] - borderline_analysis['threshold_impact'][40]
            recommendations.append(f"📈 Lowering threshold to 35 would approve {diff} additional domains")
        
        return recommendations
    
    def run_improved_analysis(self):
        """Run complete improved analysis"""
        print("🚀 IMPROVED DOMAIN ANALYSIS SUITE")
        print("=" * 50)
        
        try:
            # 1. Score distribution analysis
            score_analysis = self.analyze_score_distribution()
            
            # 2. Create visualization
            chart_file = self.create_score_distribution_chart(score_analysis)
            
            # 3. Investigate borderline cases
            borderline_analysis = self.investigate_borderline_domains()
            
            # 4. Generate recommendations
            recommendations = self.generate_recommendations(score_analysis, borderline_analysis)
            
            # 5. Print summary
            self.print_detailed_summary(score_analysis, borderline_analysis, recommendations, chart_file)
            
            return {
                'score_analysis': score_analysis,
                'borderline_analysis': borderline_analysis,
                'recommendations': recommendations,
                'chart_file': chart_file
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def print_detailed_summary(self, score_analysis: Dict, borderline_analysis: Dict, 
                             recommendations: List[str], chart_file: str):
        """Print comprehensive summary"""
        print(f"\n📊 SCORE DISTRIBUTION ANALYSIS")
        print(f"   Total domains: {score_analysis['total_domains']:,}")
        print(f"   Approved: {score_analysis['approved_count']:,} ({score_analysis['approved_count']/score_analysis['total_domains']*100:.1f}%)")
        print(f"   Rejected: {score_analysis['rejected_count']:,} ({score_analysis['rejected_count']/score_analysis['total_domains']*100:.1f}%)")
        
        print(f"\n📈 SCORE STATISTICS")
        print(f"   Approved domains: {score_analysis['approved_stats']['mean']:.1f} ± {score_analysis['approved_stats']['std']:.1f}")
        print(f"   Rejected domains: {score_analysis['rejected_stats']['mean']:.1f} ± {score_analysis['rejected_stats']['std']:.1f}")
        print(f"   Range gap: {score_analysis['approved_stats']['min']:.1f} (min approved) vs {score_analysis['rejected_stats']['max']:.1f} (max rejected)")
        
        print(f"\n🎯 BORDERLINE ANALYSIS (30-45 score range)")
        print(f"   Total borderline: {borderline_analysis['total_borderline']:,}")
        print(f"   Currently rejected: {borderline_analysis['borderline_rejected']:,}")
        print(f"   If threshold was 35: +{borderline_analysis['threshold_impact'][35] - borderline_analysis['threshold_impact'][40]} approvals")
        
        print(f"\n💡 KEY RECOMMENDATIONS")
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"\n📊 VISUALIZATION")
        print(f"   Chart created: {chart_file}")
        print(f"   View with: open {chart_file}")

def main():
    """Run improved analysis"""
    analyzer = ImprovedDomainAnalysis()
    analyzer.run_improved_analysis()

if __name__ == "__main__":
    main()
