#!/usr/bin/env python3
"""
Domain Analysis Suite
Comprehensive analysis of domain discovery rejection patterns and scoring optimization
"""

import sqlite3
import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

class DomainAnalysisSuite:
    """Comprehensive analysis of domain discovery patterns"""
    
    def __init__(self, db_path: str = "fixed_domain_discovery.db"):
        self.db_path = db_path
        self.results = {}
        
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def analyze_rejection_patterns(self) -> Dict:
        """Analyze why domains are being rejected"""
        print("🔍 Analyzing rejection patterns...")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check what columns exist in the current database
        cursor.execute("PRAGMA table_info(domains_analyzed)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Get all analyzed domains with basic fields
        cursor.execute('''
            SELECT domain, current_status, current_score, strategy_used
            FROM domains_analyzed
        ''')
        
        domains = cursor.fetchall()
        conn.close()
        
        # Analyze patterns
        total_domains = len(domains)
        approved_count = sum(1 for d in domains if d[1] == 'approved')
        rejected_count = total_domains - approved_count
        
        # Basic analysis with current schema
        score_ranges = defaultdict(int)
        strategy_breakdown = defaultdict(int)
        
        for domain_data in domains:
            domain, status, score, strategy = domain_data
            
            # Score ranges
            if score is None or score < 10:
                score_ranges['0-9'] += 1
            elif score < 20:
                score_ranges['10-19'] += 1
            elif score < 30:
                score_ranges['20-29'] += 1
            elif score < 40:
                score_ranges['30-39'] += 1
            elif score < 50:
                score_ranges['40-49'] += 1
            else:
                score_ranges['50+'] += 1
            
            # Strategy breakdown
            strategy_breakdown[strategy or 'unknown'] += 1
        
        analysis = {
            'total_domains': total_domains,
            'approved_count': approved_count,
            'rejected_count': rejected_count,
            'approval_rate': (approved_count / total_domains * 100) if total_domains > 0 else 0,
            'rejection_reasons': dict(strategy_breakdown),  # Using strategy as proxy for now
            'score_distribution': dict(score_ranges),
            'content_score_distribution': {},  # Not available in current schema
            'available_columns': columns
        }
        
        return analysis
    
    def analyze_scoring_components(self) -> Dict:
        """Analyze which scoring components correlate with approval (simplified for current schema)"""
        print("📊 Analyzing scoring component impact...")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT current_status, current_score, strategy_used
            FROM domains_analyzed
            WHERE current_score > 0
        ''')
        
        domains = cursor.fetchall()
        conn.close()
        
        approved_domains = [d for d in domains if d[0] == 'approved']
        rejected_domains = [d for d in domains if d[0] == 'rejected']
        
        def calculate_basic_stats(domain_list):
            if not domain_list:
                return {}
            
            return {
                'avg_total_score': sum(d[1] for d in domain_list) / len(domain_list),
                'count': len(domain_list),
                'min_score': min(d[1] for d in domain_list),
                'max_score': max(d[1] for d in domain_list)
            }
        
        analysis = {
            'approved_stats': calculate_basic_stats(approved_domains),
            'rejected_stats': calculate_basic_stats(rejected_domains),
            'total_approved': len(approved_domains),
            'total_rejected': len(rejected_domains),
            'note': 'Limited analysis due to simplified database schema'
        }
        
        return analysis
    
    def analyze_threshold_sensitivity(self) -> Dict:
        """Analyze how different thresholds would affect approval rates"""
        print("⚡ Analyzing threshold sensitivity...")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT current_score FROM domains_analyzed WHERE current_score > 0')
        scores = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        thresholds = [25, 30, 35, 40, 45, 50, 55, 60]
        threshold_analysis = {}
        
        for threshold in thresholds:
            would_approve = sum(1 for score in scores if score >= threshold)
            approval_rate = (would_approve / len(scores) * 100) if scores else 0
            threshold_analysis[threshold] = {
                'would_approve': would_approve,
                'approval_rate': approval_rate
            }
        
        return threshold_analysis
    
    def get_borderline_domains(self, min_score: int = 25, max_score: int = 39) -> List[Dict]:
        """Get domains that scored close to approval threshold for manual review"""
        print(f"🔎 Finding borderline domains (scores {min_score}-{max_score})...")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT domain, current_score, strategy_used
            FROM domains_analyzed 
            WHERE current_status = 'rejected' 
            AND current_score BETWEEN ? AND ?
            ORDER BY current_score DESC
        ''', (min_score, max_score))
        
        domains = cursor.fetchall()
        conn.close()
        
        borderline_domains = []
        for domain_data in domains:
            domain, score, strategy = domain_data
            
            borderline_domains.append({
                'domain': domain,
                'score': score,
                'strategy_used': strategy,
                'gap_to_approval': 40 - score,  # Current threshold is 40
                'note': 'Limited data due to simplified schema'
            })
        
        return borderline_domains
    
    def generate_optimization_recommendations(self, analysis_data: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        approval_rate = analysis_data['rejection_patterns']['approval_rate']
        
        if approval_rate < 10:
            recommendations.append("🚨 CRITICAL: <10% approval rate suggests overly strict criteria")
            recommendations.append("📉 Consider lowering approval threshold from 40 to 30-35")
        elif approval_rate < 20:
            recommendations.append("⚠️  LOW: <20% approval rate indicates strict filtering")
            recommendations.append("📊 Analyze borderline domains (30-39 scores) for quality")
        
        # Component-specific recommendations
        approved_stats = analysis_data['scoring_components']['approved_stats']
        rejected_stats = analysis_data['scoring_components']['rejected_stats']
        
        if approved_stats.get('has_ads_txt_rate', 0) > 80:
            recommendations.append("💡 ads.txt strongly correlates with approval - consider higher weighting")
        
        if approved_stats.get('avg_content_score', 0) - rejected_stats.get('avg_content_score', 0) > 15:
            recommendations.append("💡 Content score is key differentiator - maintain current weighting")
        
        if approved_stats.get('avg_b2b_relevance', 0) > 50:
            recommendations.append("💡 B2B relevance important - consider keyword expansion")
        
        # Threshold recommendations
        threshold_data = analysis_data['threshold_sensitivity']
        if threshold_data.get(35, {}).get('approval_rate', 0) > 25:
            recommendations.append("📈 Lowering threshold to 35 would increase approval rate significantly")
        
        return recommendations
    
    def export_analysis_report(self, analysis_data: Dict):
        """Export comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # HTML Report
        html_file = f'analysis/domain_analysis_report_{timestamp}.html'
        self.generate_html_report(analysis_data, html_file)
        
        # CSV Export of borderline domains
        csv_file = f'analysis/borderline_domains_{timestamp}.csv'
        self.export_borderline_csv(analysis_data['borderline_domains'], csv_file)
        
        print(f"\n📁 ANALYSIS REPORTS GENERATED:")
        print(f"   📊 HTML Report: {html_file}")
        print(f"   📋 Borderline Domains: {csv_file}")
    
    def generate_html_report(self, analysis_data: Dict, filename: str):
        """Generate HTML analysis report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Domain Discovery Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .alert {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }}
        .success {{ background: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .recommendation {{ background: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>🔍 Domain Discovery Analysis Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>📊 Overall Performance</h2>
    <div class="metric">
        <h3>Key Metrics</h3>
        <p><strong>Total Domains Analyzed:</strong> {analysis_data['rejection_patterns']['total_domains']:,}</p>
        <p><strong>Approved:</strong> {analysis_data['rejection_patterns']['approved_count']:,}</p>
        <p><strong>Rejected:</strong> {analysis_data['rejection_patterns']['rejected_count']:,}</p>
        <p><strong>Approval Rate:</strong> {analysis_data['rejection_patterns']['approval_rate']:.1f}%</p>
    </div>
    
    <h2>❌ Rejection Breakdown</h2>
    <table>
        <tr><th>Rejection Reason</th><th>Count</th><th>Percentage</th></tr>
"""
        
        total_rejected = analysis_data['rejection_patterns']['rejected_count']
        for reason, count in analysis_data['rejection_patterns']['rejection_reasons'].items():
            percentage = (count / total_rejected * 100) if total_rejected > 0 else 0
            html_content += f"<tr><td>{reason}</td><td>{count:,}</td><td>{percentage:.1f}%</td></tr>\n"
        
        html_content += """
    </table>
    
    <h2>📈 Score Distribution</h2>
    <table>
        <tr><th>Score Range</th><th>Count</th></tr>
"""
        
        for score_range, count in analysis_data['rejection_patterns']['score_distribution'].items():
            html_content += f"<tr><td>{score_range}</td><td>{count:,}</td></tr>\n"
        
        html_content += """
    </table>
    
    <h2>⚡ Threshold Analysis</h2>
    <table>
        <tr><th>Threshold</th><th>Would Approve</th><th>Approval Rate</th></tr>
"""
        
        for threshold, data in analysis_data['threshold_sensitivity'].items():
            html_content += f"<tr><td>{threshold}</td><td>{data['would_approve']:,}</td><td>{data['approval_rate']:.1f}%</td></tr>\n"
        
        html_content += """
    </table>
    
    <h2>💡 Recommendations</h2>
"""
        
        for rec in analysis_data['recommendations']:
            html_content += f'<div class="recommendation">{rec}</div>\n'
        
        html_content += """
    
    <h2>🔎 Borderline Domains Sample</h2>
    <p>Domains scoring 25-39 that might be worth reconsidering:</p>
    <table>
        <tr><th>Domain</th><th>Score</th><th>Strategy Used</th><th>Gap to Approval</th></tr>
"""
        
        # Show top 20 borderline domains
        for domain in analysis_data['borderline_domains'][:20]:
            html_content += f"""
        <tr>
            <td>{domain['domain']}</td>
            <td>{domain['score']:.1f}</td>
            <td>{domain['strategy_used'] or 'unknown'}</td>
            <td>{domain['gap_to_approval']:.1f}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <p><em>Full borderline domains list available in accompanying CSV file.</em></p>
    
</body>
</html>
"""
        
        with open(filename, 'w') as f:
            f.write(html_content)
    
    def export_borderline_csv(self, borderline_domains: List[Dict], filename: str):
        """Export borderline domains to CSV for manual review"""
        if not borderline_domains:
            return
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['domain', 'score', 'strategy_used', 'gap_to_approval', 'note']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for domain in borderline_domains:
                writer.writerow(domain)
    
    def run_complete_analysis(self):
        """Run complete analysis suite"""
        print("🚀 DOMAIN DISCOVERY ANALYSIS SUITE")
        print("=" * 50)
        
        try:
            # Check if database exists
            if not os.path.exists(self.db_path):
                print(f"❌ Database not found: {self.db_path}")
                return
            
            # Run all analyses
            analysis_data = {}
            
            analysis_data['rejection_patterns'] = self.analyze_rejection_patterns()
            analysis_data['scoring_components'] = self.analyze_scoring_components()
            analysis_data['threshold_sensitivity'] = self.analyze_threshold_sensitivity()
            analysis_data['borderline_domains'] = self.get_borderline_domains()
            analysis_data['recommendations'] = self.generate_optimization_recommendations(analysis_data)
            
            # Print summary
            self.print_summary(analysis_data)
            
            # Export reports
            self.export_analysis_report(analysis_data)
            
            print("\n✅ ANALYSIS COMPLETE")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self, analysis_data: Dict):
        """Print summary to console"""
        patterns = analysis_data['rejection_patterns']
        
        print(f"\n📊 SUMMARY")
        print(f"   Total analyzed: {patterns['total_domains']:,}")
        print(f"   Approved: {patterns['approved_count']:,}")
        print(f"   Approval rate: {patterns['approval_rate']:.1f}%")
        
        print(f"\n❌ TOP REJECTION REASONS:")
        for reason, count in sorted(patterns['rejection_reasons'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {reason}: {count:,}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        if patterns['approval_rate'] < 10:
            print("   🚨 Very low approval rate - criteria may be too strict")
        elif patterns['approval_rate'] < 20:
            print("   ⚠️  Low approval rate - consider threshold adjustment")
        else:
            print("   ✅ Approval rate seems reasonable")
        
        borderline_count = len(analysis_data['borderline_domains'])
        print(f"   🔎 {borderline_count} domains scored 25-39 (near approval)")
        
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for rec in analysis_data['recommendations'][:3]:
            print(f"   {rec}")

def main():
    """Run the analysis suite"""
    analyzer = DomainAnalysisSuite()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
