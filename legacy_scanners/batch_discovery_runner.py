#!/usr/bin/env python3
"""
Batch Domain Discovery Runner
Runs multiple consecutive discovery sessions to maximize domain findings
"""

import asyncio
import sqlite3
import time
from datetime import datetime
from typing import List, Dict, Set
from fixed_enhanced_domain_scanner import run_fixed_discovery_pipeline, FixedEnhancedDomainScanner

class BatchDiscoveryRunner:
    """Runs multiple discovery sessions and aggregates results"""
    
    def __init__(self, num_runs: int = 10):
        self.num_runs = num_runs
        self.all_approved_domains = []
        self.run_statistics = []
        self.start_time = None
        
    def get_current_approved_domains(self) -> Set[str]:
        """Get currently approved domains from database"""
        approved_domains = set()
        try:
            conn = sqlite3.connect('fixed_domain_discovery.db')
            cursor = conn.cursor()
            cursor.execute('SELECT domain FROM domains_analyzed WHERE current_status = "approved"')
            results = cursor.fetchall()
            approved_domains = {domain[0] for domain in results}
            conn.close()
        except Exception as e:
            print(f"Warning: Could not read existing approved domains: {e}")
        return approved_domains
    
    async def run_batch_discovery(self):
        """Run multiple discovery sessions consecutively"""
        print("🚀 BATCH DOMAIN DISCOVERY RUNNER")
        print("=" * 50)
        print(f"Starting {self.num_runs} consecutive discovery runs...")
        print(f"Target: 20-30 domains per run")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = time.time()
        
        # Track initial state
        initial_approved = self.get_current_approved_domains()
        print(f"📋 Starting with {len(initial_approved)} previously approved domains")
        print()
        
        total_discovered = 0
        total_validated = 0
        total_approved = 0
        
        for run_num in range(1, self.num_runs + 1):
            print(f"🔄 RUN {run_num}/{self.num_runs}")
            print("-" * 30)
            
            run_start = time.time()
            
            try:
                # Run discovery pipeline
                result = await run_fixed_discovery_pipeline(target_count=25)
                
                run_duration = time.time() - run_start
                
                # Track statistics
                run_stats = {
                    'run_number': run_num,
                    'discovered': result.get('discovered', 0),
                    'validated': result.get('validated', 0),
                    'approved': result.get('approved', 0),
                    'validation_efficiency': result.get('validation_efficiency', 0),
                    'success_rate': result.get('success_rate', 0),
                    'duration_minutes': run_duration / 60
                }
                
                self.run_statistics.append(run_stats)
                
                # Update totals
                total_discovered += run_stats['discovered']
                total_validated += run_stats['validated']
                total_approved += run_stats['approved']
                
                print(f"✅ Run {run_num} complete:")
                print(f"   Discovered: {run_stats['discovered']} domains")
                print(f"   Validated: {run_stats['validated']} domains")
                print(f"   Approved: {run_stats['approved']} domains")
                print(f"   Duration: {run_stats['duration_minutes']:.1f} minutes")
                print(f"   Success rate: {run_stats['success_rate']:.1f}%")
                print()
                
                # Brief pause between runs
                if run_num < self.num_runs:
                    print("⏸️  Pausing 10 seconds before next run...")
                    await asyncio.sleep(10)
                    print()
                
            except Exception as e:
                print(f"❌ Run {run_num} failed: {e}")
                run_stats = {
                    'run_number': run_num,
                    'discovered': 0,
                    'validated': 0,
                    'approved': 0,
                    'validation_efficiency': 0,
                    'success_rate': 0,
                    'duration_minutes': (time.time() - run_start) / 60,
                    'error': str(e)
                }
                self.run_statistics.append(run_stats)
                print()
        
        # Get final approved domains
        final_approved = self.get_current_approved_domains()
        new_domains = final_approved - initial_approved
        
        total_duration = time.time() - self.start_time
        
        print("📊 BATCH DISCOVERY COMPLETE")
        print("=" * 50)
        print(f"Total runtime: {total_duration/60:.1f} minutes")
        print(f"Runs completed: {len(self.run_statistics)}")
        print()
        print(f"CUMULATIVE RESULTS:")
        print(f"   Total discovered: {total_discovered} domains")
        print(f"   Total validated: {total_validated} domains") 
        print(f"   Total approved: {total_approved} domains")
        print(f"   New domains added: {len(new_domains)}")
        print(f"   Final approved count: {len(final_approved)}")
        print()
        
        if new_domains:
            print(f"🎉 NEW APPROVED DOMAINS ({len(new_domains)}):")
            # Get scores for new domains
            conn = sqlite3.connect('fixed_domain_discovery.db')
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(new_domains))
            cursor.execute(f'SELECT domain, current_score FROM domains_analyzed WHERE domain IN ({placeholders}) ORDER BY current_score DESC', list(new_domains))
            new_with_scores = cursor.fetchall()
            conn.close()
            
            for domain, score in new_with_scores:
                print(f"   ✅ {domain} (Score: {score:.1f})")
        
        # Export results
        await self.export_results(final_approved, new_domains)
        
        return {
            'total_runs': len(self.run_statistics),
            'total_discovered': total_discovered,
            'total_validated': total_validated,
            'total_approved': total_approved,
            'new_domains': len(new_domains),
            'final_count': len(final_approved),
            'duration_minutes': total_duration / 60
        }
    
    async def export_results(self, final_approved: Set[str], new_domains: Set[str]):
        """Export batch results to results directory"""
        # Export run statistics
        stats_file = 'results/latest_batch_discovery_stats.csv'
        with open(stats_file, 'w') as f:
            f.write('run_number,discovered,validated,approved,validation_efficiency,success_rate,duration_minutes\n')
            for stats in self.run_statistics:
                f.write(f"{stats['run_number']},{stats['discovered']},{stats['validated']},{stats['approved']},{stats['validation_efficiency']:.1f},{stats['success_rate']:.1f},{stats['duration_minutes']:.2f}\n")
        
        # Export new approved domains
        if new_domains:
            new_domains_file = 'results/latest_batch_new_domains.txt'
            with open(new_domains_file, 'w') as f:
                f.write(f'# New Approved Domains from Batch Discovery\n')
                f.write(f'# Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'# Total runs: {len(self.run_statistics)}\n')
                f.write(f'# New domains found: {len(new_domains)}\n\n')
                
                # Get domains with scores
                conn = sqlite3.connect('fixed_domain_discovery.db')
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(new_domains))
                cursor.execute(f'SELECT domain, current_score FROM domains_analyzed WHERE domain IN ({placeholders}) ORDER BY current_score DESC', list(new_domains))
                domains_with_scores = cursor.fetchall()
                conn.close()
                
                for domain, score in domains_with_scores:
                    f.write(f'{domain}\n')
            
            print(f"\n📁 FILES CREATED:")
            print(f"   📊 Statistics: {stats_file}")
            print(f"   📋 New domains: {new_domains_file}")
        else:
            print(f"\n📁 FILES CREATED:")
            print(f"   📊 Statistics: {stats_file}")

async def main():
    """Run batch discovery"""
    runner = BatchDiscoveryRunner(num_runs=10)
    results = await runner.run_batch_discovery()
    
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   Completed {results['total_runs']} discovery runs")
    print(f"   Found {results['new_domains']} new high-quality publisher domains")
    print(f"   Total approved domains: {results['final_count']}")
    print(f"   Total runtime: {results['duration_minutes']:.1f} minutes")
    print(f"   Average: {results['duration_minutes']/results['total_runs']:.1f} minutes per run")

if __name__ == "__main__":
    print("🚀 Batch Domain Discovery Runner")
    print("Running 10 consecutive discovery sessions")
    print("Estimated runtime: 30-60 minutes\n")
    
    asyncio.run(main())
