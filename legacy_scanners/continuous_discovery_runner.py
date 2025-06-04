#!/usr/bin/env python3
"""
Continuous Domain Discovery Runner
Runs discovery sessions until target number of approved domains is reached
"""

import asyncio
import sqlite3
import time
import signal
import sys
from datetime import datetime
from typing import Set
from fixed_enhanced_domain_scanner import run_fixed_discovery_pipeline

class ContinuousDiscoveryRunner:
    """Runs discovery continuously until target is reached"""
    
    def __init__(self, target_approved: int = 2000):
        self.target_approved = target_approved
        self.start_time = None
        self.run_count = 0
        self.total_discovered = 0
        self.total_validated = 0
        self.running = True
        
        # Setup progress tracking in database
        self.setup_progress_tracking()
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signal"""
        print(f"\n🛑 Interrupt received. Finishing current run...")
        self.running = False
    
    def get_current_approved_count(self) -> int:
        """Get current number of approved domains"""
        try:
            conn = sqlite3.connect('fixed_domain_discovery.db')
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM domains_analyzed WHERE current_status = "approved"')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0
    
    def get_new_approved_domains(self, initial_domains: Set[str]) -> Set[str]:
        """Get newly approved domains since start"""
        try:
            conn = sqlite3.connect('fixed_domain_discovery.db')
            cursor = conn.cursor()
            cursor.execute('SELECT domain FROM domains_analyzed WHERE current_status = "approved"')
            current_domains = {row[0] for row in cursor.fetchall()}
            conn.close()
            return current_domains - initial_domains
        except:
            return set()
    
    async def run_continuous_discovery(self):
        """Run discovery until target is reached"""
        print("🚀 CONTINUOUS DOMAIN DISCOVERY RUNNER")
        print("=" * 60)
        print(f"Target: {self.target_approved} approved domains")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = time.time()
        
        # Track initial state
        initial_approved = self.get_current_approved_count()
        initial_domains = set()
        try:
            conn = sqlite3.connect('fixed_domain_discovery.db')
            cursor = conn.cursor()
            cursor.execute('SELECT domain FROM domains_analyzed WHERE current_status = "approved"')
            initial_domains = {row[0] for row in cursor.fetchall()}
            conn.close()
        except:
            pass
        
        print(f"📋 Starting with {initial_approved} approved domains")
        remaining_needed = self.target_approved - initial_approved
        print(f"🎯 Need {remaining_needed} more domains to reach target")
        print()
        
        if remaining_needed <= 0:
            print("✅ Target already reached!")
            return
        
        last_save_time = time.time()
        save_interval = 300  # Save progress every 5 minutes
        
        while self.running:
            self.run_count += 1
            current_approved = self.get_current_approved_count()
            remaining = self.target_approved - current_approved
            
            if remaining <= 0:
                print(f"🎉 TARGET REACHED! Found {current_approved} approved domains!")
                break
            
            print(f"🔄 RUN {self.run_count} | Need {remaining} more domains")
            print("-" * 50)
            
            run_start = time.time()
            
            try:
                # Run discovery with adaptive target
                # Increase target per run if we need many more domains
                per_run_target = min(50, max(20, remaining // 10))
                
                result = await run_fixed_discovery_pipeline(target_count=per_run_target)
                
                run_duration = time.time() - run_start
                
                # Update totals
                self.total_discovered += result.get('discovered', 0)
                self.total_validated += result.get('validated', 0)
                
                # Check new approval count
                new_approved_count = self.get_current_approved_count()
                new_approvals_this_run = new_approved_count - current_approved
                
                print(f"✅ Run {self.run_count} complete:")
                print(f"   Discovered: {result.get('discovered', 0)} domains")
                print(f"   Validated: {result.get('validated', 0)} domains")
                print(f"   Approved this run: {new_approvals_this_run} domains")
                print(f"   Total approved now: {new_approved_count}")
                print(f"   Duration: {run_duration/60:.1f} minutes")
                
                # Show recently approved domains
                if new_approvals_this_run > 0:
                    new_domains = self.get_new_approved_domains(initial_domains)
                    recent_domains = sorted(list(new_domains))[-new_approvals_this_run:]
                    print(f"   New approvals: {', '.join(recent_domains)}")
                
                print()
                
                # Save progress periodically
                if time.time() - last_save_time > save_interval:
                    await self.save_progress(initial_approved, new_approved_count)
                    last_save_time = time.time()
                
                # Brief pause between runs (shorter for continuous operation)
                if self.running and remaining > 0:
                    print("⏸️  Brief pause...")
                    await asyncio.sleep(5)
                    print()
                
            except Exception as e:
                print(f"❌ Run {self.run_count} failed: {e}")
                print("🔄 Continuing with next run...")
                await asyncio.sleep(10)
                print()
        
        # Final summary
        final_approved = self.get_current_approved_count()
        total_duration = time.time() - self.start_time
        new_domains_found = final_approved - initial_approved
        
        print("📊 CONTINUOUS DISCOVERY COMPLETE")
        print("=" * 60)
        print(f"Total runtime: {total_duration/3600:.1f} hours ({total_duration/60:.1f} minutes)")
        print(f"Runs completed: {self.run_count}")
        print(f"Starting approved domains: {initial_approved}")
        print(f"Final approved domains: {final_approved}")
        print(f"New domains found: {new_domains_found}")
        print(f"Total discovered: {self.total_discovered}")
        print(f"Total validated: {self.total_validated}")
        if self.run_count > 0:
            print(f"Average per run: {new_domains_found/self.run_count:.1f} approved domains")
            print(f"Rate: {new_domains_found/(total_duration/3600):.1f} domains per hour")
        
        # Final export
        await self.final_export(initial_domains)
    
    def setup_progress_tracking(self):
        """Setup database table for progress tracking"""
        try:
            conn = sqlite3.connect('fixed_domain_discovery.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS progress_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    run_number INTEGER,
                    runtime_hours REAL,
                    domains_discovered INTEGER,
                    domains_validated INTEGER,
                    domains_approved INTEGER,
                    new_domains_this_session INTEGER,
                    target_domains INTEGER,
                    remaining_domains INTEGER
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not setup progress tracking: {e}")

    async def save_progress(self, initial_count: int, current_count: int):
        """Save progress to database and single overwritten file"""
        runtime = time.time() - (self.start_time or time.time())
        new_found = current_count - initial_count
        remaining = self.target_approved - current_count
        
        # Save to database
        try:
            conn = sqlite3.connect('fixed_domain_discovery.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO progress_tracking 
                (timestamp, run_number, runtime_hours, domains_discovered, domains_validated, 
                 domains_approved, new_domains_this_session, target_domains, remaining_domains)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(), self.run_count, runtime/3600, self.total_discovered,
                self.total_validated, current_count, new_found, self.target_approved, remaining
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not save progress to database: {e}")
        
        # Save to single overwritten progress file
        with open('current_progress.txt', 'w') as f:
            f.write(f"Continuous Discovery Progress Report\n")
            f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Runtime: {runtime/3600:.1f} hours ({runtime/60:.1f} minutes)\n")
            f.write(f"Runs completed: {self.run_count}\n")
            f.write(f"Total discovered: {self.total_discovered}\n")
            f.write(f"Total validated: {self.total_validated}\n")
            f.write(f"New domains found: {new_found}\n")
            f.write(f"Current total: {current_count}\n")
            f.write(f"Target: {self.target_approved}\n")
            f.write(f"Remaining: {remaining}\n")
            if self.run_count > 0:
                f.write(f"Rate: {new_found/(runtime/3600):.1f} domains/hour\n")
        
        print(f"💾 Progress saved to database and current_progress.txt")
    
    async def final_export(self, initial_domains: Set[str]):
        """Export final results to results directory"""
        # Get all new domains
        new_domains = self.get_new_approved_domains(initial_domains)
        
        if new_domains:
            # Save to results directory with descriptive name
            filename = 'results/latest_continuous_discovery_results.txt'
            with open(filename, 'w') as f:
                f.write(f"Continuous Discovery Results\n")
                f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Runs: {self.run_count}\n")
                f.write(f"New domains found: {len(new_domains)}\n\n")
                
                # Get domains with scores
                conn = sqlite3.connect('fixed_domain_discovery.db')
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(new_domains))
                cursor.execute(f'SELECT domain, current_score FROM domains_analyzed WHERE domain IN ({placeholders}) ORDER BY current_score DESC', list(new_domains))
                domains_with_scores = cursor.fetchall()
                conn.close()
                
                for domain, score in domains_with_scores:
                    f.write(f'{domain} ({score:.1f})\n')
            
            print(f"📄 Final results saved to {filename}")
        
        # Update current progress one final time
        runtime = time.time() - (self.start_time or time.time())
        new_found = len(new_domains)
        
        with open('current_progress.txt', 'w') as f:
            f.write(f"Continuous Discovery Session Complete\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Runtime: {runtime/3600:.1f} hours ({runtime/60:.1f} minutes)\n")
            f.write(f"Runs completed: {self.run_count}\n")
            f.write(f"Total discovered: {self.total_discovered}\n")
            f.write(f"Total validated: {self.total_validated}\n")
            f.write(f"New domains found: {new_found}\n")
            f.write(f"Status: COMPLETED\n")

async def main():
    """Run continuous discovery"""
    runner = ContinuousDiscoveryRunner(target_approved=2000)
    await runner.run_continuous_discovery()

if __name__ == "__main__":
    print("🚀 Continuous Domain Discovery Runner")
    print("Will run until 2000 approved domains are found")
    print("Press Ctrl+C to stop gracefully\n")
    
    asyncio.run(main())
