#!/usr/bin/env python3
"""
Continuous Discovery Service
Configuration-driven service that runs until target domain count reached
"""

import asyncio
import argparse
import sqlite3
import time
import yaml
import sys
import signal
from datetime import datetime, timedelta
from typing import Dict, Optional
from optimized_domain_scanner import run_optimized_discovery_pipeline

class ContinuousDiscoveryService:
    """Configuration-driven continuous discovery service"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        self.config = self._load_config()
        if config_override:
            self._deep_merge(self.config, config_override)
        
        self.db_path = "optimized_domain_discovery.db"
        self.start_time = time.time()
        self.batch_count = 0
        self.session_approved = 0
        self.running = True
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Extract continuous discovery settings
        self.cd_config = self.config.get('continuous_discovery', {})
        self.target_domains = self.cd_config.get('target_total_domains', 500)
        self.batch_size = self.cd_config.get('batch_size', 20)
        self.delay_between_batches = self.cd_config.get('delay_between_batches', 30)
        self.max_runtime_hours = self.cd_config.get('max_runtime_hours', 24)
        self.progress_report_interval = self.cd_config.get('progress_report_interval', 5)
        self.auto_export = self.cd_config.get('auto_export_on_completion', True)
        
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            with open('config/scanner_config.yaml', 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown signals"""
        print(f"\n🛑 Received shutdown signal ({signum}). Finishing current batch...")
        self.running = False
    
    def get_current_approved_count(self) -> int:
        """Get current count of approved domains"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM domains_analyzed WHERE current_status = 'approved'")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"Error getting approved count: {e}")
            return 0
    
    def calculate_time_estimates(self, current_count: int, domains_needed: int) -> Dict:
        """Calculate time estimates based on current performance"""
        if self.batch_count == 0 or self.session_approved == 0:
            return {
                'estimated_batches': max(1, domains_needed // max(1, self.batch_size // 10)),  # Assume 10% approval rate
                'estimated_time_minutes': 'Unknown (need more data)',
                'approval_rate': 'Unknown'
            }
        
        # Calculate rates
        elapsed_minutes = (time.time() - self.start_time) / 60
        approval_rate = self.session_approved / self.batch_count if self.batch_count > 0 else 0.1
        avg_batch_time = elapsed_minutes / self.batch_count if self.batch_count > 0 else 2.0
        
        # Estimate remaining work
        estimated_batches_needed = max(1, int(domains_needed / max(0.1, approval_rate)))
        estimated_time_minutes = estimated_batches_needed * avg_batch_time
        
        return {
            'estimated_batches': estimated_batches_needed,
            'estimated_time_minutes': f"{estimated_time_minutes:.1f} minutes ({estimated_time_minutes/60:.1f} hours)",
            'approval_rate': f"{approval_rate:.1f} per batch ({approval_rate/self.batch_size*100:.1f}%)",
            'avg_batch_time': f"{avg_batch_time:.1f} minutes"
        }
    
    def print_progress_report(self, current_count: int, domains_needed: int):
        """Print detailed progress report"""
        elapsed = time.time() - self.start_time
        estimates = self.calculate_time_estimates(current_count, domains_needed)
        
        print(f"\n📊 PROGRESS REPORT - Batch {self.batch_count}")
        print("=" * 60)
        print(f"   Current approved: {current_count:,}")
        print(f"   Target domains: {self.target_domains:,}")
        print(f"   Domains needed: {domains_needed:,}")
        print(f"   Progress: {(current_count/self.target_domains*100):.1f}%")
        print(f"   Session approved: {self.session_approved}")
        print(f"   Runtime: {elapsed/60:.1f} minutes")
        print(f"   Approval rate: {estimates['approval_rate']}")
        print(f"   Estimated batches remaining: {estimates['estimated_batches']}")
        print(f"   Estimated time remaining: {estimates['estimated_time_minutes']}")
        print("=" * 60)
    
    def save_progress_checkpoint(self, current_count: int):
        """Save progress checkpoint"""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'current_count': current_count,
            'target_count': self.target_domains,
            'batch_count': self.batch_count,
            'session_approved': self.session_approved,
            'runtime_minutes': (time.time() - self.start_time) / 60
        }
        
        try:
            with open(f'archives/continuous_progress_{int(time.time())}.txt', 'w') as f:
                f.write(f"Continuous Discovery Progress - {checkpoint['timestamp']}\n")
                f.write(f"Current: {current_count}/{self.target_domains} domains\n")
                f.write(f"Batches completed: {self.batch_count}\n")
                f.write(f"Session approved: {self.session_approved}\n")
                f.write(f"Runtime: {checkpoint['runtime_minutes']:.1f} minutes\n")
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")
    
    async def run_continuous_discovery(self):
        """Run continuous discovery until target reached"""
        print("🚀 CONTINUOUS DISCOVERY SERVICE STARTING")
        print("=" * 70)
        print(f"   Target domains: {self.target_domains:,}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Delay between batches: {self.delay_between_batches}s")
        print(f"   Max runtime: {self.max_runtime_hours} hours")
        print("=" * 70)
        
        while self.running:
            # Check current status
            current_count = self.get_current_approved_count()
            domains_needed = self.target_domains - current_count
            
            # Check if we've reached the target
            if domains_needed <= 0:
                print(f"\n🎉 TARGET REACHED! Current domains: {current_count:,}")
                break
            
            # Check runtime limit
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.max_runtime_hours:
                print(f"\n⏰ Max runtime ({self.max_runtime_hours}h) reached. Stopping.")
                break
            
            # Progress report
            if self.batch_count % self.progress_report_interval == 0:
                self.print_progress_report(current_count, domains_needed)
            
            # Run a batch
            print(f"\n🔄 Starting Batch {self.batch_count + 1}")
            print(f"   Need {domains_needed:,} more domains to reach target")
            
            try:
                batch_start = time.time()
                result = await run_optimized_discovery_pipeline(self.batch_size)
                batch_time = (time.time() - batch_start) / 60
                
                self.batch_count += 1
                self.session_approved += result.get('approved', 0)
                
                print(f"✅ Batch {self.batch_count} complete:")
                print(f"   Approved: {result.get('approved', 0)} domains")
                print(f"   Batch time: {batch_time:.1f} minutes")
                print(f"   Session total: {self.session_approved} approved")
                
                # Save progress
                if self.cd_config.get('save_progress_every_batch', True):
                    self.save_progress_checkpoint(current_count + result.get('approved', 0))
                
            except Exception as e:
                print(f"❌ Batch {self.batch_count + 1} failed: {e}")
                print("   Continuing with next batch...")
            
            # Check if we should continue
            if not self.running:
                break
            
            # Wait between batches
            if self.delay_between_batches > 0:
                print(f"⏳ Waiting {self.delay_between_batches}s before next batch...")
                for i in range(self.delay_between_batches):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
        
        # Final report
        await self._final_report()
    
    async def _final_report(self):
        """Generate final completion report"""
        final_count = self.get_current_approved_count()
        runtime = (time.time() - self.start_time) / 60
        
        print(f"\n🏁 CONTINUOUS DISCOVERY SERVICE COMPLETE")
        print("=" * 70)
        print(f"   Final domain count: {final_count:,}")
        print(f"   Target was: {self.target_domains:,}")
        print(f"   Batches completed: {self.batch_count}")
        print(f"   Session approved: {self.session_approved}")
        print(f"   Total runtime: {runtime:.1f} minutes ({runtime/60:.1f} hours)")
        
        if self.batch_count > 0:
            print(f"   Average per batch: {self.session_approved/self.batch_count:.1f} domains")
            print(f"   Average batch time: {runtime/self.batch_count:.1f} minutes")
        
        if final_count >= self.target_domains:
            print("   🎯 TARGET ACHIEVED!")
        else:
            print(f"   📊 Progress: {(final_count/self.target_domains*100):.1f}%")
        
        # Auto-export if requested
        if self.auto_export and final_count >= self.target_domains:
            print(f"\n📋 Auto-exporting discovered domains...")
            try:
                import subprocess
                subprocess.run([sys.executable, "utilities/export_discovered_domains.py"], check=True)
                print("✅ Service discoveries exported successfully")
            except Exception as e:
                print(f"❌ Export failed: {e}")
        
        print("=" * 70)

async def main():
    """Main entry point with command line argument support"""
    parser = argparse.ArgumentParser(description='Continuous Domain Discovery Service')
    parser.add_argument('--target', type=int, help='Override target domain count')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--delay', type=int, help='Override delay between batches (seconds)')
    parser.add_argument('--max-hours', type=float, help='Override max runtime hours')
    
    args = parser.parse_args()
    
    # Build config overrides
    config_override = {}
    if args.target:
        config_override['continuous_discovery'] = {'target_total_domains': args.target}
    if args.batch_size:
        config_override.setdefault('continuous_discovery', {})['batch_size'] = args.batch_size
    if args.delay:
        config_override.setdefault('continuous_discovery', {})['delay_between_batches'] = args.delay
    if args.max_hours:
        config_override.setdefault('continuous_discovery', {})['max_runtime_hours'] = args.max_hours
    
    # Start service
    service = ContinuousDiscoveryService(config_override)
    await service.run_continuous_discovery()

if __name__ == "__main__":
    print("🚀 Continuous Domain Discovery Service")
    print("Configuration-driven service with intelligent progress tracking")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Service stopped by user")
    except Exception as e:
        print(f"\n❌ Service error: {e}")
        sys.exit(1)
