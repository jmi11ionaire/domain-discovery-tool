#!/usr/bin/env python3
"""
Export newly discovered domains separately from existing DSP domains
Keeps clean separation between original DSP list and service discoveries
"""

import sqlite3
from datetime import datetime

def export_discovered_domains():
    """Export only service-discovered domains to separate file"""
    
    # Get approved domains from our discovery service
    conn = sqlite3.connect('domain_discovery.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT domain, current_score, last_analyzed_at, discovery_source 
        FROM domains_analyzed 
        WHERE current_status = 'approved' 
        ORDER BY current_score DESC
    """)
    discovered = cursor.fetchall()
    conn.close()
    
    if not discovered:
        print("📋 No domains discovered by service yet")
        return
    
    print(f"📋 EXPORTING {len(discovered)} SERVICE-DISCOVERED DOMAINS")
    print("=" * 60)
    
    # Export to separate file with metadata
    filename = f"discovered_domains_{datetime.now().strftime('%Y%m%d')}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"# Service-Discovered Domains - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total: {len(discovered)} domains\n")
        f.write(f"# Separate from existing DSP domains in existing_domains.txt\n")
        f.write(f"#\n")
        f.write(f"# Domain | Score | Analyzed | Source\n")
        f.write(f"#\n")
        
        for domain, score, analyzed_at, source in discovered:
            f.write(f"{domain}\n")
    
    # Also create a detailed report
    report_filename = f"discovered_domains_report_{datetime.now().strftime('%Y%m%d')}.txt"
    
    with open(report_filename, 'w') as f:
        f.write(f"SERVICE-DISCOVERED DOMAINS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Discovered: {len(discovered)}\n")
        f.write("=" * 60 + "\n\n")
        
        for domain, score, analyzed_at, source in discovered:
            f.write(f"Domain: {domain}\n")
            f.write(f"Score: {score:.1f}\n")
            f.write(f"Analyzed: {analyzed_at}\n")
            f.write(f"Source: {source}\n")
            f.write("-" * 40 + "\n")
    
    print(f"✅ Service discoveries exported to:")
    print(f"   📄 {filename} (domain list)")
    print(f"   📊 {report_filename} (detailed report)")
    
    # Show summary
    print(f"\n🎯 SERVICE DISCOVERY SUMMARY:")
    print(f"   Discovered domains: {len(discovered)}")
    
    high_quality = len([d for d in discovered if d[1] >= 80])
    if high_quality > 0:
        print(f"   High quality (80+): {high_quality}")
    
    print(f"\n📋 ORIGINAL DSP DOMAINS:")
    try:
        with open('existing_domains.txt', 'r') as f:
            original_count = len([line for line in f if line.strip() and not line.startswith('#')])
        print(f"   Original DSP domains: {original_count}")
        print(f"   Total combined: {original_count + len(discovered)}")
    except FileNotFoundError:
        print(f"   No existing_domains.txt found")
    
    print(f"\n💡 USAGE:")
    print(f"   DSP domains: existing_domains.txt")
    print(f"   New discoveries: {filename}")
    print(f"   Combined for analysis: both files are checked")

if __name__ == "__main__":
    export_discovered_domains()
