#!/usr/bin/env python3
"""
Simple script to check batch discovery results
"""

import sqlite3
import os

print("📊 BATCH DISCOVERY RESULTS")
print("=" * 30)

try:
    conn = sqlite3.connect('fixed_domain_discovery.db')
    cursor = conn.cursor()
    
    # Get approved domains
    cursor.execute('SELECT domain, current_score FROM domains_analyzed WHERE current_status = "approved" ORDER BY current_score DESC')
    approved = cursor.fetchall()
    
    # Get total count
    cursor.execute('SELECT COUNT(*) FROM domains_analyzed')
    total = cursor.fetchone()[0]
    
    print(f"✅ APPROVED: {len(approved)} domains")
    for domain, score in approved:
        print(f"   {domain} ({score:.1f})")
    
    print(f"\n📊 TOTAL ANALYZED: {total}")
    if total > 0:
        print(f"📈 SUCCESS RATE: {(len(approved)/total)*100:.1f}%")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")

# Check for export files
print(f"\n📁 FILES:")
for f in os.listdir('.'):
    if f.startswith('batch_') or f.startswith('NEW_'):
        print(f"   {f}")
