#!/usr/bin/env python3
"""
Export approved domains to publisher list
"""

import sqlite3
from datetime import datetime

# Get approved domains
conn = sqlite3.connect('fixed_domain_discovery.db')
cursor = conn.cursor()
cursor.execute('SELECT domain FROM domains_analyzed WHERE current_status = "approved" ORDER BY domain')
approved = [row[0] for row in cursor.fetchall()]
conn.close()

print(f"📋 EXPORTING {len(approved)} APPROVED DOMAINS")
print("=" * 40)

# Update existing_domains.txt
try:
    with open('existing_domains.txt', 'r') as f:
        existing = set(line.strip() for line in f if line.strip())
except FileNotFoundError:
    existing = set()

new_domains = [d for d in approved if d not in existing]

if new_domains:
    print(f"✅ Adding {len(new_domains)} new domains:")
    for domain in new_domains:
        print(f"   {domain}")
    
    with open('existing_domains.txt', 'a') as f:
        for domain in new_domains:
            f.write(f"{domain}\n")
    
    print(f"\n📝 Updated existing_domains.txt")
else:
    print("ℹ️  All approved domains already in existing_domains.txt")

print(f"\n🎯 TOTAL PUBLISHER DOMAINS: {len(existing) + len(new_domains)}")
