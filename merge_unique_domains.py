#!/usr/bin/env python3
"""
Merge unique domains from legacy files into service_discovered_domains.txt
"""

def read_domains_from_file(filepath: str) -> set:
    """Read domains from file, returning set of clean domains"""
    domains = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                domain = line.strip().replace('www.', '').lower()
                if domain and not line.startswith('#'):
                    domains.add(domain)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    return domains

def main():
    """Merge unique domains into service_discovered_domains.txt"""
    
    # Read current service domains
    service_domains = read_domains_from_file('service_discovered_domains.txt')
    print(f"📊 Current service_discovered_domains.txt: {len(service_domains)} domains")
    
    # Read legacy files
    existing_domains = read_domains_from_file('existing_domains.txt')
    discovered_domains = read_domains_from_file('discovered_domains_20250604.txt')
    
    print(f"📊 existing_domains.txt: {len(existing_domains)} domains")
    print(f"📊 discovered_domains_20250604.txt: {len(discovered_domains)} domains")
    
    # Find domains that are NOT in service file
    new_domains = set()
    
    for domain in existing_domains:
        if domain not in service_domains:
            new_domains.add(domain)
    
    for domain in discovered_domains:
        if domain not in service_domains:
            new_domains.add(domain)
    
    print(f"\n🔍 ANALYSIS:")
    print(f"   Unique domains to add: {len(new_domains)}")
    
    if new_domains:
        print(f"\n📝 Adding {len(new_domains)} unique domains to service_discovered_domains.txt:")
        
        # Append unique domains to service file
        with open('service_discovered_domains.txt', 'a') as f:
            for domain in sorted(new_domains):
                f.write(f"{domain}\n")
                print(f"   + {domain}")
        
        print(f"\n✅ MERGE COMPLETE:")
        print(f"   Added {len(new_domains)} unique domains")
        print(f"   Total in service_discovered_domains.txt: {len(service_domains) + len(new_domains)}")
    else:
        print(f"\n✅ NO ACTION NEEDED:")
        print(f"   All domains already in service_discovered_domains.txt")

if __name__ == "__main__":
    main()
