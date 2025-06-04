# Repository Cleanup Analysis

## Current Production System (KEEP)
- `fixed_enhanced_domain_scanner.py` - ✅ **CURRENT ACTIVE SYSTEM** (running successfully)
- `continuous_discovery_runner.py` - ✅ **PRODUCTION RUNNER** (uses database tracking)
- `silent_discovery_runner.py` - ✅ **SIMPLE RUNNER** (lightweight execution)
- `robust_domain_validator.py` - ✅ **CORE COMPONENT** (used by fixed scanner)
- `robust_anthropic_client.py` - ✅ **BACKUP CLIENT** (diagnostics)
- `anthropic_diagnostics.py` - ✅ **DIAGNOSTIC TOOL** (troubleshooting)

## Utility Scripts (KEEP)
- `batch_discovery_runner.py` - ✅ **BATCH PROCESSOR** (different use case)
- `check_results.py` - ✅ **ANALYSIS TOOL** 
- `export_domains.py` - ✅ **EXPORT UTILITY**

## Redundant Scanner Versions (MOVE TO ARCHIVES)
- `enhanced_unified_domain_scanner.py` - ❌ **REDUNDANT** (older version)
- `unified_domain_scanner.py` - ❌ **REDUNDANT** (oldest version)

## Ghost Files (VSCode tabs but don't exist)
These appear in tabs but don't exist in repo:
- `enterprise_discovery_5k.py`
- `improved_enterprise_discovery.py` 
- `granular_enterprise_discovery.py`
- `comprehensive_domain_rescanner.py`
- `rescan_with_improved_scoring.py`
- `domain_expansion_discovery.py`
- `simple_domain_discovery.py`
- `test_discovery.py`
- `focused_discovery.py`
- `domain_discovery.py`
- `production_domain_discovery.py`
- `compromise_domain_discovery.py`
- `unified_domain_scanner_complete.py`
- `dual_strategy_discovery.py`
- `api_enhanced_discovery.py`
- `enterprise_discovery_complete.py`

## Progress File Issue Fixed
- **OLD**: Created multiple `continuous_progress_*.txt` files (bloat)
- **NEW**: Single `current_progress.txt` + database tracking
- **ACTION**: Moved old progress files to `archives/`

## Recommended Actions
1. Move redundant scanner versions to archives
2. Keep current production system intact
3. Database tracking eliminates txt file bloat
4. Close ghost file tabs in VSCode
