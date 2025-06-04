# Continuous Discovery Service Guide

## 🚀 Quick Start

The easiest way to run continuous discovery to any target:

### 1. Edit Target in Config
```bash
# Edit config/scanner_config.yaml
# Change this line:
target_total_domains: 500  # Set to your desired total

# Then simply run:
python continuous_discovery_service.py
```

### 2. Or Override via Command Line
```bash
# Run to 500 domains
python continuous_discovery_service.py --target 500

# Run to 1000 domains with larger batches
python continuous_discovery_service.py --target 1000 --batch-size 30

# Quick test run (smaller batch, faster)
python continuous_discovery_service.py --target 150 --batch-size 10 --delay 10
```

## ⚙️ Configuration Options

All settings in `config/scanner_config.yaml` under `continuous_discovery`:

```yaml
continuous_discovery:
  target_total_domains: 500           # Stop when we reach this many approved domains
  batch_size: 20                      # Domains to process per batch
  delay_between_batches: 30           # Seconds to wait between batches
  max_runtime_hours: 24               # Safety limit - stop after this many hours
  progress_report_interval: 5         # Report progress every N batches
  auto_export_on_completion: true     # Export domains when target reached
  save_progress_every_batch: true     # Save progress after each batch
```

## 📊 What You'll See

```
🚀 CONTINUOUS DISCOVERY SERVICE STARTING
======================================================================
   Target domains: 500
   Batch size: 20
   Delay between batches: 30s
   Max runtime: 24 hours
======================================================================

📊 PROGRESS REPORT - Batch 5
============================================================
   Current approved: 147
   Target domains: 500
   Domains needed: 353
   Progress: 29.4%
   Session approved: 12
   Runtime: 15.3 minutes
   Approval rate: 2.4 per batch (12.0%)
   Estimated batches remaining: 147
   Estimated time remaining: 449.1 minutes (7.5 hours)
============================================================
```

## 🎯 Time Estimates

The service learns as it runs and gives you accurate estimates:
- **First few batches**: "Unknown (need more data)"
- **After 5+ batches**: "Estimated 3.2 hours remaining"
- **Updates continuously** as performance changes

## 🛑 Stopping & Resuming

### Graceful Stop
- **Ctrl+C**: Finishes current batch then stops
- **Automatic**: Stops when target reached or max time exceeded

### Resume Later
- Just run the service again - it checks current count and continues
- All progress is saved in `archives/continuous_progress_*.txt`

## 📈 Current Status

We currently have **136 approved domains**. Examples of recent targets:

- **500 domains**: ~364 more needed (~18-30 hours estimated)
- **200 domains**: ~64 more needed (~3-6 hours estimated)  
- **300 domains**: ~164 more needed (~8-15 hours estimated)

## 🎛️ Environment Controls

Different settings for dev vs production:

```yaml
environments:
  development:
    continuous_discovery:
      delay_between_batches: 10  # Faster for testing
  
  production:
    continuous_discovery:
      delay_between_batches: 60  # More conservative
```

## ⚡ Examples

```bash
# Quick test: Add 20 more domains
python continuous_discovery_service.py --target 156

# Overnight run: Get to 500 domains
python continuous_discovery_service.py --target 500

# Large production run: Get to 1000 domains
python continuous_discovery_service.py --target 1000 --batch-size 25 --delay 45 --max-hours 48
```

---

**Simple Rule**: Change the target number, run the service, walk away! ✨
