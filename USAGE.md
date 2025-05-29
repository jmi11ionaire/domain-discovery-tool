# Usage Guide: High-ROI Domain Discovery System

## Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source ai_publisher_env/bin/activate

# Verify dependencies
pip install -r requirements_ai.txt
```

### 2. Basic Usage (No AI)
```bash
# Run discovery with traditional analysis only
python domain_discovery.py
```

### 3. Enhanced Usage (With AI)
```bash
# Set your API key (choose one)
export OPENAI_API_KEY="your-openai-key-here"
# OR
export ANTHROPIC_API_KEY="your-anthropic-key-here"

# Run enhanced discovery with LLM final recommendations
python domain_discovery.py
```

## How It Works

### Phase 1: High-Volume Filtering
- Analyzes 25 high-traffic domains
- **Filters for ads.txt files only** (efficiency)
- Parses ads.txt for premium DSPs and direct deals
- Estimates ad slot availability

### Phase 2: LLM Enhancement (Optional)
- Takes domains with 60+ scores
- Sends clean content + metrics to LLM
- Gets AI recommendation: APPROVE/REJECT/CAUTION
- Combines scores: `(base_score * 0.6) + (ai_score * 0.4)`

## Output
- **Console**: Detailed analysis with AI recommendations
- **CSV**: `high_roi_publishers.csv` for immediate use
- **Database**: `high_roi_publishers.db` for persistence

## Example Results
```
🥇 PREMIUM INVENTORY (1 domains):
   financialexpress.com      Final: 96.5
      Premium DSPs: 295 | Direct Deals: 173 | Ad Slots:  47
      🤖 AI: APPROVE | LLM Score: 85.0
      💭 High-quality financial content with strong B2B focus...
```

## API Keys
- **OpenAI**: Uses GPT-3.5-turbo ($0.002/1K tokens)
- **Anthropic**: Uses Claude-3-haiku ($0.00025/1K tokens) 
- **Without keys**: System runs traditional analysis only

## Customization
Edit `get_discovery_targets()` in `domain_discovery.py` to add your specific domains.

## ROI Focus
- ✅ Only analyzes domains with ads.txt (confirmed ad sellers)
- ✅ Prioritizes premium DSP partnerships
- ✅ Identifies direct deal opportunities
- ✅ AI validates quality and brand safety
