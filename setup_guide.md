# Setup Guide for AI-Enhanced Publisher Discovery System

## Quick Start

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements_ai.txt

# Install spaCy language model
python -m spacy download en_core_web_sm
```

### 2. Set Up API Keys

You have two options for LLM providers:

**Option A: OpenAI**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

**Option B: Anthropic Claude**
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

**Option C: Both (Recommended)**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

### 3. Run Demo

```bash
# Run the demo script
python demo_ai.py
```

## Detailed Setup

### Dependencies Breakdown

The `requirements_ai.txt` includes:

**Core Web Framework:**
- `aiohttp` - Async HTTP client for web scraping
- `beautifulsoup4` - HTML parsing
- `pandas` - Data analysis

**AI & LLM Libraries:**
- `openai` - OpenAI GPT-4 integration
- `anthropic` - Anthropic Claude integration
- `transformers` - Hugging Face models
- `torch` - PyTorch for ML models

**NLP & ML:**
- `spacy` - Advanced NLP processing
- `scikit-learn` - Machine learning algorithms
- `numpy` - Numerical computing

### API Key Setup Options

#### Environment Variables (Recommended)
```bash
# Add to your .bashrc or .zshrc
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### .env File
Create a `.env` file in your project directory:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

### Getting API Keys

**OpenAI:**
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy the key (starts with "sk-...")

**Anthropic:**
1. Go to https://console.anthropic.com/
2. Generate API key
3. Copy the key (starts with "sk-ant-...")

### Installation Troubleshooting

**If you get import errors:**
```bash
# Update pip first
pip install --upgrade pip

# Install with verbose output to see issues
pip install -r requirements_ai.txt -v
```

**For Apple Silicon Macs:**
```bash
# Install with conda for better compatibility
conda install pytorch transformers -c pytorch
pip install openai anthropic spacy scikit-learn
```

**For older Python versions:**
```bash
# Check Python version (requires 3.8+)
python --version

# Upgrade if needed
brew install python@3.11  # macOS
```

### Testing Installation

Test your setup step by step:

```python
# Test 1: Basic imports
import aiohttp
import pandas as pd
print("✅ Core libraries work")

# Test 2: AI libraries
try:
    import openai
    import anthropic
    print("✅ LLM libraries work")
except ImportError as e:
    print(f"❌ LLM library issue: {e}")

# Test 3: ML libraries
try:
    import transformers
    import sklearn
    import spacy
    print("✅ ML libraries work")
except ImportError as e:
    print(f"❌ ML library issue: {e}")

# Test 4: spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded")
except OSError:
    print("❌ Run: python -m spacy download en_core_web_sm")
```

### Running Without AI APIs

If you don't have API keys yet, you can still test the fallback functionality:

```python
# This will use rule-based analysis instead of LLMs
python demo_ai.py
```

The system will automatically fall back to traditional analysis methods when API keys aren't available.

### Cost Considerations

**OpenAI GPT-4 Pricing (approximate):**
- Input: $0.03 per 1K tokens
- Output: $0.06 per 1K tokens
- Typical analysis: ~$0.01-0.02 per domain

**Anthropic Claude Pricing (approximate):**
- Input: $0.015 per 1K tokens  
- Output: $0.075 per 1K tokens
- Typical analysis: ~$0.005-0.015 per domain

**Cost optimization tips:**
- Use content sampling (first 4000 chars)
- Batch multiple domains
- Cache results to avoid re-analysis
- Start with smaller batches for testing

### Production Configuration

For production use, consider:

1. **Rate Limiting:**
```python
discovery = AIPublisherDiscovery(
    max_concurrent=2,  # Limit concurrent requests
    api_delay=1.0      # Add delay between requests
)
```

2. **Database Configuration:**
```python
# Use PostgreSQL for production
import psycopg2
# Update database configuration in the code
```

3. **Monitoring:**
```python
import logging
logging.basicConfig(level=logging.INFO)
# Add metrics tracking for API costs and performance
```

## Next Steps

1. **Install dependencies** from requirements_ai.txt
2. **Get API keys** from OpenAI or Anthropic
3. **Run demo** to see AI capabilities
4. **Analyze your domains** using the AI system
5. **Review integration guide** in ai_integration_overview.md

Ready to transform your publisher discovery with AI! 🚀
