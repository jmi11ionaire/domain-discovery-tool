# Domain Discovery System

An intelligent domain and publisher discovery system that combines traditional web scraping with AI-powered content analysis to identify and classify websites and publishers.

## Features

- **AI-Enhanced Publisher Discovery**: Uses OpenAI and Anthropic APIs for intelligent content analysis
- **Traditional Web Scraping**: Robust scraping capabilities with aiohttp and BeautifulSoup
- **Content Classification**: Automatic categorization of websites and publishers
- **Async Processing**: High-performance asynchronous operations for scalable discovery
- **Multiple Data Sources**: Supports various input formats and sources
- **Comprehensive Analysis**: Domain authority, content quality, and relevance scoring

## Project Structure

```
domain_discovery/
├── publisher_discovery.py          # Core discovery engine
├── ai_enhanced_publisher_discovery.py  # AI-powered analysis
├── demo.py                        # Basic demo
├── demo_ai.py                     # AI demo
├── test_setup.py                  # Environment testing
├── requirements.txt               # Basic dependencies
├── requirements_ai.txt            # AI dependencies
├── setup_guide.md                 # Setup instructions
├── ai_integration_overview.md     # AI features documentation
└── publisher_discovery_system.md  # System documentation
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv ai_publisher_env
source ai_publisher_env/bin/activate  # On Windows: ai_publisher_env\Scripts\activate

# Install dependencies
pip install -r requirements_ai.txt
```

### 2. Configuration

Set up your API keys in environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Run Demos

```bash
# Basic demo
python demo.py

# AI-enhanced demo
python demo_ai.py
```

### 4. Test Your Setup

```bash
python test_setup.py
```

## Dependencies

### Core Dependencies
- `aiohttp` - Async HTTP client
- `pandas` - Data manipulation
- `beautifulsoup4` - HTML parsing
- `sqlite3` - Database operations

### AI Dependencies
- `openai` - OpenAI API integration
- `anthropic` - Anthropic API integration
- `transformers` - Hugging Face transformers
- `torch` - PyTorch for ML models
- `scikit-learn` - Machine learning utilities
- `spacy` - Natural language processing

## Usage

### Basic Publisher Discovery

```python
from publisher_discovery import PublisherDiscoverySystem

# Initialize system
discovery = PublisherDiscoverySystem()

# Discover publishers
results = await discovery.discover_publishers(["example.com"])
```

### AI-Enhanced Discovery

```python
from ai_enhanced_publisher_discovery import AIEnhancedPublisherDiscovery

# Initialize with AI capabilities
ai_discovery = AIEnhancedPublisherDiscovery()

# Analyze with AI
analysis = await ai_discovery.analyze_content_with_ai(content, url)
```

## API Reference

See the detailed documentation in:
- `ai_integration_overview.md` - AI features and capabilities
- `publisher_discovery_system.md` - Core system documentation
- `setup_guide.md` - Detailed setup instructions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is open source. Please see the LICENSE file for details.

## Support

For questions or issues, please open a GitHub issue or contact the maintainer.
