# Create test_setup.py
def test_imports():
    tests = []
    
    # Test core libraries
    try:
        import aiohttp, pandas, sqlite3
        from bs4 import BeautifulSoup
        tests.append("✅ Core libraries")
    except ImportError as e:
        tests.append(f"❌ Core libraries: {e}")
    
    # Test AI libraries
    try:
        import openai, anthropic, tiktoken
        tests.append("✅ LLM libraries")
    except ImportError as e:
        tests.append(f"❌ LLM libraries: {e}")
    
    # Test ML libraries
    try:
        import transformers, sklearn, numpy
        import torch
        tests.append("✅ ML libraries")
    except ImportError as e:
        tests.append(f"❌ ML libraries: {e}")
    
    # Test spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        tests.append("✅ spaCy + model")
    except Exception as e:
        tests.append(f"❌ spaCy: {e}")
    
    return tests

if __name__ == "__main__":
    for test in test_imports():
        print(test)
