# RAG System

Retrieval-Augmented Generation system using news data with LLM integration.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your API key to .env
```

## Usage

```bash
python rag_script.py
```

## Requirements

- CSV file: `news_data_dedup.csv` with columns: `title`, `description`, `published_at`, `url`
- API key: OpenAI or Anthropic (set in `.env` file)

## API Key Configuration

Create `.env` file:
```
OPENAI_API_KEY=your-key-here
```

## Programmatic Usage

```python
from rag_script import ProductionRAG

rag = ProductionRAG('news_data_dedup.csv')
result = rag.query("Your question here")
print(result['response'])
```