# GraphRAG Tutorial: Historical Documents to Knowledge Graphs

## Overview

GraphRAG combines knowledge graphs with Large Language Models to answer complex relationship queries on unstructured text. This tutorial demonstrates the technique using 1828-1829 historical ledger data.

## Prerequisites

- Docker Desktop running
- Python 3.10+
- OpenAI API key
- 4GB RAM minimum

## Installation

### 1. Start Neo4j Database
Run Neo4j in Docker without additional plugins for simplicity:
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

Verify at http://localhost:7474 (credentials: neo4j/password)

### 2. Install Python Dependencies
```bash
pip install langchain langchain-neo4j langchain-openai langchain-experimental python-dotenv
```

### 3. Configure Environment
Create `.env` file:
```
OPENAI_API_KEY=sk-proj-your-key-here
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## How It Works

### Process Flow

1. **Text Input**: Historical ledger entries describing transactions, people, and dates
2. **Entity Extraction**: LLM identifies entities (persons, items, money) and relationships (bought, sold, paid)
3. **Graph Creation**: Entities become nodes, relationships become edges in Neo4j
4. **Natural Language Queries**: Questions convert to Cypher queries, traverse the graph, return answers

### Key Components

**Neo4j Graph Database** stores the knowledge graph structure. Unlike relational databases, it optimizes for relationship traversal.

**LangChain Framework** orchestrates the LLM interactions, managing both entity extraction and query generation.

**LLMGraphTransformer** processes documents to extract structured information. It uses predefined entity and relationship types to guide extraction.

**GraphCypherQAChain** converts natural language questions into Cypher queries, executes them, and formats responses.

## Implementation Steps

The system initializes by loading configuration from environment variables and establishing database connection. It clears any existing data to ensure clean state.

The transformer processes text documents, extracting entities and relationships based on configured types. For historical ledgers, this includes Person, Item, Money, and Date entities with relationships like BOUGHT, SOLD, and PAID.

Manual Cypher queries create nodes and relationships, avoiding APOC dependencies. Each entity uses MERGE to prevent duplicates, while relationships connect existing nodes.

The QA chain handles natural language queries by generating appropriate Cypher, executing against the graph, and formatting results in natural language.

## Example Queries

Questions that work well:
- "Who was involved in transactions?"
- "What items were purchased?"
- "Who paid with cash versus goods?"
- "What was the most expensive transaction?"

These demonstrate relationship traversal impossible with keyword search alone.

## Troubleshooting

**Docker Issues**: Ensure virtualization is enabled in BIOS. Check container status with `docker ps`.

**Connection Failures**: Verify Neo4j is running and accessible. Test with Neo4j Browser before running Python code.

**API Errors**: Confirm OpenAI key is valid and has available credits. Check rate limits if receiving 429 errors.

**APOC Dependencies**: The tutorial avoids APOC by using manual Cypher queries. Set `refresh_schema=False` in connections.

## Workshop Usage (15 minutes)

1. **Introduction** (2 min): Show original ledger, explain knowledge graphs
2. **Demonstration** (8 min): Run script, show extraction and queries
3. **Visualization** (3 min): Display graph in Neo4j Browser
4. **Discussion** (2 min): Scaling possibilities, research applications

## Digital Humanities Applications

Historical ledgers reveal economic networks and social relationships. The graph structure preserves community connections lost in traditional databases.

Court records benefit from tracking legal relationships and precedents. Literary correspondence maps intellectual networks across time and geography.

The technique scales from single documents to entire archives, enabling macro-level analysis while preserving micro-level detail.

## Production Considerations

**Data Quality**: Historical documents require OCR correction and entity normalization. Names and dates need standardization.

**Performance**: Process documents in batches. Create indexes on frequently queried properties. Cache extracted graphs to avoid reprocessing.

**Costs**: Estimate $0.01-0.03 per page with GPT-4. Consider GPT-3.5 for initial testing. Monitor API usage carefully.

## Limitations

- Extraction accuracy depends on text quality and LLM capabilities
- Historical terminology may require specialized prompts
- Large corpora need chunking strategies
- Graph complexity grows rapidly with document count
