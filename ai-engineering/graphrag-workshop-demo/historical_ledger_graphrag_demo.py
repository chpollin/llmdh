# historical_ledger_graphrag_demo.py
"""
Digital Humanities GraphRAG Demo
Using 1828-1829 Wheaton Day Book ledger entries
Works without APOC plugin
"""

import os
import sys
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain.docstore.document import Document

class HistoricalGraphRAG:
    """GraphRAG system for historical document analysis"""
    
    def __init__(self, openai_key: str, neo4j_url: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        """Initialize the GraphRAG system"""
        
        # Set OpenAI key
        os.environ["OPENAI_API_KEY"] = openai_key
        
        # Store connection parameters
        self.neo4j_url = neo4j_url
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Initialize components
        self.graph = None
        self.llm = None
        self.transformer = None
        self.chain = None
        
    def connect_neo4j(self) -> bool:
        """Establish Neo4j connection without APOC dependency"""
        try:
            # Try connection without APOC
            self.graph = Neo4jGraph(
                url=self.neo4j_url,
                username=self.neo4j_user,
                password=self.neo4j_password,
                refresh_schema=False,  # Skip APOC-dependent schema refresh
                enhanced_schema=False  # Don't use enhanced schema features
            )
            
            # Test connection with simple query
            test_result = self.graph.query("RETURN 1 as test")
            if test_result[0]['test'] == 1:
                return True
            return False
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def initialize_llm(self, model: str = "gpt-4-turbo-preview"):
        """Initialize the language model"""
        self.llm = ChatOpenAI(
            temperature=0,
            model=model
        )
        
    def setup_transformer(self):
        """Configure the graph transformer for historical documents"""
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=[
                "Person", 
                "Item", 
                "Money", 
                "Work", 
                "Animal",
                "Date",
                "Location"
            ],
            allowed_relationships=[
                "BOUGHT", 
                "SOLD", 
                "PAID", 
                "WORKED_FOR", 
                "RECEIVED",
                "OWES",
                "SETTLED"
            ],
            additional_instructions="""
            Extract transactions from this 1828 historical ledger.
            - Preserve historical names exactly as written
            - Dates should be extracted when mentioned
            - Money amounts should include currency/units when specified
            - Work exchanges are also transactions
            """
        )
        
    def clear_graph(self):
        """Clear all existing data from the graph"""
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            return True
        except Exception as e:
            print(f"Error clearing graph: {e}")
            return False
    
    def load_ledger_data_manual(self, ledger_text: str) -> Dict[str, int]:
        """Load ledger data using manual Cypher queries instead of APOC"""
        
        # Create document and transform
        doc = Document(page_content=ledger_text)
        graph_documents = self.transformer.convert_to_graph_documents([doc])
        
        # Manually insert nodes and relationships
        node_count = 0
        rel_count = 0
        
        for graph_doc in graph_documents:
            # Insert nodes
            for node in graph_doc.nodes:
                try:
                    # Create node with MERGE to avoid duplicates
                    query = f"""
                    MERGE (n:{node.type} {{id: $id}})
                    RETURN n
                    """
                    self.graph.query(query, {"id": node.id})
                    node_count += 1
                except Exception as e:
                    print(f"Error creating node {node.id}: {e}")
            
            # Insert relationships
            for rel in graph_doc.relationships:
                try:
                    query = f"""
                    MATCH (a {{id: $source_id}})
                    MATCH (b {{id: $target_id}})
                    MERGE (a)-[r:{rel.type}]->(b)
                    RETURN r
                    """
                    self.graph.query(query, {
                        "source_id": rel.source.id,
                        "target_id": rel.target.id
                    })
                    rel_count += 1
                except Exception as e:
                    print(f"Error creating relationship: {e}")
        
        # Get actual counts from database
        actual_nodes = self.graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        actual_rels = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        
        return {"nodes": actual_nodes, "relationships": actual_rels}
    
    def setup_qa_chain(self):
        """Initialize the question-answering chain"""
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=False,  # Set to True to see generated Cypher queries
            allow_dangerous_requests=True,
            top_k=10  # Limit results for better performance
        )
    
    def query(self, question: str) -> str:
        """Query the graph with natural language"""
        try:
            result = self.chain.invoke({"query": question})
            return result['result']
        except Exception as e:
            return f"Query error: {str(e)}"
    
    def get_people(self) -> List[str]:
        """Get all people mentioned in the ledger"""
        try:
            people = self.graph.query("""
                MATCH (p:Person)
                RETURN DISTINCT p.id as name
                ORDER BY name
            """)
            return [p['name'] for p in people]
        except:
            return []
    
    def get_transactions(self) -> List[Dict[str, Any]]:
        """Get all transactions from the graph"""
        try:
            transactions = self.graph.query("""
                MATCH (p1)-[r:BOUGHT|SOLD|PAID|RECEIVED]->(p2)
                RETURN p1.id as from_person, 
                       type(r) as transaction_type, 
                       p2.id as to_person
                LIMIT 20
            """)
            return transactions
        except:
            return []

def load_configuration():
    """Load configuration from .env file"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get configuration
    config = {
        'openai_key': os.getenv('OPENAI_API_KEY'),
        'neo4j_url': os.getenv('NEO4J_URL', 'bolt://localhost:7687'),
        'neo4j_user': os.getenv('NEO4J_USER', 'neo4j'),
        'neo4j_password': os.getenv('NEO4J_PASSWORD', 'password'),
        'model': os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
    }
    
    # Validate required settings
    if not config['openai_key']:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        print("\nPlease create a .env file with:")
        print("OPENAI_API_KEY=your-actual-key-here")
        sys.exit(1)
    
    # Mask API key for display
    masked_key = config['openai_key'][:8] + "..." + config['openai_key'][-4:]
    print(f"   Using API key: {masked_key}")
    
    return config

def main():
    """Main demonstration function"""
    
    print("=" * 60)
    print("HISTORICAL LEDGER GRAPHRAG DEMO")
    print("Wheaton Day Book (1828-1829)")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_configuration()
    print("   ✓ Configuration loaded")
    
    # Initialize system
    system = HistoricalGraphRAG(
        openai_key=config['openai_key'],
        neo4j_url=config['neo4j_url'],
        neo4j_user=config['neo4j_user'],
        neo4j_password=config['neo4j_password']
    )
    
    # Connect to Neo4j
    print("\n2. Connecting to Neo4j...")
    if not system.connect_neo4j():
        print("   ✗ Failed to connect to Neo4j")
        print("\nTroubleshooting:")
        print("1. Check if Neo4j is running: docker ps")
        print("2. Verify connection details in .env file")
        print("3. Check Neo4j logs: docker logs neo4j")
        return
    print("   ✓ Connected successfully")
    
    # Initialize LLM
    print("\n3. Initializing language model...")
    system.initialize_llm(model=config['model'])
    system.setup_transformer()
    print(f"   ✓ LLM configured (model: {config['model']})")
    
    # Clear and load data
    print("\n4. Loading historical ledger entries...")
    system.clear_graph()
    
    # Historical ledger text (simplified for demo)
    ledger_text = """
    From Laban Morey Wheaton's Day Book, September-October 1828:
    
    Monday September 15th, 1828:
    Derius Drake received one ax delivered by Puffer, value $1.50
    Drake provided cutting two sticks as payment to Puffer
    Pliny Puffer credited for delivering ax to Drake
    
    Monday September 22nd, 1828:
    Asa Danforth worked one day with oxen and wheels
    Derius Drake paid order upon T. Smith for 50 cents
    
    October 3rd, 1828:
    Thomas Danforth bought 1 pound cut nails for 8 cents and 1 bushel rye for 90 cents
    John Deane paid cash $1.00
    Nathaniel Lincoln paid cash 50 cents
    
    October 27th, 1828:
    Holmes Richmond bought a heifer and calf for $20.00
    John Deane paid additional cash $1.00
    Oliver Clapp credited for 16 days work at $16.00
    """
    
    try:
        # Use manual loading to avoid APOC dependency
        stats = system.load_ledger_data_manual(ledger_text)
        print(f"   ✓ Created {stats['nodes']} nodes and {stats['relationships']} relationships")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Query the data
    print("\n5. Querying the historical network...")
    print("-" * 40)
    
    system.setup_qa_chain()
    
    questions = [
        "Who was involved in transactions in 1828?",
        "What items were purchased?",
        "Who paid with cash?",
        "What was the most expensive transaction?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        answer = system.query(question)
        print(f"A{i}: {answer}")
    
    # Direct graph queries
    print("\n6. Direct graph analysis:")
    print("-" * 40)
    
    people = system.get_people()
    if people:
        print("\nPeople in the ledger:")
        for person in people[:10]:  # Limit to first 10
            print(f"   • {person}")
    
    transactions = system.get_transactions()
    if transactions:
        print("\nSample transactions:")
        for trans in transactions[:5]:  # Show first 5
            print(f"   • {trans['from_person']} {trans['transaction_type']} {trans['to_person']}")
    
    # Visualization instructions
    print("\n7. Visualization:")
    print("-" * 40)
    print("Open Neo4j Browser at http://localhost:7474")
    print(f"Username: {config['neo4j_user']}, Password: {config['neo4j_password']}")
    print("\nUseful queries to visualize:")
    print("1. All relationships: MATCH (n)-[r]->(m) RETURN n, r, m")
    print("2. Person network: MATCH (p:Person) RETURN p")
    print("3. Transaction flow: MATCH path=(p1:Person)-[*1..2]->(p2:Person) RETURN path")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("This technique scales to thousands of ledger pages!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("\nPlease check:")
        print("1. Docker is running")
        print("2. Neo4j container is active") 
        print("3. OpenAI API key is valid in .env file")
        print("4. All packages are installed:")
        print("   pip install python-dotenv langchain-neo4j langchain-openai langchain-experimental")