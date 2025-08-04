#!/usr/bin/env python3
"""
Production RAG Script
Retrieval-Augmented Generation using news data with real LLM calls and secure API key handling.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import os
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Using system environment variables only.")

class ProductionRAG:
    def __init__(self, csv_path: str, api_key: Optional[str] = None):
        """Initialize RAG system with news data and LLM configuration."""
        print("Initializing RAG system...")
        
        # Load and prepare data
        self.df = self._load_data(csv_path)
        print(f"Loaded {len(self.df)} articles from {csv_path}")
        
        # Build search index
        print("Building search index...")
        self.corpus = self._build_corpus()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        print("Search index ready!")
        
        # Configure LLM API
        self._setup_llm_config(api_key)
        print(f"LLM provider: {self.provider}")
        print("RAG system ready!\n")
    
    def _load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate CSV data."""
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Validate required columns
        required_cols = ['title', 'description', 'published_at', 'url']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean data
        for col in ['title', 'description']:
            df[col] = df[col].fillna('')
        
        return df
    
    def _build_corpus(self) -> List[str]:
        """Build search corpus from title and description."""
        return (self.df['title'] + ' ' + self.df['description']).tolist()
    
    def _setup_llm_config(self, api_key: Optional[str] = None):
        """Setup LLM configuration with secure API key handling."""
        # Try multiple sources for API key
        self.api_key = (
            api_key or 
            os.getenv('OPENAI_API_KEY') or 
            os.getenv('ANTHROPIC_API_KEY') or
            os.getenv('LLM_API_KEY')  # Generic fallback
        )
        
        if not self.api_key:
            raise ValueError(
                "API key required. Choose one option:\n"
                "1. Set environment variable: export OPENAI_API_KEY='your-key'\n"
                "2. Create .env file with: OPENAI_API_KEY=your-key\n"
                "3. Pass api_key parameter: ProductionRAG(csv_path, api_key='your-key')\n"
                "4. Set generic: export LLM_API_KEY='your-key'"
            )
        
        # Determine provider based on key format or explicit environment variable
        if (os.getenv('ANTHROPIC_API_KEY') or 
            (self.api_key and self.api_key.startswith('sk-ant')) or
            os.getenv('LLM_PROVIDER') == 'anthropic'):
            self._setup_anthropic()
        else:
            self._setup_openai()
    
    def _setup_anthropic(self):
        """Configure Anthropic Claude API."""
        self.provider = 'anthropic'
        self.api_url = 'https://api.anthropic.com/v1/messages'
        self.headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        self.model = 'claude-3-sonnet-20240229'
    
    def _setup_openai(self):
        """Configure OpenAI GPT API."""
        self.provider = 'openai'
        self.api_url = 'https://api.openai.com/v1/chat/completions'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        # Use GPT-4 if available, fallback to GPT-3.5
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    def retrieve(self, query: str, top_k: int = 3, relevance_threshold: float = 0.05) -> Tuple[List[int], List[float]]:
        """Retrieve most relevant document indices with their similarity scores."""
        if not query.strip():
            return [], []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get all documents above relevance threshold
        relevant_pairs = [(idx, similarities[idx]) for idx in range(len(similarities)) 
                         if similarities[idx] > relevance_threshold]
        
        # Sort by similarity score (highest first)
        relevant_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k most relevant
        top_pairs = relevant_pairs[:top_k]
        
        indices = [pair[0] for pair in top_pairs]
        scores = [pair[1] for pair in top_pairs]
        
        return indices, scores
    
    def print_retrieved_documents(self, indices: List[int], scores: List[float] = None):
        """Print retrieved documents with titles for easy CSV lookup."""
        if not indices:
            print("❌ No relevant documents found above relevance threshold")
            return
        
        print(f"📋 Retrieved Documents (Total: {len(indices)}):")
        print("-" * 80)
        
        for i, idx in enumerate(indices, 1):
            try:
                doc = self.df.iloc[idx]
                score_info = f" | Similarity: {scores[i-1]:.3f}" if scores else ""
                print(f"[{i}] CSV Row: {idx}{score_info}")
                print(f"    Title: {doc['title']}")
                print(f"    Published: {doc['published_at']}")
                print()
            except IndexError:
                print(f"[{i}] Error: Document index {idx} not found")
        
        print("-" * 80)
    
    def format_documents(self, indices: List[int], scores: List[float] = None) -> str:
        """Format retrieved documents for LLM context."""
        if not indices:
            return "No relevant documents found in the news database."
        
        formatted_docs = []
        for i, idx in enumerate(indices):
            try:
                doc = self.df.iloc[idx]
                score_info = f" (similarity: {scores[i]:.3f})" if scores else ""
                formatted_doc = f"""[Article {i+1}]{score_info}
Title: {doc['title']}
Summary: {doc['description']}
Published: {doc['published_at']}
Source: {doc['url']}"""
                formatted_docs.append(formatted_doc)
            except IndexError:
                formatted_docs.append(f"[Article {i+1}] Error: Document not found")
        
        return "\n\n".join(formatted_docs)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create optimized prompt for the LLM."""
        return f"""You are a helpful assistant with access to recent news from 2024. Answer the user's question using both your general knowledge and the provided recent news context when relevant.

If the news context is relevant to the question, incorporate specific details from it. If the context isn't directly relevant, answer based on your general knowledge but mention that no recent specific news was found on this topic.

User Question: {query}

Recent News Context (2024):
{context}

Please provide a comprehensive answer:"""
    
    def _create_payload(self, prompt: str) -> Dict:
        """Create API payload based on provider."""
        if self.provider == 'anthropic':
            return {
                "model": self.model,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        else:  # OpenAI
            return {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7
            }
    
    def _call_llm(self, payload: Dict) -> str:
        """Make secure API call to LLM provider."""
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            
            # Handle HTTP errors
            if response.status_code == 401:
                return "Authentication Error: Invalid API key"
            elif response.status_code == 429:
                return "Rate Limit Error: Too many requests. Please try again later."
            elif response.status_code >= 400:
                return f"API Error ({response.status_code}): {response.text}"
            
            response.raise_for_status()
            data = response.json()
            
            # Extract response based on provider
            if self.provider == 'anthropic':
                return data['content'][0]['text']
            else:  # OpenAI
                return data['choices'][0]['message']['content']
                
        except requests.exceptions.Timeout:
            return "Request timeout. Please try again."
        except requests.exceptions.ConnectionError:
            return "Connection error. Please check your internet connection."
        except requests.exceptions.RequestException as e:
            return f"Request failed: {str(e)}"
        except (KeyError, IndexError, TypeError) as e:
            return f"Response parsing error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def query(self, question: str, top_k: int = 3, use_rag: bool = True, verbose: bool = False, relevance_threshold: float = 0.05) -> Dict[str, any]:
        """Execute RAG query and return structured response."""
        if not question.strip():
            return {
                "query": question,
                "error": "Empty query provided",
                "response": "Please provide a valid question."
            }
        
        result = {
            "query": question,
            "provider": self.provider,
            "model": self.model,
            "rag_enabled": use_rag,
            "retrieved_docs_count": 0,
            "relevance_threshold": relevance_threshold
        }
        
        try:
            if use_rag:
                # Retrieve relevant documents with scores
                if verbose:
                    print(f"Searching for documents with relevance > {relevance_threshold:.2%}...")
                
                relevant_indices, scores = self.retrieve(question, top_k, relevance_threshold)
                result["retrieved_docs_count"] = len(relevant_indices)
                result["retrieved_indices"] = relevant_indices
                result["similarity_scores"] = scores
                
                # Print document information for CSV lookup
                if verbose:
                    self.print_retrieved_documents(relevant_indices, scores)
                
                context = self.format_documents(relevant_indices, scores)
                result["context"] = context
                
                # Create prompt with context
                prompt = self._create_prompt(question, context)
            else:
                # Query without RAG context
                prompt = f"Please answer this question: {question}"
                result["context"] = "No RAG context used"
            
            # Create payload and call LLM
            payload = self._create_payload(prompt)
            
            if verbose:
                print("Calling LLM...")
            
            response = self._call_llm(payload)
            result["response"] = response
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            result["response"] = f"System error: {str(e)}"
            result["success"] = False
        
        return result
    
    def compare_responses(self, question: str, top_k: int = 3, relevance_threshold: float = 0.05) -> Dict[str, any]:
        """Compare RAG vs non-RAG responses for the same question."""
        print(f"Comparing responses for: {question}\n")
        
        # Get both responses
        rag_result = self.query(question, top_k=top_k, use_rag=True, relevance_threshold=relevance_threshold)
        no_rag_result = self.query(question, use_rag=False)
        
        return {
            "query": question,
            "with_rag": rag_result,
            "without_rag": no_rag_result,
            "comparison": {
                "rag_found_docs": rag_result.get("retrieved_docs_count", 0),
                "relevance_threshold": relevance_threshold,
                "both_successful": rag_result.get("success", False) and no_rag_result.get("success", False)
            }
        }

def main():
    """Execute production RAG example."""
    try:
        # Initialize RAG system
        rag = ProductionRAG('news_data_dedup.csv')
        
        # Single demo query
        demo_query = "What are the latest developments in artificial intelligence in 2024?"
        
        print("=== RAG SYSTEM DEMONSTRATION ===\n")
        print(f"Demo Query: {demo_query}")
        print("-" * 60)
        
        # Get RAG-enhanced response
        result = rag.query(demo_query, top_k=5, use_rag=True, verbose=True, relevance_threshold=0.05)
        
        if result.get("success"):
            print(f"\n🤖 LLM Response ({result['provider']} - {result['model']}):")
            print(result['response'])
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*80 + "\n")
        
        # Interactive mode
        print("Interactive Mode - Enter your own queries (type 'quit' to exit):")
        while True:
            try:
                user_query = input("\nYour question: ").strip()
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                if user_query:
                    result = rag.query(user_query, top_k=5, use_rag=True, verbose=True, relevance_threshold=0.05)
                    print(f"\n🤖 Response: {result['response']}")
            except KeyboardInterrupt:
                break
        
        print("\nThanks for using the RAG system!")
        
    except Exception as e:
        print(f"❌ System Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure news_data_dedup.csv is in the current directory")
        print("2. Check that your API key is properly set")
        print("3. Verify internet connection for API calls")

if __name__ == "__main__":
    main()