# app/hybrid_retriever.py
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple
from langchain.schema import Document
from langchain.vectorstores import FAISS

class HybridRetriever:
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self._initialize_bm25()
        
    def _initialize_bm25(self):
        """Initialize BM25 with document corpus"""
        self.documents = list(self.vector_store.docstore._dict.values())
        self.bm25 = BM25Okapi([
            doc.page_content.lower().split() 
            for doc in self.documents
        ])
        
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5, 
                     min_score: float = 0.6) -> List[Tuple[Document, float]]:
        """Perform hybrid search with score thresholding"""
        # Semantic search
        vector_results = self.vector_store.similarity_search_with_relevance_scores(
            query, 
            k=top_k*2
        )
        
        # Keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Combine results
        combined = []
        for (doc, vector_score), bm25_score in zip(vector_results, bm25_scores):
            if vector_score < 0.2:  # Filter low vector scores
                continue
                
            combined_score = (alpha * vector_score) + ((1 - alpha) * bm25_score)
            combined.append((doc, combined_score))
        
        # Sort and filter
        combined.sort(key=lambda x: x[1], reverse=True)
        filtered = [item for item in combined if item[1] >= min_score]
        
        return filtered[:top_k]