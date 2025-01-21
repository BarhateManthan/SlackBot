import os
from typing import List, Dict, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import numpy as np
from collections import Counter
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class EnhancedPDFProcessor:
    def __init__(self):
        self.vector_store_path = r"C:\HDFC\vector_store\faiss_index"
        self.tfidf_path = r"C:\HDFC\vector_store\tfidf_vectorizer.pkl"
        
        # Download all required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK resources...")
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('maxent_ne_chunker')
            nltk.download('words')
        
        # Using a more powerful embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        # Cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing."""
        # Basic cleaning
        text = text.lower()
        # Remove special characters while preserving spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        try:
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords and short tokens
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Warning: Error in tokenization, falling back to basic splitting: {str(e)}")
            # Fallback to basic splitting if NLTK tokenization fails
            words = text.split()
            return ' '.join([w for w in words if w not in self.stop_words and len(w) > 2])

    def create_intelligent_chunks(self, text: str) -> List[str]:
        """Create chunks based on semantic boundaries."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        
        # Merge short chunks and split long ones
        processed_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) < 1000:
                current_chunk += " " + chunk
            else:
                if current_chunk:
                    processed_chunks.append(current_chunk.strip())
                current_chunk = chunk
                
        if current_chunk:
            processed_chunks.append(current_chunk.strip())
            
        return processed_chunks

    def process_pdfs(self):
        kb_path = r"C:\HDFC\knowledge_base"
        if not os.path.exists(kb_path):
            raise ValueError("Knowledge base directory not found")

        all_docs = []
        all_texts = []

        # Process each PDF
        for pdf_file in os.listdir(kb_path):
            if pdf_file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(kb_path, pdf_file))
                documents = loader.load()
                
                for doc in documents:
                    # Preprocess the text
                    processed_text = self.preprocess_text(doc.page_content)
                    # Create intelligent chunks
                    chunks = self.create_intelligent_chunks(processed_text)
                    
                    for chunk in chunks:
                        # Store original text for TF-IDF
                        all_texts.append(chunk)
                        # Create document with metadata
                        doc.page_content = chunk
                        doc.metadata['chunk_id'] = len(all_docs)
                        all_docs.append(doc)

        # Create and save vector store
        vector_store = FAISS.from_documents(all_docs, self.embeddings)
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        vector_store.save_local(self.vector_store_path)

        # Create and save TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=10000)
        tfidf.fit(all_texts)
        import joblib
        joblib.dump(tfidf, self.tfidf_path)

    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Implement hybrid search combining semantic and keyword search."""
        vector_store = self.load_vector_store()
        
        # Semantic Search
        semantic_results = vector_store.similarity_search_with_score(query, k=k*2)
        
        # Keyword Search using TF-IDF
        import joblib
        tfidf = joblib.load(self.tfidf_path)
        query_vec = tfidf.transform([query])
        doc_vectors = tfidf.transform([doc[0].page_content for doc in semantic_results])
        keyword_scores = (doc_vectors @ query_vec.T).toarray().flatten()
        
        # Combine results
        combined_results = []
        for (doc, semantic_score), keyword_score in zip(semantic_results, keyword_scores):
            # Normalize scores
            norm_semantic_score = 1 - (semantic_score / 2)  # Convert distance to similarity
            norm_keyword_score = keyword_score
            
            # Weighted combination
            final_score = (0.7 * norm_semantic_score) + (0.3 * norm_keyword_score)
            combined_results.append((doc, final_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Re-rank top results using cross-encoder
        if len(combined_results) > 0:
            cross_encoder_inputs = [(query, doc[0].page_content) for doc in combined_results[:k*2]]
            cross_encoder_scores = self.cross_encoder.predict(cross_encoder_inputs)
            
            # Final ranking
            final_results = [(doc[0], score) for doc, score in zip(combined_results[:k*2], cross_encoder_scores)]
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            return final_results[:k]
        
        return combined_results[:k]

    def load_vector_store(self):
        if not os.path.exists(self.vector_store_path):
            raise ValueError("Vector store not found. Process PDFs first.")
        return FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)