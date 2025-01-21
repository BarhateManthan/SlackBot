import os
import time
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from pdf_processing_module import EnhancedPDFProcessor
import ollama
from typing import List, Tuple

load_dotenv()

class FinanceAgent:
    def __init__(self):
        self.app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
        self.processor = EnhancedPDFProcessor()
        self.vector_store_path = r"C:\HDFC\vector_store\faiss_index"
        self.setup_handlers()
        
        # Load vector store
        self.vector_store = self.processor.load_vector_store()

    def generate_response(self, query: str) -> str:
        # Perform hybrid search
        results = self.processor.hybrid_search(query, k=3)
        
        if not results:
            return "I couldn't find relevant information in the documents."
        
        # Create enhanced context with confidence scores
        context_parts = []
        for doc, score in results:
            confidence = f"{score:.2%}"
            context_parts.append(f"[Confidence: {confidence}]\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a financial expert. Answer based on the following context, ordered by relevance:
        {context}
        
        Important:
        - Only use information from the provided context
        - If the context doesn't contain enough information, say so
        - Be specific and cite numbers/facts when available
        - Format the response in a clear, structured way
        
        Question: {query}
        
        Answer:"""
        
        try:
            response = ollama.generate(
                model='llama3.1:8b',
                prompt=prompt,
                options={
                    'num_predict': 500,
                    'temperature': 0.7,
                    'top_p': 0.9
                }
            )
            
            return self._format_response(response['response'].strip(), results)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _format_response(self, response: str, results: List[Tuple]) -> str:
        # Add source information
        source_info = "\n\n*Sources:*"
        for i, (doc, score) in enumerate(results, 1):
            confidence = f"{score:.2%}"
            if 'page' in doc.metadata:
                source_info += f"\n{i}. Page {doc.metadata['page']} (Confidence: {confidence})"
            else:
                source_info += f"\n{i}. Document section {i} (Confidence: {confidence})"
        
        return f"{response}{source_info}"

    def setup_handlers(self):
        @self.app.event("app_mention")
        def handle_mention(event, say):
            try:
                query = event['text'].split('>', 1)[1].strip()
                say(f"<@{event['user']}> Processing your query...")
                
                response = self.generate_response(query)
                say(response)
                
            except Exception as e:
                say(f"Sorry <@{event['user']}>, encountered an error: {str(e)}")

        @self.app.command("/query")
        def handle_query(ack, respond, command):
            ack()
            try:
                query = command['text']
                if not query:
                    respond("Please provide a question after the command")
                    return
                    
                respond(f"Processing your query: '{query}'...")
                response = self.generate_response(query)
                respond(response)
                
            except Exception as e:
                respond(f"Error processing request: {str(e)}")

    def start(self):
        SocketModeHandler(self.app, os.environ["SLACK_APP_TOKEN"]).start()

if __name__ == "__main__":
    agent = FinanceAgent()
    agent.start()