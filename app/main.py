import os
import time
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from pdf_processing_module import PDFProcessor
import ollama

load_dotenv()

# Initialize components
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
processor = PDFProcessor()

# Load or create vector store
vector_store_path = r"C:\HDFC\vector_store\faiss_index"
# if not os.path.exists("vector_store_path"):
#     print("Building vector store...")
#     processor.process_pdfs()
vector_store = processor.load_vector_store()

def generate_response(query):
    # Search with score threshold
    docs = vector_store.similarity_search_with_score(query, k=3)
    context = "\n\n".join([doc[0].page_content for doc in docs if doc[1] < 0.8])  # Confidence threshold
    
    if not context:
        return "I couldn't find relevant information in the documents."
    
    prompt = f"""You are a financial expert. Answer strictly using only this context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    try:
        response = ollama.generate(
            model='llama3.1:8b',
            prompt=prompt,
            options={'num_predict': 500}
        )
        return response['response'].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.event("app_mention")
def handle_mention(event, say):
    try:
        query = event['text'].split('>', 1)[1].strip()
        say(f"<@{event['user']}> Processing your query...")
        
        # Get response
        response = generate_response(query)
        
        # Format sources
        final_response = f"{response}\n\n_Sources: Documents 1-3_"
        say(final_response)
        
    except Exception as e:
        say(f"Sorry <@{event['user']}>, encountered an error: {str(e)}")

@app.command("/query")
def handle_query(ack, respond, command):
    ack()
    try:
        query = command['text']
        if not query:
            respond("Please provide a question after the command")
            return
            
        respond(f"Processing your query: '{query}'...")
        response = generate_response(query)
        respond(response)
        
    except Exception as e:
        respond(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()