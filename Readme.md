# Financial Document Query Bot

This project implements a Slack bot that can process and query financial documents in PDF format. The bot uses advanced natural language processing techniques to provide accurate and context-aware responses to user queries.

## Features

- **PDF Processing**: Extracts and processes text from PDF documents.
- **Hybrid Search**: Combines semantic and keyword search for better results.
- **Slack Integration**: Responds to user queries directly in Slack.
- **Ollama Integration**: Uses a local language model for generating responses.

## Project Setup & Environment

1. **Create project directory**
2. **Clone the repository**
   ```bash
   git clone <https://github.com/BarhateManthan/SlackBot.git>
   ```
3. **Set up Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. **Initialize Git repository**
   ```bash
   git init
   ```
5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Slack Bot Setup

1. **Create new Slack app** at [api.slack.com](https://api.slack.com/apps)
2. **Configure OAuth & Permissions**:
   - Add `chat:write`
   - Add `app_mentions:read`
   - Add `im:history`
3. **Install app to workspace**
4. **Enable Socket Mode**
5. **Enable Event Subscription** to read messages
6. **Generate bot and app tokens**
7. **Store credentials in `.env` file**

## PDF Processing System

1. **Implement PDF loader** using `PyPDF2`
2. **Create text splitter** for document chunking
3. **Set up embedding model** (e.g., `all-MiniLM-L6-v2`)
4. **Implement FAISS vector store**
5. **Create document ingestion pipeline**:
   - PDF → text extraction
   - Text chunking (500-800 characters)
   - Vector embeddings
   - FAISS index storage

## Ollama Integration and Pipeline

1. **Install Ollama locally**
2. **Download preferred LLM** (e.g., `llama3.1:8b`)
3. **Create query pipeline**:
   - User question → vector similarity search
   - Context assembly from top 3 matches
   - Prompt engineering template:
     ```plaintext
     Answer based ONLY on this context:
     {context}

     Question: {question}
     ```
4. **Implement response validation** to prevent hallucinations

## File Overview

### `pdf_processing_module.py`

This module handles the processing of PDF documents. It includes:

- **Text extraction** from PDFs
- **Text preprocessing** including tokenization and stopword removal
- **Chunking** of text into manageable pieces
- **Vector embeddings** using HuggingFace models
- **Hybrid search** combining semantic and keyword search

### `main.py`

This is the main entry point for the Slack bot. It includes:

- **Slack bot initialization** and event handling
- **Query processing** using the `EnhancedPDFProcessor`
- **Response generation** using the Ollama language model
- **Slack command handlers** for user interactions

## Usage

1. **Start the bot**:
   ```bash
   python main.py
   ```
2. **Interact with the bot** in Slack:
   - Mention the bot in a message to ask a question
   - Use the `/query` command followed by your question
     ( note /query needs to be added in commands config of the bot ) 
## Requirements

- Python 3.8+
- Slack workspace with app permissions
- Ollama installed locally
- Required Python packages (see `requirements.txt`)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For more details, please refer to the source code and inline documentation.
