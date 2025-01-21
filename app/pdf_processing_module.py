import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class PDFProcessor:
    def __init__(self):
        self.vector_store_path = r"C:\HDFC\vector_store\faiss_index"
        # Use SentenceTransformer embeddings (All-MiniLM-L6-v2)
        self.embeddings = self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def process_pdfs(self):
        kb_path = r"C:\HDFC\knowledge_base"
        if not os.path.exists(kb_path):
            # os.mkdir("knowledge_base")
            print("Add your PDFs to the 'knowledge_base' directory and re-run.")
            exit()

        docs = []
        for pdf_file in os.listdir(kb_path):
            if pdf_file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(kb_path, pdf_file))
                documents = loader.load()
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = splitter.split_documents(documents)
                docs.extend(split_docs)

        vector_store = FAISS.from_documents(docs, self.embeddings)
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        vector_store.save_local(self.vector_store_path)

    def load_vector_store(self):
        if not os.path.exists(self.vector_store_path):
            raise ValueError("Vector store not found. Process PDFs first.")
        return FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
