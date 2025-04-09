import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma"

def load_and_index_docs():
    documents = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(os.path.join(DATA_DIR, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print("✅ Documentos indexados correctamente.")

if __name__ == "__main__":
    load_and_index_docs()
