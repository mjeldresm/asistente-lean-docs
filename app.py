from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()

app = FastAPI()

# Inicializar componentes
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectordb = Chroma(persist_directory="chroma", embedding_function=embeddings)
retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")), retriever=retriever)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(item: Question):
    response = qa.run(item.question)
    return {"answer": response}
