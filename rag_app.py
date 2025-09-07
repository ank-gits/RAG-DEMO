import re
from os import path
from glob import glob  
import langchain, ollama
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

class RAG:
    def __init__(self, vector_collection_name):
        self.pdf_dir = "uploaded_pdfs"
        self.embedding_model = "all-minilm:latest"
        self.vector_collection_name = vector_collection_name
        self.llm_model = "llama3.1:8b"
        self.documents = []
        self.doc_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.persist_directory = "./my_chroma_db"
        self.embeddings_ollama = OllamaEmbeddings(model=self.embedding_model)
        self.vectorstore = Chroma(persist_directory= self.persist_directory, collection_name=self.vector_collection_name, embedding_function=self.embeddings_ollama)
        self.init_retriever_and_pipeline()

    def read_pdfs(self,pdf_list:list=None):

        if len(pdf_list)==0 or pdf_list is None:
            pdf_list = glob(path.join(self.pdf_dir, "*.pdf"))
        for pdf in pdf_list:
            loader = PyPDFLoader(pdf)
            pages = loader.load()
            pages = [page for page in pages if int(page.metadata['page']) >= 2]
            self.documents.extend(pages)

    def add_to_vector_db(self):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=100
        )
        self.doc_chunks = text_splitter.split_documents(self.documents)
        
        uuids = [str(i) for i in range(len(self.doc_chunks))]
        self.vectorstore.add_documents(documents=self.doc_chunks, ids=uuids)
        rag.init_retriever_and_pipeline()
    
    def init_retriever_and_pipeline(self):
        self.retriever = self.vectorstore.as_retriever(k=4)
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. Use the following documents to answer the question. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise:
            Question: {question}
            Documents: {documents}
            Answer:""",
            input_variables=["question", "documents"],
        )
        llm = ChatOllama(model=self.llm_model, temperature=0)
        self.rag_chain = prompt | llm | StrOutputParser()

    def answer_query(self, question):
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

# Usage example:
# UPLOAD_DIR = "uploaded_pdfs"
# VECTOR_DBB_COLLECTION_NAME = "Earnings_Call_Transcripts"

# rag = RAG(vector_collection_name=VECTOR_DBB_COLLECTION_NAME)
# # rag.read_pdfs()
# # rag.add_to_vector_db()
# # rag.init_retriever_and_pipeline()
# question = "What is the overall sentiment from Gartner's Earnings Call? Is it positive or negative? Explain with reasons."
# answer = rag.answer_query(question)
# print("Question:", question)
# print("Answer:", answer)
