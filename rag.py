from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import pdfplumber

def extract_tables_as_text(pdf_path: str) -> list:
    table_docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                rows = []
                for row in table:
                    cleaned = [cell if cell else "" for cell in row]
                    rows.append(" | ".join(cleaned))
                table_text = "\n".join(rows)
                if table_text.strip():
                    table_docs.append(Document(
                        page_content=f"[TABLE - Page {i}]\n{table_text}",
                        metadata={"page": i, "type": "table"}
                    ))
    return table_docs

def load_and_index_pdf(pdf_path: str, persist_dir: str = "chroma_db"):
    # Load regular text
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Extract tables separately and add to chunks
    table_chunks = extract_tables_as_text(pdf_path)
    all_chunks = chunks + table_chunks

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore

def get_qa_chain(persist_dir: str = "chroma_db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = OllamaLLM(model="mistral")

    prompt = ChatPromptTemplate.from_template("""
You are a financial analyst assistant. Answer using only the context below.
Tables are marked with [TABLE - Page N]. Use table data when relevant.
If the answer is not in the context, say "I couldn't find that in the document."

Context: {context}
Question: {question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return retriever, chain