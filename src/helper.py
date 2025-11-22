from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader  # This forces it to use the better loader
    )
    
    documents = loader.load()
    return documents


from typing import List
from langchain.schema import Document

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Returns docs with only 'source', 'page', and 'page_content'.
    """
    minimal_docs: List[Document] = []
    
    for doc in docs:
        # Extract the specific metadata you need
        src = doc.metadata.get("source")
        page_num = doc.metadata.get("page", 0) # Default to 0 if no page found
        
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": src,
                    "page": page_num + 1  # PyMuPDF starts at 0, humans start at 1
                }
            )
        )
    return minimal_docs


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # Big enough for tables, fits in MiniLM limit
        chunk_overlap=100,   # Ensures sentences/logic aren't cut off
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Tries to split by paragraph first
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


from langchain.embeddings import HuggingFaceEmbeddings

def download_embeddings():
    # Using MPNET because it allows 2000+ characters per chunk
    # This is perfect for your 1100 chunk size!
    model_name = "sentence-transformers/all-mpnet-base-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings