from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split,download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

folder_path = r"C:\Users\ASUS\Desktop\chat-bot\data" 
extracted_data = load_pdf_files(folder_path)
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "maternal-chatbot" 


if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=768,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)


from langchain_pinecone import PineconeVectorStore
import time

print("Starting manual upload...")

# 1. Initialize the Vector Store (But don't add documents yet)
docsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

# 2. Define a small batch size (Safe for your internet)
batch_limit = 5 

# 3. Loop through your chunks and add them piece by piece
total_chunks = len(text_chunks)

for i in range(0, total_chunks, batch_limit):
    # Slice the list to get just 5 documents
    batch = text_chunks[i : i + batch_limit]
    
    # Upload them
    docsearch.add_documents(batch)
    
    # Print progress
    print(f"Uploaded: {min(i + batch_limit, total_chunks)} / {total_chunks}")
    
    # Small sleep to prevent hitting API rate limits
    time.sleep(1)

print("Upload Complete!")




