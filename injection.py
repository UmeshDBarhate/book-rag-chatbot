from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import time
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1️⃣ Load PDF
loader = PyPDFLoader("files/python_book.pdf")
documents = loader.load()

# 2️⃣ Calculate total text size
total_length = sum(len(doc.page_content) for doc in documents)

# 3️⃣ Dynamic chunk size
if total_length < 50000:
    chunk_size = 500
elif total_length < 200000:
    chunk_size = 700
else:
    chunk_size = 900

# 4️⃣ Dynamic chunk overlap
chunk_overlap = int(chunk_size * 0.15)

# 5️⃣ Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ".", " "]
)

# 6️⃣ Split documents
chunks = text_splitter.split_documents(documents)

# 7️⃣ Add metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["source"] = "book.pdf"

print(f"Total chunks created: {len(chunks)}")

# 8️⃣ Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 9️⃣ Batch Embedding + Retry + Rate limit handling
batch_size = 50
vector_store = None

for i in range(0, len(chunks), batch_size):

    batch = chunks[i:i+batch_size]
    retries = 5
    delay = 10

    for attempt in range(retries):

        try:
            print(f"Processing batch {i//batch_size + 1}")

            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)

            break

        except Exception as e:

            print("Error:", e)
            print(f"Retrying in {delay} seconds...")

            time.sleep(delay)
            delay *= 2

# 🔟 Save vector database
vector_store.save_local("document_file_db")

print("✅ All embeddings created successfully")
