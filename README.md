# 📚 Book RAG Chatbot

An AI-powered chatbot that answers questions from books using **Retrieval-Augmented Generation (RAG)**.
The system retrieves relevant content from a PDF and generates accurate answers using an LLM.

---

## 🚀 Features

* 📄 Chat with PDF books
* 🔎 Semantic search using vector embeddings
* 🧠 Context-aware answers
* ⚡ Fast retrieval using FAISS vector database
* 💬 Interactive chat interface
* 📚 Source citation from book pages

---

## 🛠 Tech Stack

* **Python**
* **LangChain**
* **FAISS Vector Database**
* **Sentence Transformers**
* **Google Gemini LLM**
* **Streamlit UI**

---

## 🧠 How It Works

1. Load the PDF document
2. Split the text into smaller chunks
3. Convert chunks into embeddings
4. Store embeddings in a vector database
5. Retrieve relevant chunks based on user question
6. Send context + question to the LLM
7. Generate the final answer

This technique is called **Retrieval-Augmented Generation (RAG)**.

---

## 📂 Project Structure

```
book-rag-chatbot
│
├── retriever.py          # Chat interface and RAG pipeline
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
│
├── document_file_db/     # FAISS vector database
└── files/
    └── book.pdf          # Source document
```

---

## ⚙️ Installation

Clone the repository

```
git clone https://github.com/UmeshDBarhate/book-rag-chatbot.git
```

Go to the project folder

```
cd book-rag-chatbot
```

Create virtual environment

```
python -m venv env
```

Activate environment

Windows:

```
env\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file and add your API key:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

Start the Streamlit app

```
streamlit run retriever.py
```

Open browser:

```
http://localhost:8501
```

---

## 📸 Demo

Ask questions about your book and get AI-generated answers based on the document.

Example:

```
User: What is Python?
AI: Python is a high-level programming language designed for readability and simplicity.
```

---

## 🔮 Future Improvements

* Upload multiple PDFs
* Conversation memory
* Hybrid search (BM25 + vector)
* Deploy on cloud
* Multi-user support

---

## 👨‍💻 Author

**Umesh Barhate**

Software Developer | Python | AI | Backend Development
