# 🚀 Finance AI Assistant

![Finance Chatbot](https://img.shields.io/badge/Status-Beta-brightgreen.svg)
![React](https://img.shields.io/badge/Frontend-React%20Vite-%2361DAFB?logo=react&logoColor=white)
![Express](https://img.shields.io/badge/Backend-Node.js%2FExpress-339933?logo=nodedotjs&logoColor=white)
![Hugging Face](https://img.shields.io/badge/AI-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)

Welcome to the **Finance AI Assistant**! A full-stack, RAG-capable chatbot designed exclusively for navigating the world of finance, investments, and market insights. 

Powered by the robust **qwen-finance-7b** model on Hugging Face, it answers your nuanced finance queries and can even take uploaded documents for targeted, context-aware answers.

---

## ✨ Features

- **Dark Mode UI**: Sleek, high-contrast user interface matching premium fintech tools.
- **Micro-Interactions**: Hover states, animated thinking indicators, and smooth chat flows.
- **Financial Intelligence**: Interacts directly with fine-tuned financial LLMs.
- **RAG-Ready**: Drag and drop support via `multer` for `.pdf`, `.csv`, or `.txt` text parsing.
- **Prompt Suggestions**: Auto-fill complex questions with built-in suggestion cards.

---

## 🛠 Tech Stack

### Client-side
- **React 18** (Vite compiler for ultra-fast HMR)
- **Vanilla CSS** with CSS Variables for responsive and scalable design
- **Lucide-React** for minimalist SVG icons

### Server-side
- **Node.js** + **Express** API framework
- **Multer** for seamless multipart/form-data file handling
- **@huggingface/inference** interface for zero-friction LLM connectivity

---

## 🏎 How to Run Locally

To get the Finance AI fully running on your local machine, you'll need two terminal windows.

### 1. The Backend (Node/Express Server)
Navigate to the `server` directory and hook up the AI.

```bash
cd server
npm install
```

**CRITICAL:** Create a `.env` file in the `server` folder, and add your Hugging Face inference key:
```env
HF_TOKEN=hf_your_secret_token_here
PORT=5000
```

Start the server:
```bash
node server.js
```

### 2. The Frontend (Vite/React Client)
Navigate to the `client` directory and start the Vite development server.

```bash
cd client
npm install
npm run dev
```

🚀 Now open [http://localhost:5173/](http://localhost:5173/) to chat with the AI!

---

## 🔮 Future Enhancements
- [] **Vector Database Setup:** Connect to ChromaDB/Pinecone to fully embed uploaded historical data.
- [] **User Auth:** Secure the dashboard with JWT authentication.
- [] **Live Market Data:** Integration with Alpha Vantage or YFinance APIs for live ticker checking.

> *"An investment in knowledge pays the best interest." — Benjamin Franklin*
