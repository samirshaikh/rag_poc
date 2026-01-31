# Local RAG System (Ollama + TypeScript)

A fully local Retrieval-Augmented Generation (RAG) system using **LlamaIndex.TS** and **Ollama**. This setup bypasses standard library limitations to provide precise, page-level citations.

---

## 🛠 1. Prerequisites
- **Node.js**: v22+ (v25 recommended)
    pacman -Sy nodejs npm 

- **Ollama**: Running locally with `ollama pull qwen2.5:3b`
    
- **PDFs**: Placed in the `./pdf` directory

---

## 📂 2. Project Structure
- `pdf/`: Source documents.
- `storage/`: Local vector cache (ignored by Git).
- `build_index.ts`: Manually parses PDFs page-by-page to ensure metadata accuracy.
- `chat.ts`: Query engine with custom numeric citation sorting.

---

## 🚀 3. Installation & Usage

### Setup
```bash
npm install


