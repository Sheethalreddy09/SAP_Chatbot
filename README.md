# SAP Knowledge Assistant (SAP_BRIM)

Enterprise AI Assistant with Developer-Managed RAG, strict SAP domain boundaries, hybrid retrieval (pgvector dense embeddings + lexical search + cross-encoder reranking), grounding guardrails, and authoritative SAP web fallback.

---

## 🚀 Quick Start Guide

### 1. Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 18.x or higher & **npm**
- **Docker** *(Optional)*: Required only if running PostgreSQL with pgvector container. A pre-indexed SQLite database (`backend/sap_assistant.db`) is included as an automatic fallback with zero configuration needed.

---

### 2. Backend Setup & Execution

#### Step 1: Navigate to the project root
Open your terminal in the `SAP_BRIM` directory:
```bash
cd SAP_BRIM
```

#### Step 2: Configure Environment Variables
Create or verify your `backend/.env` file (copied from `backend/.env.example`):
```bash
cp backend/.env.example backend/.env
```

Key environment configurations in `backend/.env`:
```ini
# Database (Auto-falls back to SQLite if PostgreSQL is unreachable)
DATABASE_URL=

# JWT Security
JWT_SECRET=sap-super-secret-jwt-key-2026-production-ready
ADMIN_API_KEY=sap-admin-dev-secret-key-999

# LLM Provider Options: 'openai', 'groq', or 'local'
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_or_groq_api_key_here
LLM_MODEL=gpt-4o-mini
GROQ_API_KEY=your_groq_key_if_using_groq

# Web Fallback (Tavily search for official SAP docs)
WEB_FALLBACK_ENABLED=True
WEB_SEARCH_API_KEY=your_tavily_key_here
```

#### Step 3: Install Python Dependencies
```bash
pip install -r backend/requirements.txt
```

#### Step 4: (Optional) Start PostgreSQL with pgvector via Docker
If you wish to use PostgreSQL with vector extensions instead of the local SQLite fallback:
```bash
docker-compose up -d
```

#### Step 5: (Optional) Ingest Knowledge Base Documents
The project already includes pre-indexed embeddings in `backend/sap_assistant.db`. If you add new PDFs to `knowledge_base/documents/` or rebuild indices:
```bash
python -m backend.app.ingestion.ingest
```

#### Step 6: Run the Backend Server
From the `SAP_BRIM` directory, run:
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

### 3. Frontend Setup & Execution

Open a separate terminal window:

#### Step 1: Navigate to the Frontend Directory
```bash
cd SAP_BRIM/frontend
```

#### Step 2: Install Frontend Dependencies
```bash
npm install
```

#### Step 3: Run the Development Server
```bash
npm run dev
```

The frontend will start with Vite HMR:
- **Application URL**: [http://localhost:5173](http://localhost:5173)
- API requests (`/api/*`) are automatically proxied to the backend at `http://localhost:8000`.

#### Step 4: Build for Production
To build static production assets:
```bash
npm run build
```

---

### 4. Running Backend Tests
From the `SAP_BRIM` directory:
```bash
pytest backend/tests -v
```

---

## 🛠️ Summary of Running Commands

| Component | Working Directory | Command | URL |
| :--- | :--- | :--- | :--- |
| **Backend API** | `SAP_BRIM` | `python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload` | [http://localhost:8000](http://localhost:8000) |
| **Backend Docs** | `SAP_BRIM` | *(Included with backend)* | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **Frontend Dev** | `SAP_BRIM/frontend` | `npm run dev` | [http://localhost:5173](http://localhost:5173) |
| **Ingestion Pipeline** | `SAP_BRIM` | `python -m backend.app.ingestion.ingest` | - |
| **Postgres (Docker)**| `SAP_BRIM` | `docker-compose up -d` | `localhost:5434` |
| **Run Unit Tests** | `SAP_BRIM` | `pytest backend/tests` | - |

---

## 🏛️ System Architecture

- **Backend Framework**: FastAPI (Async Python)
- **Database**: PostgreSQL with `pgvector` extension + SQLite zero-config fallback
- **RAG Pipeline**:
  - Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Hybrid Search: Dense Vector Retrieval + BM25 Lexical Retrieval
  - Re-ranking: Cross-Encoder reranker
  - Grounding Guardrails: Strict relevance & hallucination score checks
  - SAP Restriction: Pre-flight domain intent classifier
- **Web Fallback**: Authoritative SAP documentation retrieval via Tavily (`help.sap.com`, `community.sap.com`)
- **Frontend**: React 19, TypeScript, Vite, TailwindCSS, Lucide Icons, React Markdown
