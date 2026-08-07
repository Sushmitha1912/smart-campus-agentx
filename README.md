# 🎓 Smart Campus Multi-Agent AI System — AgentX 2026

A working proof-of-concept for the AgentX National Level Hackathon "Smart Campus Multi-Agent AI System" problem statement. An **Orchestrator Agent** (built with LangGraph) plans a user's request and coordinates specialized agents — Academic, Placement, Events, Student Services, Communication, Notification/Calendar, and a RAG-powered Knowledge Agent — to answer questions and execute multi-step campus workflows autonomously.

## ✅ What this demonstrates

| Requirement | How |
|---|---|
| Multi-agent architecture | 7 specialized agents + orchestrator, LangGraph `StateGraph` |
| Autonomous planning | Planner node decomposes any request into an ordered agent plan |
| Agent-to-agent collaboration | Router dispatches sequentially; later agents (e.g. Communication) use earlier agents' results as context |
| RAG | Knowledge Agent: Gemini embeddings + cosine similarity search over policy documents |
| Tool / Function calling | Every agent explicitly selects and calls mock tools (calendar, registration, email, DB lookups) |
| Memory | SQLite short-term conversation history + long-term per-student facts |
| Context-aware conversations | Planner is given recent chat history |
| Error handling / fallback | Try/except around every tool + agent call, with graceful natural-language fallback messages |
| End-to-end workflow execution | E.g. "check eligibility → register for event → add to calendar → set reminder" in one request |

## 🗂️ Project structure

```
smart-campus-agentx/
├── app.py                     # Streamlit chat UI
├── backend/
│   ├── config.py               # env/config
│   ├── llm_client.py           # Gemini wrapper (generate / generate_json / embed)
│   ├── tools.py                 # mock campus APIs (calendar, email, registration, DB)
│   ├── rag.py                   # Knowledge Agent (embeddings + retrieval)
│   ├── memory.py                # SQLite short-term + long-term memory
│   ├── agents.py                # 6 specialized agents
│   └── orchestrator.py          # LangGraph orchestrator (planner + router + synthesizer)
├── data/                        # mock JSON "database" + knowledge_base/*.md policy docs
├── architecture.md              # architecture diagram (Mermaid)
├── requirements.txt
└── .env.example
```

## 🚀 Setup (run locally — needs internet + a Gemini API key)

1. **Get a free Gemini API key**: https://aistudio.google.com/apikey

2. **Install dependencies** (Python 3.10+):
   ```bash
   cd smart-campus-agentx
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure your API key**:
   ```bash
   cp .env.example .env
   # edit .env and paste your GEMINI_API_KEY
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`. The first query will take a few extra seconds while the Knowledge Agent embeds and caches the policy documents (`kb_embeddings_cache.json`) — subsequent runs reuse the cache.

5. **Pick a demo student** from the sidebar and try the example prompts shown there, e.g.:
   > "I'm a third-year CSE student. Am I eligible for the Google internship? If yes, register me for tomorrow's placement workshop, add it to my calendar, and remind me one hour before."

   Expand **"How I got this answer"** under any response to see the exact plan, which agent handled each step, and which tool was called — great for live demo/judging.

## 🧠 How a request flows

1. User message hits the **Orchestrator** (`orchestrator.py`).
2. **Planner node** asks Gemini (JSON mode) to break the request into ordered `{agent, task}` steps, using recent chat history + long-term memory for context.
3. The **router** (`route_after_step`) sends control to each agent node in sequence.
4. Each **specialized agent** picks a tool from its own menu, calls it against the mock data in `data/`, and summarizes the result.
5. The **synthesizer** combines every step's summary into one natural final answer and saves the turn to memory.

See `architecture.md` for the full diagram.

## 🔧 Extending this

- Swap `backend/tools.py` mock functions for real API calls (calendar, email, ERP) — agent logic doesn't need to change.
- Swap the in-memory vector store in `rag.py` for FAISS/Chroma/Pinecone for larger document sets.
- Add new agents by writing a `run_x_agent(task, student_id)` function and registering it in `AGENT_REGISTRY` in `orchestrator.py`.
- Stretch goals not yet wired up: voice interaction, multilingual support, human-in-the-loop approval before executing actions like registration/email-sending, vision/OCR for scanned documents.

## ⚠️ Known limitations (proof-of-concept scope)

- All campus data is mocked (JSON files), per the hackathon brief — no real institutional integration.
- Event registration counts and calendar/reminders are simulated in-memory/mock responses, not persisted to a real calendar.
- Single-turn planning: the planner creates one plan per user message rather than re-planning mid-execution if a step fails (errors are caught and reported gracefully instead).
