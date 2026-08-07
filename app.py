"""
Streamlit demo UI for the Smart Campus Multi-Agent AI System (AgentX 2026).

Run with: streamlit run app.py
"""

import json
import streamlit as st

from backend import memory, tools
from backend.orchestrator import run_orchestrator
from backend.auth import verify_password

st.set_page_config(page_title="Smart Campus AI Assistant", page_icon="🎓", layout="wide")
memory.init_db()

STUDENTS = json.load(open("data/students.json"))
EMAIL_TO_ID = {s["email"].lower(): sid for sid, s in STUDENTS.items()}


def _login_screen():
    st.title("🎓 Smart Campus Assistant")
    st.caption("AgentX 2026 — Multi-Agent AI System")
    st.subheader("Login")

    with st.form("login_form"):
        email = st.text_input("College email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        student_id = EMAIL_TO_ID.get(email.strip().lower())
        student = STUDENTS.get(student_id) if student_id else None
        if student and verify_password(password, student["password_hash"]):
            st.session_state.authenticated_student = student_id
            st.rerun()
        else:
            st.error("Invalid email or password.")


# ---------------- Auth gate ----------------
if "authenticated_student" not in st.session_state:
    _login_screen()
    st.stop()

student_id = st.session_state.authenticated_student
profile = STUDENTS[student_id]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎓 Smart Campus Assistant")
    st.caption("AgentX 2026 — Multi-Agent AI System")

    st.markdown(f"**{profile['name']}**  \n{profile['branch']} · Year {profile['year']} · CGPA {profile['cgpa']}")
    st.caption(profile["email"])

    if st.button("🚪 Log out"):
        del st.session_state.authenticated_student
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Try asking:**")
    st.markdown(
        "- *Am I eligible for the Google internship? If yes, register me for tomorrow's placement "
        "workshop and add it to my calendar.*\n"
        "- *Summarize the exam regulations and check my attendance.*\n"
        "- *Show today's classes and recommend AI clubs.*\n"
        "- *What's the hostel leave process? Draft an email to the warden requesting leave.*"
    )
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("⚙️ Debug: last agent trace"):
        trace = st.session_state.get("last_trace")
        if trace:
            st.json(trace)
        else:
            st.caption("Ask something to see the orchestration trace here.")

# ---------------- Chat state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "student_id_prev" not in st.session_state or st.session_state.student_id_prev != student_id:
    st.session_state.messages = []
    st.session_state.student_id_prev = student_id

st.header("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _process_user_message(text: str):
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant"):
        with st.spinner("Orchestrator is planning and coordinating agents..."):
            try:
                chat_history = [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
                ]
                result_state = run_orchestrator(text, student_id, chat_history)
                final_response = result_state["final_response"]
                st.session_state.last_trace = {
                    "plan": result_state["plan"],
                    "step_results": result_state["step_results"],
                }
            except Exception as e:
                final_response = (
                    f"Sorry, something went wrong while processing your request ({e}). "
                    f"Please try rephrasing, or check that GROQ_API_KEY is set correctly."
                )
        st.markdown(final_response)

        with st.expander("🔍 How I got this answer (agent trace)"):
            trace = st.session_state.get("last_trace")
            if trace:
                st.markdown(f"**Plan:** {len(trace['plan'])} step(s)")
                for i, (p, r) in enumerate(zip(trace["plan"], trace["step_results"])):
                    st.markdown(f"**Step {i+1} — {r['agent']}**: {p['task']}")
                    for tc in r.get("tool_calls", []):
                        st.code(f"tool: {tc['tool']}\nargs: {tc['args']}\nresult: {tc['result']}", language="text")

    st.session_state.messages.append({"role": "assistant", "content": final_response})


# ---------------- Text input ----------------
user_input = st.chat_input("Ask me anything about academics, placements, events, hostel, library...")
if user_input:
    _process_user_message(user_input)