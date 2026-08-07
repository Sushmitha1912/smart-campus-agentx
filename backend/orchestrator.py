"""
Orchestrator Agent, built as a LangGraph StateGraph.

Flow:
  START -> planner -> [dispatch to one agent node] -> dispatch -> ... -> synthesizer -> END

The planner asks Gemini to break the user's request into an ordered list of
sub-tasks, each assigned to a specialized agent. A shared `router` function
reads how many plan steps remain and routes control to the right agent node
(or to the synthesizer once the plan is exhausted). This is a supervisor /
plan-and-execute pattern: a single planning step up front, followed by
sequential tool-using execution, which keeps runs fast and debuggable while
still demonstrating autonomous multi-step planning and agent-to-agent
hand-off.
"""

from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from backend import agents, memory
from backend.llm_client import generate, generate_json

AGENT_REGISTRY = {
    "academic": ("Academic Agent", agents.run_academic_agent),
    "placement": ("Placement Agent", agents.run_placement_agent),
    "events": ("Events Agent", agents.run_events_agent),
    "student_services": ("Student Services Agent", agents.run_student_services_agent),
    "communication": ("Communication Agent", agents.run_communication_agent),
    "notification": ("Notification/Calendar Agent", agents.run_notification_agent),
    "knowledge": ("Knowledge Agent", agents.run_knowledge_agent),

    # NEW
    "campus": ("Campus Information Agent", agents.run_campus_agent),
}
AGENT_CAPABILITIES = """
- academic: timetable, attendance, course lists, elective recommendations
- placement: internship/placement eligibility checks, listing active drives, placement statistics (past offers, top recruiters, department-wise placements)
- events: workshops/hackathons discovery, club recommendations, event registration
- student_services: hostel, library, transport, scholarships, filing grievances
- communication: drafting emails, sending notifications
- notification: adding events to calendar, setting reminders
- knowledge: answering policy/regulation questions from official documents (attendance policy, exam regulations, placement policy, hostel rules, library rules, scholarships, grievance redressal) via RAG
- campus: departments, faculty, buildings, campus navigation, campus FAQs, important contacts
""".strip()


class CampusState(TypedDict):
    user_query: str
    student_id: str
    chat_history: List[Dict[str, str]]
    plan: List[Dict[str, str]]
    step_index: int
    step_results: List[Dict[str, Any]]
    final_response: str


def planner_node(state: CampusState) -> CampusState:
    history_str = "\n".join(f"{h['role']}: {h['content']}" for h in state["chat_history"][-6:])
    facts = memory.get_facts(state["student_id"])
    facts_str = ", ".join(f"{k}={v}" for k, v in facts.items()) or "none recorded yet"

    prompt = f"""
You are the Orchestrator Agent for a Smart Campus Multi-Agent AI System.
Break the user's request into an ordered list of sub-tasks, each assigned to exactly one specialized agent.
Only include a "communication" or "notification" step if the user explicitly asked for an email/reminder/calendar action.
Combine everything needed for one agent into a single step rather than splitting unnecessarily.

Available agents:
{AGENT_CAPABILITIES}

Recent conversation:
{history_str or "none"}

Known long-term facts about this student: {facts_str}

User request: "{state['user_query']}"

Respond ONLY with JSON in this exact shape:
{{"plan": [{{"agent": "<agent_key>", "task": "<specific sub-task description>"}}]}}
"""
    result = generate_json(prompt, system_instruction="You are a careful task planner. Only use agent keys from the provided list.")
    plan = result.get("plan", [])
    plan = [p for p in plan if p.get("agent") in AGENT_REGISTRY]
    if not plan:
        # graceful fallback: route everything to the knowledge agent as a general Q&A catch-all
        plan = [{"agent": "knowledge", "task": state["user_query"]}]

    state["plan"] = plan
    state["step_index"] = 0
    state["step_results"] = []
    return state


def make_agent_node(agent_key: str):
    display_name, fn = AGENT_REGISTRY[agent_key]

    def node(state: CampusState) -> CampusState:
        step = state["plan"][state["step_index"]]
        try:
            if agent_key == "communication" and state["step_results"]:
                prior_context = "\n".join(
                    f"- {r['agent']}: {r['summary']}" for r in state["step_results"]
                )
                result = fn(step["task"], state["student_id"], extra_context=prior_context)
            else:
                result = fn(step["task"], state["student_id"])
        except Exception as e:
            result = {
                "agent": display_name,
                "task": step["task"],
                "tool_calls": [],
                "summary": f"This step could not be completed due to an error ({e}). Continuing with the rest of the request.",
            }
        state["step_results"].append(result)
        state["step_index"] += 1
        return state

    return node


def route_after_step(state: CampusState) -> str:
    if state["step_index"] >= len(state["plan"]):
        return "synthesizer"
    return state["plan"][state["step_index"]]["agent"]


def synthesizer_node(state: CampusState) -> CampusState:
    results_str = "\n\n".join(
        f"Step {i+1} ({r['agent']} — {r['task']}):\n{r['summary']}"
        for i, r in enumerate(state["step_results"])
    )
    prompt = f"""
The user asked: "{state['user_query']}"

Here is what each specialized agent found/did, in order:

{results_str}

Write one clear, friendly, well-organized final response to the user that weaves these together
(use short paragraphs or a short list where helpful). Do not mention "agents" or internal steps explicitly;
just answer naturally as their campus assistant.

CRITICAL: Only use facts, names, numbers, locations, and details that literally appear in the step
results above. Do NOT invent, assume, embellish, or guess any detail that is not explicitly present
in the results -- this includes building names, floor numbers, directions, phone numbers, emails, or
any other specifics. If a result says information is missing, unavailable, or an error occurred, say
so honestly to the user instead of making up a plausible-sounding answer. It is better to say
"I don't have that specific detail on file" than to fabricate one.

If the user asked about a specific named entity (a place, route, company, club, person, etc.) and that
exact name does not appear in the step results, do NOT say "yes" or otherwise confirm anything about it
by generalizing from a nearby total, count, or list of different items. State plainly that it is not in
the available data, and mention what IS covered instead.
"""
    final = generate(
        prompt,
        system_instruction=(
            "You are the unified Smart Campus Assistant giving the final answer to a student. "
            "You must never fabricate information that is not present in the provided step results, "
            "and must never confirm details about a named entity that is not literally present in those results."
        ),
    )
    state["final_response"] = final

    memory.save_message(state["student_id"], "user", state["user_query"])
    memory.save_message(state["student_id"], "assistant", final)
    return state


def build_graph():
    graph = StateGraph(CampusState)
    graph.add_node("planner", planner_node)
    for agent_key in AGENT_REGISTRY:
        graph.add_node(agent_key, make_agent_node(agent_key))
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")

    routing_map = {k: k for k in AGENT_REGISTRY}
    routing_map["synthesizer"] = "synthesizer"

    graph.add_conditional_edges("planner", route_after_step, routing_map)
    for agent_key in AGENT_REGISTRY:
        graph.add_conditional_edges(agent_key, route_after_step, routing_map)

    graph.add_edge("synthesizer", END)
    return graph.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_orchestrator(user_query: str, student_id: str, chat_history: List[Dict[str, str]] | None = None) -> CampusState:
    app = get_app()
    initial_state: CampusState = {
        "user_query": user_query,
        "student_id": student_id,
        "chat_history": chat_history or memory.get_recent_history(student_id),
        "plan": [],
        "step_index": 0,
        "step_results": [],
        "final_response": "",
    }
    return app.invoke(initial_state)