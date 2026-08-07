"""
Specialized agents. Each agent:
  1. Receives a natural-language sub-task (from the orchestrator's plan) plus
     the student_id and any prior step results for context.
  2. Asks Gemini (constrained to JSON) which of its own tools to call and
     with what arguments -- this is the "Tool / Function Calling" +
     "Multi-step Reasoning" requirement, kept explicit and inspectable
     rather than hidden inside SDK auto function-calling.
  3. Executes that tool against the mock data layer.
  4. Asks Gemini to turn the raw tool result into a short natural-language
     answer for that sub-task.

Every agent returns a dict: {"agent": ..., "task": ..., "tool_calls": [...], "summary": "..."}
"""

import inspect

from backend import tools
from backend.rag import answer_from_knowledge_base
from backend.llm_client import generate, generate_json

STUDENT_CONTEXT_NOTE = (
    "The student's own profile is looked up automatically -- you do not need "
    "to ask the student for their student_id, branch, or year."
)


def _decide_and_call(agent_name: str, task: str, student_id: str, tool_menu: str, tool_map: dict) -> dict:
    """Generic helper: ask Gemini to choose one tool call, then execute it.

    Hardened so that a malformed/unparseable LLM response never raises --
    it degrades to a "no matching tool" result instead, which downstream
    summarization can turn into an honest message rather than the
    orchestrator's generic error fallback.
    """
    prompt = f"""
You are the {agent_name}. Sub-task: "{task}"
Student ID: {student_id}
{STUDENT_CONTEXT_NOTE}

Available tools (choose exactly one that best fits the sub-task):
{tool_menu}

If none of the available tools can answer the sub-task (for example, no tool
exposes the specific data being asked about), respond with:
{{"tool": null, "args": {{}}}}

Respond ONLY with JSON: {{"tool": "<tool_name_or_null>", "args": {{...}}}}
Use the exact argument names shown. Always include "student_id" if the tool needs it.
"""
    try:
        decision = generate_json(prompt, system_instruction=f"You are the {agent_name} in a campus multi-agent system.")
    except Exception:
        decision = {}

    tool_name = decision.get("tool") if isinstance(decision, dict) else None
    args = (decision.get("args") if isinstance(decision, dict) else None) or {}
    args.setdefault("student_id", student_id)

    fn = tool_map.get(tool_name)
    if not fn:
        return {
            "tool": tool_name,
            "args": args,
            "result": {"not_available": "No tool currently exposes this specific information."},
        }

    # Only pass args the function actually accepts
    valid_params = set(inspect.signature(fn).parameters.keys())
    filtered_args = {k: v for k, v in args.items() if k in valid_params}
    try:
        result = fn(**filtered_args)
    except Exception as e:  # graceful fallback per non-functional requirements
        result = {"error": f"Tool execution failed: {e}"}
    return {"tool": tool_name, "args": filtered_args, "result": result}


def _summarize(agent_name: str, task: str, tool_call: dict) -> str:
    prompt = (
        f"Sub-task: {task}\n"
        f"Tool used: {tool_call['tool']}\n"
        f"Tool result: {tool_call['result']}\n\n"
        f"Write a short (2-4 sentence) natural-language answer to the sub-task based on this result. "
        f"If the result contains an error, explain it gracefully and suggest what the student could do instead. "
        f"If the result says information is not available or not tracked, say so plainly and matter-of-factly "
        f"(e.g. 'There's no interview schedule on file for this week') -- do NOT describe it as a technical "
        f"issue, glitch, or error, and do NOT tell the student to check a portal or calendar that hasn't been "
        f"mentioned to you.\n\n"
        f"CRITICAL -- do not confirm or deny anything about a specific named entity (a place, route, company, "
        f"club, course, person, etc.) unless that exact name literally appears in the tool result above. "
        f"For example, if the sub-task asks about a specific location and the tool result lists routes/items "
        f"that do NOT include that location by name, you must say it is not listed in the available data -- "
        f"never answer 'yes' by generalizing from an unrelated total or count (e.g. do not turn 'the campus "
        f"has 27 buses across 4 other routes' into 'yes, that location has 27 buses'). When the named entity "
        f"is absent from the result, name which entities/items ARE covered instead of affirming the one that "
        f"isn't. Only state facts, names, and numbers that literally appear in the tool result above -- never "
        f"invent or guess any detail that isn't there, even to sound more complete or helpful."
    )
    return generate(
        prompt,
        system_instruction=(
            f"You are the {agent_name}. Never fabricate information not present in the tool result. "
            f"Never confirm details about a named entity that does not literally appear in the tool result."
        ),
    )


# ---------------- Academic Agent ----------------

ACADEMIC_TOOLS = """
- get_timetable(student_id, day): today's/weekly class schedule
- get_attendance(student_id, course_code): attendance percentage
- get_courses(branch): list of courses for the student's branch
- recommend_electives(branch, interest_keyword): elective suggestions matching an interest
""".strip()

ACADEMIC_TOOL_MAP = {
    "get_timetable": tools.get_timetable,
    "get_attendance": tools.get_attendance,
    "get_courses": tools.get_courses,
    "recommend_electives": tools.recommend_electives,
}


def run_academic_agent(task: str, student_id: str) -> dict:
    profile = tools.get_student_profile(student_id)
    tool_menu = ACADEMIC_TOOLS
    call = _decide_and_call("Academic Agent", task, student_id, tool_menu, ACADEMIC_TOOL_MAP)
    if call["tool"] in ("get_courses", "recommend_electives"):
        call["args"].setdefault("branch", profile.get("branch"))
        call["result"] = ACADEMIC_TOOL_MAP[call["tool"]](**{k: v for k, v in call["args"].items() if k != "student_id"})
    summary = _summarize("Academic Agent", task, call)
    return {"agent": "Academic Agent", "task": task, "tool_calls": [call], "summary": summary}


# ---------------- Placement Agent ----------------

PLACEMENT_TOOLS = """
- check_placement_eligibility(student_id, company_name): checks if the student is eligible for a company's drive
- list_placement_drives(type): lists active placement/internship drives, including role, deadline, and eligibility. Pass type="Internship" if the student specifically asked about internships, type="Full-time" for full-time roles only, or omit type to list everything.
- get_placement_statistics(): historical placement stats -- total offers, top recruiters and packages, department-wise placement counts, for the last completed placement season

Note: there is no tool that tracks a separate "interview schedule". If asked
about upcoming interviews, use list_placement_drives (deadlines are the
closest available data) and choose {"tool": null} if that still does not
answer what was asked.
""".strip()

PLACEMENT_TOOL_MAP = {
    "check_placement_eligibility": tools.check_placement_eligibility,
    "list_placement_drives": tools.list_placement_drives,
    "get_placement_statistics": tools.get_placement_statistics,
}


def run_placement_agent(task: str, student_id: str) -> dict:
    call = _decide_and_call("Placement Agent", task, student_id, PLACEMENT_TOOLS, PLACEMENT_TOOL_MAP)

    # If the model couldn't find a fitting tool (e.g. an "interviews this week"
    # style question with no matching data), fall back to showing active
    # drives instead of surfacing a dead end, and be explicit that no
    # interview-schedule data is tracked.
    if call["tool"] is None or "not_available" in call.get("result", {}):
        drives = tools.list_placement_drives()
        call = {
            "tool": "list_placement_drives",
            "args": {},
            "result": {
                "note": "No interview-schedule data is tracked by this system.",
                "active_drives": drives,
            },
        }

    summary = _summarize("Placement Agent", task, call)
    return {"agent": "Placement Agent", "task": task, "tool_calls": [call], "summary": summary}


# ---------------- Events Agent ----------------

EVENTS_TOOLS = """
- list_events(tag): list upcoming workshops/hackathons, optionally filtered by tag (e.g. "AI", "placement")
- list_clubs(interest_keyword): list clubs matching an interest keyword
- register_for_event(student_id, event_id): register the student for an event
""".strip()

EVENTS_TOOL_MAP = {
    "list_events": tools.list_events,
    "list_clubs": tools.list_clubs,
    "register_for_event": tools.register_for_event,
}


def run_events_agent(task: str, student_id: str) -> dict:
    call = _decide_and_call("Events Agent", task, student_id, EVENTS_TOOLS, EVENTS_TOOL_MAP)
    summary = _summarize("Events Agent", task, call)
    return {"agent": "Events Agent", "task": task, "tool_calls": [call], "summary": summary}


# ---------------- Student Services Agent ----------------

SERVICES_TOOLS = """
- get_service_info(category): category is one of "hostel", "library", "scholarships", "transport"
- file_grievance(student_id, category, description): file a grievance ticket
""".strip()

SERVICES_TOOL_MAP = {
    "get_service_info": tools.get_service_info,
    "file_grievance": tools.file_grievance,
}


def run_student_services_agent(task: str, student_id: str) -> dict:
    call = _decide_and_call("Student Services Agent", task, student_id, SERVICES_TOOLS, SERVICES_TOOL_MAP)
    summary = _summarize("Student Services Agent", task, call)
    return {"agent": "Student Services Agent", "task": task, "tool_calls": [call], "summary": summary}


# ---------------- Communication Agent ----------------

COMMUNICATION_TOOLS = """
- draft_email(to, subject, key_points): drafts an email
- send_notification(student_id, message): sends an in-app notification
""".strip()

COMMUNICATION_TOOL_MAP = {
    "draft_email": tools.draft_email,
    "send_notification": tools.send_notification,
}


def run_communication_agent(task: str, student_id: str, extra_context: str = "") -> dict:
    full_task = task if not extra_context else f"{task}\n\nContext from earlier steps:\n{extra_context}"
    call = _decide_and_call("Communication Agent", full_task, student_id, COMMUNICATION_TOOLS, COMMUNICATION_TOOL_MAP)
    summary = _summarize("Communication Agent", task, call)
    return {"agent": "Communication Agent", "task": task, "tool_calls": [call], "summary": summary}


# ---------------- Notification / Calendar Agent ----------------

CALENDAR_TOOLS = """
- add_to_calendar(student_id, title, date, time): add an event to the student's calendar
- set_reminder(student_id, title, event_datetime, remind_before_minutes): set a reminder (event_datetime as ISO format YYYY-MM-DDTHH:MM)
""".strip()

CALENDAR_TOOL_MAP = {
    "add_to_calendar": tools.add_to_calendar,
    "set_reminder": tools.set_reminder,
}


def run_notification_agent(task: str, student_id: str) -> dict:
    call = _decide_and_call("Notification/Calendar Agent", task, student_id, CALENDAR_TOOLS, CALENDAR_TOOL_MAP)
    summary = _summarize("Notification/Calendar Agent", task, call)
    return {"agent": "Notification/Calendar Agent", "task": task, "tool_calls": [call], "summary": summary}


# ---------------- Knowledge Agent (RAG) ----------------

def run_knowledge_agent(task: str, student_id: str) -> dict:
    result = answer_from_knowledge_base(task)
    return {
        "agent": "Knowledge Agent",
        "task": task,
        "tool_calls": [{"tool": "rag_retrieve", "args": {"query": task}, "result": result}],
        "summary": result["answer"],
    }


# ---------------- Campus Information Agent ----------------

CAMPUS_TOOLS = """
- get_departments(): list all departments.
- get_department(name): Use ONLY for department information.
- get_faculty(department): Use ONLY for HOD or faculty questions.
- get_building(name): Use for locations, directions, buildings, offices, library, placement cell, examination branch, hostel, cafeteria and campus navigation.
- search_faq(question): Use ONLY if the question is about procedures, hostel process, scholarship process, leave process or other FAQs.
- get_institute_info(): Use for questions about the Principal, Chairman, Director, institute establishment year, affiliation, or top-level institute contact info.
- get_induction_program(): Use for questions about orientation/induction programme for newly admitted students -- schedule, activities, timing.
- get_sports_info(): Use for questions about sports facilities, playgrounds, courts, indoor games, or the gym.
""".strip()


CAMPUS_TOOL_MAP = {
    "get_departments": tools.get_departments,
    "get_department": tools.get_department,
    "get_faculty": tools.get_faculty,
    "get_building": tools.get_building,
    "search_faq": tools.search_faq,
    "get_institute_info": tools.get_institute_info,
    "get_induction_program": tools.get_induction_program,
    "get_sports_info": tools.get_sports_info,
}


def run_campus_agent(task: str, student_id: str) -> dict:
    call = _decide_and_call(
        "Campus Information Agent",
        task,
        student_id,
        CAMPUS_TOOLS,
        CAMPUS_TOOL_MAP,
    )

    summary = _summarize(
        "Campus Information Agent",
        task,
        call,
    )

    return {
        "agent": "Campus Information Agent",
        "task": task,
        "tool_calls": [call],
        "summary": summary,
    }