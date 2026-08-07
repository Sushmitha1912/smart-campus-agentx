"""
Mock campus tools / APIs.

Per the hackathon brief, teams are NOT expected to integrate with real
institutional systems. These functions simulate the External Tools / APIs
layer (Calendar, Email, Registration, Database) from the reference
architecture, operating on the JSON files in /data. Swap these for real
integrations later without changing agent logic, since agents only see
the function signatures below.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from backend.config import DATA_DIR

_cache: dict = {}


def _load(name: str):
    if name not in _cache:
        with open(DATA_DIR / f"{name}.json") as f:
            _cache[name] = json.load(f)
    return _cache[name]


# ---------- Academic tools ----------

def get_student_profile(student_id: str) -> dict:
    return _load("students").get(student_id, {"error": "student not found"})


def get_timetable(student_id: str, day: str | None = None) -> dict:
    tt = _load("timetable").get(student_id, {})
    if day:
        return {day: tt.get(day, [])}
    return tt


def get_attendance(student_id: str, course_code: str | None = None) -> dict:
    att = _load("attendance").get(student_id, {})
    if course_code:
        return {course_code: att.get(course_code, "no data")}
    return att


def get_courses(branch: str) -> list:
    return _load("courses").get(branch, [])


def recommend_electives(branch: str, interest_keyword: str) -> list:
    courses = _load("courses").get(branch, [])
    keyword = interest_keyword.lower()
    return [c for c in courses if c.get("electives") and keyword in c["name"].lower()]


def get_academic_calendar() -> dict:
    return _load("academic_calender")


def get_programmes() -> dict:
    return _load("programmes")


# ---------- Placement tools ----------

def check_placement_eligibility(student_id: str, company_name: str) -> dict:
    student = get_student_profile(student_id)
    if "error" in student:
        return student
    companies = _load("placements")["companies"]
    match = next((c for c in companies if c["name"].lower() == company_name.lower()), None)
    if not match:
        return {"error": f"No active drive found for {company_name}"}

    reasons = []
    eligible = True
    if student["cgpa"] < match["min_cgpa"]:
        eligible = False
        reasons.append(f"CGPA {student['cgpa']} is below required {match['min_cgpa']}")
    if student["branch"] not in match["eligible_branches"]:
        eligible = False
        reasons.append(f"Branch {student['branch']} not eligible")
    if student["year"] < match["min_year"]:
        eligible = False
        reasons.append(f"Year {student['year']} below minimum year {match['min_year']}")

    return {
        "student": student["name"],
        "company": match["name"],
        "role": match["role"],
        "eligible": eligible,
        "reasons": reasons or ["Meets all published eligibility criteria"],
        "deadline": match["deadline"],
        "notes": match["notes"],
    }


def list_placement_drives(type: str | None = None) -> list:
    """List active drives. Pass type="Internship" or type="Full-time" to filter, or omit for all."""
    companies = _load("placements")["companies"]
    if type:
        return [c for c in companies if c.get("type", "").lower() == type.lower()]
    return companies


def get_placement_statistics() -> dict:
    """Historical placement stats: total offers, top recruiters, department-wise breakdown."""
    return _load("placements").get("statistics", {"error": "No placement statistics on file"})


# ---------- Events tools ----------

def list_events(tag: str | None = None) -> list:
    events = _load("events")["events"]
    if tag:
        return [e for e in events if tag.lower() in [t.lower() for t in e["tags"]]]
    return events


def list_clubs(interest_keyword: str | None = None) -> list:
    clubs = _load("clubs")
    if interest_keyword:
        keyword = interest_keyword.lower()
        return [c for c in clubs if any(keyword in f.lower() for f in c["focus"])]
    return clubs


def register_for_event(student_id: str, event_id: str) -> dict:
    events = _load("events")["events"]
    match = next((e for e in events if e["id"] == event_id), None)
    if not match:
        return {"error": f"Event {event_id} not found"}
    if match["registered"] >= match["capacity"]:
        return {"error": f"Event {match['title']} is at full capacity"}
    match["registered"] += 1  # in-memory only for this demo session
    return {"status": "registered", "event": match["title"], "date": match["date"], "time": match["time"]}


# ---------- Calendar / Notification tools ----------

def add_to_calendar(student_id: str, title: str, date: str, time: str) -> dict:
    return {
        "status": "added_to_calendar",
        "student_id": student_id,
        "title": title,
        "date": date,
        "time": time,
    }


def set_reminder(student_id: str, title: str, event_datetime: str, remind_before_minutes: int = 60) -> dict:
    try:
        dt = datetime.fromisoformat(event_datetime)
        remind_at = (dt - timedelta(minutes=remind_before_minutes)).isoformat()
    except ValueError:
        remind_at = f"{remind_before_minutes} minutes before {event_datetime}"
    return {
        "status": "reminder_set",
        "student_id": student_id,
        "title": title,
        "remind_at": remind_at,
    }


# ---------- Communication tools ----------

def draft_email(to: str, subject: str, key_points: str) -> dict:
    body = (
        f"Dear Sir/Madam,\n\n"
        f"{key_points}\n\n"
        f"Regards,\n"
        f"[Student Name]"
    )
    return {"to": to, "subject": subject, "body": body}


def send_notification(student_id: str, message: str) -> dict:
    return {"status": "notification_sent", "student_id": student_id, "message": message}


# ---------- Student services tools ----------

def get_service_info(category: str) -> dict:
    category_lower = category.lower().strip()
    if category_lower == "library":
        return _load("library")
    services = _load("services")
    return services.get(category, {"error": f"Unknown service category: {category}"})


def file_grievance(student_id: str, category: str, description: str) -> dict:
    return {
        "status": "grievance_filed",
        "ticket_id": f"GRV-{abs(hash(student_id + description)) % 10000:04d}",
        "student_id": student_id,
        "category": category,
        "description": description,
        "sla": "Acknowledgement within 48 hours, resolution update within 15 working days",
    }
# ---------- Department tools ----------

def get_departments():
    return _load("departments")


def get_department(name: str):
    departments = _load("departments")

    for dept in departments:
        if (
            name.lower() in dept["department"].lower()
            or name.lower() == dept["short_name"].lower()
        ):
            return dept

    return {"error": "Department not found"}
# ---------- Faculty tools ----------

def get_faculty(department: str):
    faculty = _load("faculty")

    department = department.lower().strip()

    for key, value in faculty.items():
        if department in key.lower() or key.lower() in department:
            return value

    return {"error": "Faculty information not available"}
# ---------- Campus Navigation ----------

def get_building(name: str):

    buildings = _load("buildings")

    for building in buildings:

        if name.lower() in building["name"].lower():
            return building

    return {"error": "Building not found"}
# ---------- Institute Leadership ----------

def get_institute_info():
    """Institute-level info: name, establishment, principal, chairman, director, contact."""
    return _load("institute")


# ---------- Campus Life ----------

def get_induction_program():
    """Details of the orientation/induction programme for newly admitted students."""
    return _load("campus_life").get("induction_program", {"error": "No induction program data on file"})


def get_sports_info():
    """Sports facilities available on campus: grounds, courts, indoor games, fitness."""
    return _load("campus_life").get("sports", {"error": "No sports facilities data on file"})
# ---------- FAQ ----------

def search_faq(question: str):
    faqs = _load("campus_faq")

    question = question.lower()

    for faq in faqs:
        if any(word in faq["question"].lower() for word in question.split()):
            return faq

    return {"error": "No FAQ found"}