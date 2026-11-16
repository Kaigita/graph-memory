import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_memory import ChatMemory, session_record


def state(slot, value, step=1, status="set"):
    return {"subject": "the user", "slot": slot, "value": value, "status": status, "time": "", "steps": [step]}


def rule(slot, depends_on, value="", new="", step=1):
    return {"subject": "the user", "slot": slot, "depends_on": depends_on, "depends_on_value": value,
            "new_value": new, "steps": [step]}


SESSIONS = [
    ("2024/03/01 (Fri) 10:00",
     [{"role": "user", "content": "I take Quelmithin for my lactose intolerance. If my health condition changes, my "
                                  "medication will be a multivitamin."},
      {"role": "assistant", "content": "Noted."},
      {"role": "user", "content": "My pull requests are reviewed by Dana. The reviewer is assigned by my team lead, "
                                  "Priya, so if the team lead changes, the reviewer would likely change too."},
      {"role": "assistant", "content": "Got it."}],
     {"start_place": "", "start_have": [],
      "states": [state("health condition", "lactose intolerance"), state("medication", "Quelmithin"),
                 state("team lead", "Priya", 3), state("code reviewer", "Dana", 3)],
      "rules": [rule("medication", "health condition", "lactose intolerance", "multivitamin"),
                rule("code reviewer", "team lead", "Priya", step=3)]}),
    ("2024/03/09 (Sat) 09:30",
     [{"role": "user", "content": "Update: my doctor now says I have high blood pressure. Also, Marco replaced Priya "
                                  "as my team lead."},
      {"role": "assistant", "content": "Thanks for the update."}],
     {"start_place": "", "start_have": [], "rules": [],
      "states": [state("health condition", "high blood pressure"), state("team lead", "Marco")]}),
]
QUESTIONS = ["What medication am I taking?", "Who is my code reviewer now?"]


def main():
    mem = ChatMemory()
    for when, turns, facts in SESSIONS:
        mem.add_facts(facts, session_record(turns, when), when)
    for q in QUESTIONS:
        print(f"Q: {q}\n")
        print("worked out (G):")
        print(mem.recall(q, budget=700))
        print("\nretrieved only (Gr):")
        print(mem.recall(q, budget=700, propagate=False))
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
