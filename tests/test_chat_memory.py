import json
from types import SimpleNamespace

from graph_memory.chat import ChatMemory, session_record, turn_texts


class FakeLLM:

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def chat(self, messages, **kw):
        self.prompts.append(messages)
        return SimpleNamespace(content=json.dumps(self.replies.pop(0)), usage={}, finish_reason="stop")


def state(slot, value, steps=(1,), status="set"):
    return {"subject": "the user", "slot": slot, "value": value, "status": status, "time": "", "steps": list(steps)}


S1 = [{"role": "user", "content": "I take Quelmithin for my lactose intolerance. If my health condition changes, my "
                                  "medication will be a multivitamin. My weekly tutoring session also depends on it."},
      {"role": "assistant", "content": "Noted, thanks for telling me."}]
S2 = [{"role": "user", "content": "Update: my doctor now says I have high blood pressure."},
      {"role": "assistant", "content": "Thanks, I will keep that in mind."}]
F1 = {"start_place": "", "start_have": [],
      "states": [state("health condition", "lactose intolerance"), state("medication", "Quelmithin"),
                 state("regular appointment", "weekly tutoring session")],
      "rules": [{"subject": "the user", "slot": "medication", "depends_on": "health condition",
                 "depends_on_value": "lactose intolerance", "new_value": "multivitamin", "steps": [1]},
                {"subject": "the user", "slot": "regular appointment", "depends_on": "health condition",
                 "depends_on_value": "", "new_value": "", "steps": [1]}]}
F2 = {"start_place": "", "start_have": [], "states": [state("health condition", "high blood pressure")], "rules": []}


async def test_sessions_to_propagated_answers(tmp_path):
    llm = FakeLLM([F1, F2])
    mem = ChatMemory(llm)
    await mem.add_session(S1, "2024/03/01 (Fri) 10:00")
    await mem.add_session(S2, "2024/03/09 (Sat) 09:30")
    assert "medication: Quelmithin" in llm.prompts[1][0]["content"]
    text = mem.recall("What medication am I taking?")
    assert "multivitamin (now)" in text and "lactose intolerance → high blood pressure" in text
    assert "unknown now" in mem.recall("When is my regular appointment?")
    retrieved = mem.recall("What medication am I taking?", propagate=False)
    assert "Quelmithin" in retrieved and "multivitamin (now)" not in retrieved
    path = tmp_path / "memory.json"
    mem.save(path)
    again = ChatMemory.load(path)
    assert again.recall("What medication am I taking?") == text and again.sessions == 2


def test_session_record_cuts_assistant_turns():
    rec = session_record([{"role": "user", "content": "u" * 500}, {"role": "assistant", "content": "a" * 500}],
                         "2024-01-01")
    assert rec.header == "Chat session dated 2024-01-01 between the user and an AI assistant."
    assert len(rec.turns[0].observation) == 500 and len(rec.turns[1].observation) == 300
    assert turn_texts(rec)[0].startswith("chat session dated")
