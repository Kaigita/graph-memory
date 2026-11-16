from graph_memory.chat import add_facts, parts, world
from graph_memory.extract import PARTS, PARTS_S1, PARTS_S2, SLOT_M1, SLOT_S2, add_extracted, canon_slot, prompt, validate
from graph_memory.reader import Reader
from graph_memory.store import Store, fact_lines


def st(slot, value, status="set", steps=(1,), time="", subject="the user"):
    return {"subject": subject, "slot": slot, "value": value, "status": status, "time": time, "steps": list(steps)}


def rule(slot, on, new="", on_value="", steps=(1,), subject="the user", said=None):
    r = {"subject": subject, "slot": slot, "depends_on": on, "depends_on_value": on_value, "new_value": new,
         "steps": list(steps)}
    if said is not None:
        r["said"] = said
    return r


def rec(states=(), rules=()):
    return {"start_place": "", "start_have": [], "states": list(states), "rules": list(rules)}


def personal(change=True, restate=False):
    s = Store()
    add_extracted(s, rec([st("health condition", "lactose intolerance", time="2023-03-07"),
                          st("partner", "James", time="2023-03-03")]), "e#1")
    add_extracted(s, rec(
        [st("medication", "Quelmithin", steps=(1,), time="2023-03-17"),
         st("regular appointment", "tutoring session (weekly)", steps=(3,), time="2023-03-17"),
         st("exercise routine", "swimming 2x/week", steps=(5,), time="2023-03-17"),
         st("fitness facility", "Crysthene Pool", steps=(7,), time="2023-03-17")],
        [rule("medication", "health condition", on_value="lactose intolerance", steps=(1,)),
         rule("regular appointment", "health condition", on_value="lactose intolerance", steps=(3,)),
         rule("exercise routine", "health condition", on_value="lactose intolerance", steps=(5,)),
         rule("fitness facility", "exercise routine", on_value="swimming 2x/week", steps=(7,)),
         rule("exercise routine", "health condition", new="cycling weekends", steps=(9,)),
         rule("medication", "health condition", new="multivitamin", steps=(11,)),
         rule("fitness facility", "exercise routine", new="Velthari Studio", steps=(13,))]), "e#2")
    if change:
        add_extracted(s, rec([st("health condition", "lactose intolerance", "ended", time="2023-03-19"),
                              st("health condition", "high blood pressure", time="2023-03-19"),
                              st("partner", "James", "forget", steps=(5,), time="2023-03-19")]), "e#3")
    if restate:
        add_extracted(s, rec([st("medication", "aspirin", time="2023-03-21")]), "e#4")
    return s


def key(s, slot):
    return ("user", canon_slot(s, slot))


def test_store_slots_rules_forget():
    s = personal()
    assert {s.slot_names[e["slot"]] for e in s.events if e.get("slot")} >= {"medication", "health condition"}
    assert len(s.rules) == 7 and all(r["seq"] > 0 for r in s.rules)
    assert not any(s.name(e["o"]) == "James" for e in s.events)
    assert any(e.get("status") == "forget" and e["o"] == "" for e in s.events)
    lines = fact_lines(s)
    assert "the user asked to remove the partner from memory (2023-03-19)" in lines
    assert any("depends on the user's health condition (lactose intolerance)" in ln for ln in lines)
    assert not any("James" in ln for ln in lines)


def test_propagate_one_and_two_hops():
    s = personal()
    r = Reader(s)
    med = r.resolve(key(s, "medication"))
    assert med["value"] == "multivitamin" and not med["unknown"]
    app = r.resolve(key(s, "regular appointment"))
    assert app["value"] is None and app["unknown"]
    fac = r.resolve(key(s, "fitness facility"))
    assert fac["value"] == "Velthari Studio" and fac["why"]["kind"] == "derived"
    line = r.state_line(key(s, "fitness facility"))
    assert "Velthari Studio (now)" in line and "lactose intolerance → high blood pressure" in line
    assert "unknown now" in r.state_line(key(s, "regular appointment"))
    text, meta = r.brief("What medication am I taking?", 6000)
    assert meta["slots"][0] == "medication" and "multivitamin (now)" in text.split("\n")[1]


def test_before_change_and_restated():
    s = personal(change=False)
    r = Reader(s)
    assert s.name(r.resolve(key(s, "medication"))["value"]) == "Quelmithin"
    assert r.resolve(key(s, "regular appointment"))["changed"] is None
    s2 = personal(restate=True)
    assert s2.name(Reader(s2).resolve(key(s2, "medication"))["value"]) == "aspirin"


def test_retrieve_only_and_deleted():
    s = personal()
    g0 = Reader(s, propagate=False)
    assert s.name(g0.resolve(key(s, "medication"))["value"]) == "Quelmithin"
    line = g0.retrieve_line(key(s, "fitness facility"))
    assert "exercise routine" in line and "high blood pressure" in line and "(now)" not in line
    text, _ = Reader(s).brief("What's my partner's name?", 6000)
    assert "James" not in text and "removed from memory" in text


def test_source_by_value():
    s = Store()
    add_extracted(s, rec([st("testing tool", "Tarvex Verify"), st("framework", "Synthari")]), "e#1")
    add_extracted(s, rec([st("test command", "tarvex verify --integration")],
                         [rule("testing tool", "framework", on_value="Synthari"),
                          rule("test command", "test framework", on_value="Tarvex Verify")]), "e#2")
    add_extracted(s, rec([st("framework", "Veltrion")]), "e#3")
    cmd = Reader(s).resolve(key(s, "test command"))
    assert cmd["unknown"] and cmd["why"]["kind"] == "derived"


def test_rule_references_aligned():
    s = Store()
    add_extracted(s, rec([st("office location", "Quivira Springs"), st("commute duration", "about 40 min"),
                          st("authentication method", "mTLS"), st("auth middleware", "proxy v4")],
                         [rule("commute duration", "office location", on_value="Quivira Springs"),
                          rule("commute duration", "work location", new="around 10 min"),
                          rule("auth middleware", "authentication method", on_value="mTLS"),
                          rule("middleware config", "auth method", new="session middleware v1.8")]), "e#1")
    add_extracted(s, rec([st("office location", "Luminara Flats"), st("authentication method", "cookies")]), "e#2")
    r = Reader(s)
    assert r.resolve(key(s, "commute duration"))["value"] == "around 10 min"
    assert r.resolve(key(s, "auth middleware"))["value"] == "session middleware v1.8"


def test_event_store_reads_without_slots():
    s = Store()
    s.add_event("user", "lives in", "boston", "2023-05-01", "x:[1]")
    assert "rules" not in s.to_json() and "slot" not in s.events[0]
    r = Reader(s)
    assert r.keys == [] and "slots" not in r.brief("where do I live", 4000)[1]


def test_validate_states_rules():
    data = validate(rec([st("medication", "Quelmithin")], [rule("medication", "health condition")]),
                    ("states", "rules"), 3)
    assert data["states"][0]["slot"] == "medication"


TEXTS = ["chat session dated 2024-01-01 between the user and an ai assistant.",
         "i cycle everywhere, i never drive.", "great, cycling is healthy.",
         "my doctor says i cannot put weight on my wrist for two months.", "sorry to hear that.",
         "we just finished unpacking at the new apartment.", "congrats!"]


def base():
    s = Store()
    add_facts(s, rec([st("transportation preference", "cycling for basically everything instead of driving")],
                     [rule("transportation preference", "physical health", said="no")]), "x#0", "2024-01-01", TEXTS,
              "s1")
    return s


def test_prompt_versions():
    assert prompt("w", ("states", "rules"), Store()) == prompt("w", ("states", "rules"), Store(), PARTS)
    assert PARTS_S1["states"][1].startswith(PARTS["states"][1]) and PARTS_S1["rules"][1].startswith(PARTS["rules"][1])
    assert "said" in PARTS_S1["rules"][0] and "said" not in PARTS["rules"][0]
    assert parts("m1") is None and parts("s1") is PARTS_S1 and parts("s2") is PARTS_S2
    assert PARTS_S2["rules"] == PARTS_S1["rules"]
    assert PARTS_S2["states"][1] == PARTS_S1["states"][1].replace(SLOT_M1, SLOT_S2)


def test_world_notes_list_known_slots_and_dependency_kinds():
    s = base()
    assert "Kinds of value that known slots depend on" not in world(s, "m1")
    w = world(s, "s1")
    assert "transportation preference: cycling for basically everything instead of dr" in w
    assert w.endswith("physical health")


def test_implied_value_kept_copy_dropped_header_dropped():
    s = base()
    add_facts(s, rec([st("physical health", "left wrist injured, no weight for two months", steps=(3,)),
                      st("home city", "a new apartment (city not named)", steps=(5,)),
                      st("transportation preference", "cycling for basically everything instead of driving",
                         steps=(5,)),
                      st("hobby", "chess", steps=(0,))]), "x#1", "2024-02-01", TEXTS, "s1")
    vals = [s.name(e["o"]) for e in s.events if e.get("slot")]
    assert "left wrist injured, no weight for two months" in vals
    assert "a new apartment (city not named)" in vals
    assert vals.count("cycling for basically everything instead of driving") == 1
    assert "chess" not in vals


def test_ended_on_known_slot_maps_to_current_value():
    s = base()
    add_facts(s, rec([st("transportation preference", "cycling for basically everything instead of driv…",
                         "ended", steps=(3,))]), "x#1", "2024-02-01", TEXTS, "s1")
    r = Reader(s)
    out = r.resolve(key(s, "transportation preference"))
    assert out["value"] is None and out["status"] == "ended"
    assert "none now" in r.state_line(key(s, "transportation preference"))


def test_propagation_along_inferred_rule_and_switch():
    s = base()
    add_facts(s, rec([st("physical health", "left wrist injured", steps=(3,))]), "x#1", "2024-02-01", TEXTS, "s1")
    assert s.rules[0]["said"] == "no"
    g = Reader(s).state_line(key(s, "transportation preference"))
    assert g.startswith("the user's transportation preference: unknown now")
    off = Reader(s, inferred=False).state_line(key(s, "transportation preference"))
    assert "unknown now" not in off
    assert any("depends on the user's physical health" in ln for ln in fact_lines(s))


def test_m1_rules_have_no_said_field():
    s = Store()
    add_extracted(s, rec([st("medication", "X")], [rule("medication", "health condition")]), "e#1")
    assert "said" not in s.rules[0]
