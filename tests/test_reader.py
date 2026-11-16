from __future__ import annotations

from datetime import date

from graph_memory.reader import BM25, Reader, parse_date, seeds
from graph_memory.store import Store


def store(rows: list[tuple[str, str, str, str]]) -> Store:
    s = Store()
    for subj, rel, obj, time in rows:
        for n in (subj, obj):
            s.entity(n.lower(), n, "thing")
        s.add_event(subj.lower(), rel, obj.lower(), time, "src")
    return s


def test_parse_date_formats():
    assert parse_date("2023-05-20") == (date(2023, 5, 20), "day")
    assert parse_date("2023/05/20 (Sat) 02:21") == (date(2023, 5, 20), "day")
    assert parse_date("7 May 2023") == (date(2023, 5, 7), "day")
    assert parse_date("May 7, 2023") == (date(2023, 5, 7), "day")
    assert parse_date("May 2023")[1] == "month"
    assert parse_date("in 2022")[1] == "year"
    assert parse_date("yesterday") is None


def test_router_cues():
    r = Reader(store([("the user", "visited", "Spain", "2023-01-01")]))
    assert r.ops("How many books did I read?")[0] == "collect"
    assert r.ops("When did I buy the bike?")[0] == "timeline"
    assert r.ops("Where do I live now?") == ["current"]
    assert r.ops("What did I cook for dinner?") == ["neighbor"]
    assert "join" in r.ops("Where does the person who introduced me to Bob work?")


def test_collect_lists_every_item_of_the_entity():
    rows = [("the user", "visited", c, f"2023-0{i + 1}-10") for i, c in enumerate(
        ["Spain", "Italy", "France", "Japan", "Peru", "Chile", "Kenya", "Egypt"])]
    rows += [("Alice", "visited", c, "2023-02-02") for c in ["Norway", "Sweden", "Finland"]]
    rows += [("the user", "likes", f"dish {i}", "2023-03-03") for i in range(30)]
    r = Reader(store(rows))
    text, meta = r.brief("Which countries have I visited?", 1500)
    for c in ["Spain", "Italy", "France", "Japan", "Peru", "Chile", "Kenya", "Egypt"]:
        assert c in text
    assert "Norway" not in text.split("Other related records")[0]
    assert "[8 distinct]" in text
    assert len(text) <= 1500 and meta["ops"][0] == "collect"


def test_current_gives_latest_and_value_at_an_event():
    rows = [("the user", "lives in", "Boston", "2022-01-05"), ("the user", "lives in", "Seattle", "2023-03-01"),
            ("the user", "lives in", "Denver", "2023-09-10"), ("the user", "adopted", "cat", "2023-05-01")]
    r = Reader(store(rows))
    lines = r.current("Where do I live now?", seeds(r.s, "Where do I live now?"))
    assert "latest Denver" in lines[0][0] and "Seattle" in lines[0][0]
    q = "Where did I live when I adopted my cat?"
    lines = r.current(q, seeds(r.s, q))
    assert any("at the time of" in ln and "Seattle" in ln for ln, _ in lines)


def test_timeline_computes_gaps():
    rows = [("the user", "visited", "MoMA", "2023-05-01"), ("the user", "saw", "Ancient Civilizations exhibit",
                                                                "2023-05-08"),
            ("the user", "ran", "charity 5K", "2023-06-01")]
    r = Reader(store(rows))
    q = "How many days passed between my visit to MoMA and the Ancient Civilizations exhibit?"
    text, _ = r.brief(q, 2000)
    assert ": 7 days" in text
    q2 = "How many days ago did I run the charity 5K?"
    text2, _ = r.brief(q2, 2000, now="2023/06/11 (Sun) 10:00")
    assert "to now (2023-06-11): 10 days" in text2


def test_join_reaches_the_unnamed_entity():
    rows = [("Dan", "introduced the user to", "Bob", "2022-04-01"), ("Dan", "works at", "Acme", "2021-01-01"),
            ("Bob", "works at", "Globex", "2021-01-01"), ("Carol", "works at", "Initech", "2021-01-01")]
    r = Reader(store(rows))
    q = "Where does the person who introduced me to Bob work?"
    lines = r.join(q, seeds(r.s, q))
    text = "\n".join(ln for ln, _ in lines)
    assert "Dan introduced the user to Bob" in text and "Dan works at Acme" in text


def test_brief_reports_shown_facts_and_keeps_budget():
    rows = [("the user", "read", f"book {i}", f"2023-01-{i + 1:02d}") for i in range(25)]
    r = Reader(store(rows))
    text, meta = r.brief("How many books did I read?", 300)
    assert len(text) <= 300 and meta["shown"] and all(0 <= i < 25 for i in meta["shown"])


def test_bm25_prefers_matching_lines():
    bm = BM25(["the user visited spain", "the user likes tea", "bob visited norway"])
    sc = bm.scores("which countries did I visit")
    assert sc[0] > sc[1]
