from graph_benchmarks import locomo as LOCOMO
from graph_benchmarks import longmemeval as LME
from graph_benchmarks import meme as MEME
from graph_benchmarks import stale as STALE
from graph_benchmarks.baselines import bm25, cap
from graph_memory.reader import parse_date


def test_locomo_porter_and_official_scoring():
    for w, s in {"caresses": "caress", "ponies": "poni", "agreed": "agre", "motoring": "motor", "happy": "happi",
                 "relational": "relat", "generalization": "gener", "adoption": "adopt", "replacement": "replac"}.items():
        assert LOCOMO.PS.stem(w) == s
    assert LOCOMO.score("7 May 2023", "7 May, 2023", 2) == 1.0
    assert LOCOMO.score("pottery, camping", "pottery, camping, painting", 1) == 2 / 3
    assert LOCOMO.score("likely no", "Likely no; she wants to be a counselor", 3) == 1.0
    assert LOCOMO.score("Not mentioned.", "anything", 5) == 1.0 and LOCOMO.score("yes", "anything", 5) == 0.0


def test_longmemeval_time_fixes():
    day = "2023/05/20"
    assert LME.v3_time("2022-07 (last summer)", day) == "July 2022 (last summer)"
    assert parse_date(LME.v3_time("2022-07 (last summer)", day))[1] == "month"
    assert LME.v3_time("2023-05-13 (last week)", day) == "2023-05-13 (last week)"
    assert LME.v3_time("2022-2023", day) == "2022-2023"
    assert LME.v3_time("last week", day) == "last week (said 2023/05/20)"
    assert LME.v3_time("", day) == "2023/05/20 (said)"
    assert LME.shift_time("2023-12-02 (this weekend)", -185) == "2023-05-31 (this weekend)"
    assert LME.shift_time("May 7, 2023", 3) == "2023-05-10"
    assert LME.shift_time("July 2022 (last summer)", 77) == "September 2022 (last summer)"
    assert LME.shift_time("2020 (a few years ago)", 300) == "2020 (a few years ago)"


def test_stale_parse_judge():
    good = '```json\n{"dim1_eval": {"reasoning": "x", "pass": true}, "dim2_eval": {"pass": false}, ' \
           '"dim3_eval": {"pass": "true"}}\n```'
    assert STALE.parse_judge(good) == {"dim1": True, "dim2": False, "dim3": True}
    assert STALE.parse_judge('{"dim1_eval": {"pass": true}}') is None
    assert STALE.parse_judge("no json") is None


def test_meme_verdicts_and_trivial_pass_filter():
    it = {"task": "Cas", "entity_values": {"medication": "multivitamin"}}
    assert MEME.parse_verdict(it, '```json\n{"correct": true, "reason": "x"}\n```')
    assert not MEME.parse_verdict(it, "not json")
    agg = {"task": "Agg", "entity_values": {"a": "x", "b": "y"}}
    assert MEME.parse_verdict(agg, '{"results": [{"value": "x", "present": true}, {"value": "y", "present": true}]}')
    rows = [{"unit": "u", "entity": "e", "phase": "before", "task": "Cas", "pass": 0},
            {"unit": "u", "entity": "e", "phase": "after", "task": "Cas", "pass": 1}]
    assert MEME.score(rows)["Cas"] == 0.0


def test_baseline_ranking():
    lines = ["the user likes tea", "the user moved to Denver", "Bob visited Norway"]
    assert bm25(lines, "where has the user moved")[0] == "the user moved to Denver"
    assert bm25(lines, "Norway") == ["Bob visited Norway"]
    assert cap(lines, 30) == "- the user likes tea"
