"""Gol Radar testleri. Veri SENTETİKTİR (Nesine'nin gerçek JSON şeklini taklit eder); canlı Nesine'ye bağlanmaz."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import nesine, bot


def _ev(i, home="A", away="B", ms=(2.0, 3.2, 3.5), extra=True, broken=False):
    ma = [{"ID": 1, "NO": 1, "MTID": 1, "MV": 1, "SOV": 0, "MBS": 1,
           "OCA": [{"N": n + 1, "O": o} for n, o in enumerate(ms)]},
          {"ID": 2, "NO": 2, "MTID": 12, "SOV": 2.5, "MBS": 1,
           "OCA": [{"N": 1, "O": 1.85}, {"N": 2, "O": 1.85}]}]
    if extra:
        ma.append({"ID": 3, "NO": 3, "MTID": 450, "SOV": 0, "MBS": 1,
                   "OCA": [{"N": 1, "O": 2.5}, {"N": 2, "O": 1.5}]})
    if broken:
        ma[0]["OCA"] = [{"N": 1, "O": None}, {"N": 2, "O": "x"}]
    return {"C": i, "EV": f"{home}-{away}", "HN": home, "AN": away, "TYPE": 1, "D": "21.10.2026",
            "T": "20:00", "LC": "Test Ligi", "MA": ma}


def _dom(evs):
    """Ekrandan okunmuş maçları taklit eder (kalibrasyon için)."""
    out = []
    for e in evs:
        ms = [o["O"] for o in e["MA"][0]["OCA"]]
        out.append({"time": e["T"], "home": e["HN"], "away": e["AN"], "ms": ms, "ou25": [1.85, 1.85], "kg": [None, None],
                    "cs": [None] * 3, "hnd": {"line": "", "odds": [None] * 3}})
    return out


def _run(evs):
    """Kalibrasyon en az 3 ekran eşleşmesi ister; dolgu maçları eklenir ve sonuçtan ayıklanır."""
    fill = [_ev(900 + i, f"F{i}", f"G{i}", ms=(2.0 + i / 10, 3.2, 3.5)) for i in range(4)]
    for f in fill:
        f["T"] = f"1{i}:00" if False else "19:00"
    allev = list(evs) + fill
    out = nesine.matches_from_prebulten(_body(allev), _dom(allev))
    return [m for m in out if m.get("id") not in {f["C"] for f in fill}]


def _body(evs):
    return json.dumps({"sg": {"EA": evs, "LA": []}})


def test_all_markets_captured():
    ms = _run([_ev(1)])
    assert len(ms) == 1 and len(ms[0]["markets"]) == 3


def test_duplicate_matches_collapsed():
    ms = _run([_ev(1), _ev(1)])
    assert len(ms) == 1


def test_missing_or_corrupt_odds_do_not_crash():
    ms = _run([_ev(2, broken=True)])
    assert isinstance(ms, list)


def test_non_football_ignored():
    e = _ev(3); e["TYPE"] = 2
    assert _run([e]) == []


def test_odds_history_appends_only_on_change(tmp_path):
    p = tmp_path / "h.jsonl"
    ms = _run([_ev(1)])
    assert nesine.append_history(ms, "t1", p) == 1
    assert nesine.append_history(ms, "t2", p) == 0
    ms2 = _run([_ev(1, ms=(1.9, 3.3, 3.8))])
    assert nesine.append_history(ms2, "t3", p) == 1
    assert len(p.read_text().splitlines()) == 2


def test_history_survives_corrupt_line(tmp_path):
    p = tmp_path / "h.jsonl"; p.write_text("{bozuk\n")
    ms = _run([_ev(1)])
    assert nesine.append_history(ms, "t", p) == 1


def test_titles_from_text():
    ev = {"markets": [{"mtid": 450, "line": 0.0, "outs": [{"o": 2.5}, {"o": 1.5}]}]}
    got = nesine._titles_from_text("Maç\nİlk Yarı Sonucu\n2.50\n1.50\n", ev)
    assert got.get("450|0") == ["İlk Yarı Sonucu"]


def test_fitted_probabilities_valid():
    q = bot._fit_lambdas((0.45, 0.28, 0.27, 0.5))
    assert abs(q["p1"] + q["px"] + q["p2"] - 1) < 1e-6
    assert all(0 <= q[k] <= 1 for k in ("p1", "px", "p2", "over25"))


def test_league_name_from_LA_and_no_ms_kept():
    e = _ev(7); e["LC"] = 2
    e["MA"][0]["OCA"] = [{"N": 1, "O": None}, {"N": 2, "O": None}, {"N": 3, "O": None}]
    fill = [_ev(900 + i, f"F{i}", f"G{i}", ms=(2.0 + i / 10, 3.2, 3.5)) for i in range(4)]
    allev = [e] + fill
    body = json.dumps({"sg": {"EA": allev, "LA": [{"LID": 2, "N": "UEFA ŞAMPİYONLAR LİGİ", "CC": "EU"}]}})
    dom = _dom(fill)
    out = nesine.matches_from_prebulten(body, dom)
    m = [x for x in out if x["id"] == 7]
    assert m and m[0]["league"] == "UEFA ŞAMPİYONLAR LİGİ"


CC = ('x;OutcomeGroups:{1:{1:{Title:"Maç Sonucu",CouponTitle:"Maç Sonucu",Value:[{id:1,title:"MS 1",couponTitle:"1"},{id:2,title:"MS X",couponTitle:"X"},{id:3,title:"MS 2",couponTitle:"2"}],BultenValue:[{id:1,title:"1",couponTitle:"1"},{id:2,title:"X",couponTitle:"X"},{id:3,title:"2",couponTitle:"2"}]},'
      '5:{Title:"İlk Yarı/Maç Sonucu",CouponTitle:"İY/MS",Value:[{id:1,title:"1/1",couponTitle:"1/1"},{id:2,title:"1/X",couponTitle:"1/X"}],BultenValue:[{id:1,title:"1/1",couponTitle:"1/1"},{id:2,title:"1/X",couponTitle:"1/X"}]}},2:{9:{Title:"Basket",CouponTitle:"B",Value:[{id:1,title:"a"}],BultenValue:[{id:1,title:"a"}]}}}')
BL = ('{title:"Handikaplı Maç Sonucu",outcomeGroups:[{title:"Handikaplı Maç Sonucu",key:"268",liveKey:"60",specialSelector:"HM",'
      'outcomeMasks:[{title:"HND",handicap:!0,handicapType:1},{title:"HMS 1",outcomeID:"1"},{title:"HMS X",outcomeID:"2"},{title:"HMS 2",outcomeID:"3"}]}]}')


def test_parse_market_defs():
    d = nesine.parse_market_defs(CC, BL)
    assert d["def|1"]["name"] == "Maç Sonucu" and d["def|1"]["labels"] == {1: "1", 2: "X", 3: "2"}
    assert d["def|5"]["name"] == "İlk Yarı/Maç Sonucu"
    assert d["def|268"]["name"] == "Handikaplı Maç Sonucu" and d["def|268"]["labels"][2] == "HMS X"
    assert "def|9" not in d                       # futbol dışı bloklar alınmaz


def test_markets_use_defs_without_inventing():
    d = nesine.parse_market_defs(CC, BL)
    e = _ev(1)
    e["MA"].append({"ID": 9, "NO": 9, "MTID": 268, "SOV": 1.0, "MBS": 1,
                    "OCA": [{"N": 1, "O": 1.47}, {"N": 2, "O": 3.3}, {"N": 3, "O": 3.28}]})
    e["MA"].append({"ID": 10, "NO": 10, "MTID": 7777, "SOV": 0, "MBS": 1, "OCA": [{"N": 1, "O": 2.0}, {"N": 2, "O": 2.0}]})
    mk = nesine._markets_full(e, d)
    by = {m["mtid"]: m for m in mk}
    assert by[268]["name"].startswith("Handikaplı Maç Sonucu") and by[268]["labels"] == ["HMS 1", "HMS X", "HMS 2"]
    assert by[7777]["name"] is None


def test_results_feed_and_store(tmp_path):
    feed = json.dumps({"sc": 200, "d": [
        {"C": 11, "S": 4, "ES": [{"H": 2, "A": 0, "T": 1}, {"H": 1, "A": 0, "T": 2}]},
        {"C": 12, "S": 2, "ES": [{"H": 0, "A": 0, "T": 1}]}, {"C": None}, {"x": 1}]})
    f = nesine.results_from_feed(feed)
    assert f[11] == {"s": 4, "t": {1: [2, 0], 2: [1, 0]}} and 12 in f and len(f) == 2
    js = [{"id": 11, "home": "A", "away": "B", "date": "2099-01-01", "league": "L"}]
    p = tmp_path / "r.json"
    ms, rs = nesine.update_result_store(js, f, p)
    assert rs["11"]["t"]["1"] == [2, 0] and "12" not in rs
    assert nesine.results_from_feed("bozuk") == {}


def test_coupon_graded_by_nid(tmp_path, monkeypatch):
    import pandas as pd
    p = tmp_path / "r.json"
    nesine.update_result_store([{"id": 5, "home": "Leipzig", "away": "X", "date": "2026-10-21", "league": "L"}],
                               {5: {"s": 5, "t": {1: [2, 1]}}}, p)
    monkeypatch.setattr(nesine, "RESULT_FILE", p)
    entry = {"list": [{"odds": 3.0, "p": 0.3, "legs": [{"home": "RB Leipzig", "away": "X", "sel": "MS 1", "nid": 5}]}]}
    byid, _ = bot.nesine_results()
    out = bot._grade_day("2026-10-21", entry, {}, byid)
    assert out[0]["legs"][0]["ok"] is True and out[0]["status"] == "won"


def test_import_raw_without_screen(tmp_path):
    evs = [_ev(i, f"H{i}", f"A{i}", ms=(2.0 + i / 10, 3.2, 3.5)) for i in range(1, 5)]
    evs[0]["LC"] = 5
    raw = {"fetched": "2026-10-22T05:00:00+00:00",
           "prebulten": json.dumps({"sg": {"EA": evs, "LA": [{"LID": 5, "N": "Test Ligi 5", "CC": "TR"}]}})}
    p, o = tmp_path / "ham.json", tmp_path / "n.json"
    p.write_text(json.dumps(raw))
    assert nesine.import_raw(p, o) == 0
    d = json.loads(o.read_text())
    assert len(d["matches"]) == 4 and d["matches"][0]["league"] == "Test Ligi 5"
    assert d["matches"][0]["ms"] == [2.1, 3.2, 3.5] and d["fetched"] == raw["fetched"]


def test_import_raw_bad_file(tmp_path):
    p = tmp_path / "x.json"; p.write_text("bozuk")
    assert nesine.import_raw(p, tmp_path / "o.json") == 1
