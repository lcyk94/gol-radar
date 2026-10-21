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
