import math
"""Kupon motoru testleri. Veri SENTETİKTİR: kurgusal takımlar, Nesine'nin pazar yapısını taklit eden oranlar."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
import bot, pazarlar as pz


def synth(days=3, seed=3):
    rng = np.random.default_rng(seed)
    hist = bot.demo_history()
    hist["HTHG"] = rng.binomial(hist["FTHG"].astype(int), 0.44)
    hist["HTAG"] = rng.binomial(hist["FTAG"].astype(int), 0.44)
    shares = pz.half_shares(hist)
    rows = []
    for k, (r, p, _) in enumerate(bot.demo_rows(days)):
        ctx = pz.Ctx(p, pz.shares_for(shares, r["Div"]))
        mks = []
        for mtid, line in bot.STD_MARKETS:
            ps = pz.probs(ctx, mtid, line)
            if ps is None:
                continue
            outs = []
            for i, pi in enumerate(ps):
                q = pi * rng.normal(1, .06) * 0.93          # piyasa payı ve gürültü
                outs.append({"n": i + 1, "o": round(max(1 / q, 1.01), 2) if q > 0.03 else None})
            mks.append({"mtid": mtid, "line": float(line), "mbs": 1, "outs": outs, "name": None, "labels": None})
        r = r.copy()
        r["NESACT"], r["SRC"], r["NID"] = True, "model", 1000 + k
        r["NES"] = {"mbs": 1, "markets": mks, "ms": [None] * 3, "ou25": [None] * 2, "cs": [None] * 3, "kg": [None] * 2}
        rows.append((r, p, None))
    return hist, rows


def _env(hist, rows, gate_all=True):
    perf = {"fam": [{"fam": f, "n": 99, "p": .6, "hit": .6, "ok": True} for f in
                    sorted({v["fam"] for v in pz.SPECS.values()})]} if gate_all else None
    return bot.setup_env(rows, hist, perf)


def test_markets_validate_against_synthetic_odds():
    hist, rows = synth()
    v = _env(hist, rows)
    assert v and all(x["ok"] for x in v.values()), {k: x for k, x in v.items() if not x["ok"]}


def test_wrong_option_order_is_rejected():
    hist, rows = synth()
    for r, p, _ in rows:                       # Alt/Üst seçeneklerinin sırasını boz
        for mk in r["NES"]["markets"]:
            if mk["mtid"] == 13:
                mk["outs"] = list(reversed([{**o} for o in mk["outs"]]))
                for i, o in enumerate(mk["outs"]):
                    o["n"] = i + 1
    v = _env(hist, rows)
    assert not v[13]["ok"] and v[1]["ok"]


def test_coupon_rules_and_variety():
    hist, rows = synth(days=3)
    _env(hist, rows)
    res = bot.build_coupons(rows)
    assert res["list"], "kupon çıkmadı"
    seen, fams = set(), []
    for k in res["list"]:
        assert len(k["legs"]) == bot.COUPON_LEGS and k["odds"] >= bot.COUPON_MIN_ODDS - 1e-9
        for l in k["legs"]:
            key = (l["home"], l["away"])
            assert key not in seen                          # her kupon farklı maçlar
            seen.add(key)
            assert l["o"] >= bot.COUPON_MIN_LEG_ODDS and l["p"] >= bot.COUPON_MIN_P
            assert "Alt" not in l["sel"].split() and l["why"] and l["edge"] >= bot.MIN_EDGE
            fams.append(l["fam"])
    assert len(set(fams)) >= 3, fams                       # çifte şansa sıkışmaz


def test_selection_ignores_odds_level():
    """Aynı olasılıklar, bütün oranlar 1.5 kat artırılınca da aynı seçenekleri verir (oran yalnızca süzgeç)."""
    hist, rows = synth(days=2, seed=5)
    _env(hist, rows)
    a = bot._legs(rows[0][0], rows[0][1])
    for r, p, _ in rows:
        for mk in r["NES"]["markets"]:
            for o in mk["outs"]:
                if o["o"]:
                    o["o"] = round(o["o"] * 1.5, 2)
    b = bot._legs(rows[0][0], rows[0][1])
    ea, eb = {x[0]: x[3]["edge"] for x in a}, {x[0]: x[3]["edge"] for x in b}
    common = set(ea) & set(eb)
    assert common and all(abs(ea[k] - eb[k]) < 1e-12 for k in common)   # puan oranlardan etkilenmez


def test_grade_matches_probability_definition():
    for mtid, line, n in ((1, 0, 3), (343, 2.5, 6), (5, 0, 9), (416, 0, 6), (43, 0, 4)):
        for hg, ag, h1, a1 in ((2, 1, 1, 0), (0, 0, 0, 0), (1, 3, 1, 1), (4, 2, 2, 1)):
            hits = [pz.outcome(mtid, line, i, hg, ag, h1, a1) for i in range(n)]
            assert sum(hits) == 1, (mtid, hg, ag, hits)    # dışlayan pazarlarda tam bir seçenek tutar
    assert pz.outcome(7, 0, 0, 2, 1) is None               # yarı skoru yoksa sonuç bilinmez


def test_probs_sum_to_one_for_exclusive_markets():
    p = bot.predict_lambdas(1.7, 1.1)
    cx = pz.Ctx(p, (.44, .44))
    for mtid, line in bot.STD_MARKETS:
        if pz.SPECS[mtid]["fam"] in ("ÇŞ", "İY ÇŞ", "2Y ÇŞ"):
            continue
        assert abs(sum(pz.probs(cx, mtid, line)) - 1) < 0.01, mtid


def test_grade_day_uses_market_and_half_scores():
    leg = {"home": "A", "away": "B", "sel": "İY 0,5 Üst", "mtid": 209, "line": 0.5, "idx": 1, "nid": 7, "o": 1.3, "p": .7}
    entry = {"list": [{"odds": 1.3, "p": .7, "legs": [leg]}]}
    g = bot._grade_day("2026-10-25", entry, {}, {7: (2, 1, 1, 0)})
    assert g[0]["status"] == "won"
    g = bot._grade_day("2026-10-25", entry, {}, {7: (2, 1, None, None)})      # yarı skoru yok: bekleniyor
    assert g[0]["status"] == "pending"
    g = bot._grade_day("2026-10-25", entry, {}, {7: (1, 0, 0, 0)})
    assert g[0]["status"] == "lost"


def test_negative_scores_do_not_crash():
    """Çeşitlilik cezası yüzünden en iyi puan eksiye düşse de kupon kurulur (sabah çalışması bu yüzden çökmüştü)."""
    hist, rows = synth(days=3, seed=9)
    _env(hist, rows)
    old = bot.FAM_PEN
    bot.FAM_PEN = 5.0
    try:
        fam_use = {f: 3 for f in {v["fam"] for v in pz.SPECS.values()}}
        items = [(k, r, bot._legs(r, p), p) for k, (r, p, _) in enumerate(rows[:12]) if bot._legs(r, p)]
        best = bot._best_combo(items, fam_use=fam_use)
        assert best is not None and best[0] < 0
    finally:
        bot.FAM_PEN = old


def _espn_payload():
    def ev(h, a, hs, as_, done=True, ls=None):
        mk = lambda side, name, sc, l: {"homeAway": side, "score": str(sc), "team": {"displayName": name, "shortDisplayName": name},
                                        "linescores": [{"value": v} for v in (l or [])]}
        return {"competitions": [{"status": {"type": {"completed": done}},
                                  "competitors": [mk("home", h, hs, ls[0] if ls else None), mk("away", a, as_, ls[1] if ls else None)]}]}
    return {"events": [ev("Sandefjord", "Sarpsborg 08", 2, 1, ls=([1, 1], [0, 1])),
                       ev("FC Copenhagen", "OB Odense", 3, 0, ls=([2, 1], [0, 0])),
                       ev("Baska", "Takim", 1, 1, done=False)]}


def test_espn_fallback_finds_scores_and_half_scores(monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "_http_json", lambda url, timeout=10: calls.append(url) or _espn_payload())
    pend = [("2026-10-26", {"home": "Sandefjord", "away": "Sarpsborg 08", "league": "Norveç Eliteserien"}),
            ("2026-10-26", {"home": "FC Copenhagen", "away": "Odense", "league": "Danimarka Süper Lig"}),
            ("2026-10-26", {"home": "Yok", "away": "Maç", "league": "?"})]
    out = bot.espn_scores(pend)
    assert out[("sandefjord", "sarpsborg 08")][1:] == (2, 1, 1, 0)
    assert out[("fc copenhagen", "odense")][1:] == (3, 0, 2, 0)
    assert ("yok", "maç") not in out
    assert any("soccer/all/scoreboard" in c for c in calls)


def test_fetch_unlive_stores_finished_results(tmp_path, monkeypatch):
    import json, datetime, nesine
    monkeypatch.setattr(nesine, "RESULT_FILE", tmp_path / "sonuc.json")
    body = json.dumps({"sc": 200, "d": [
        {"MID": 1, "C": 3241666, "S": 5, "ES": [{"T": 1, "H": 2, "A": 1}, {"T": 2, "H": 1, "A": 0}]},
        {"MID": 2, "C": 3241581, "S": 5, "ES": [{"T": 1, "H": 3, "A": 0}, {"T": 2, "H": 2, "A": 0}]},
        {"MID": 3, "C": 999, "S": 1, "ES": []}]})
    calls = []

    class R:
        status = 200
        def text(self): return body

    class Ctx:
        class request:
            @staticmethod
            def get(url, headers=None, timeout=None):
                calls.append(url)
                return R()
    js = [{"id": 3241666, "home": "Sandefjord", "away": "Sarpsborg 08", "date": "2026-10-26", "league": "x"},
          {"id": 3241581, "home": "FC Copenhagen", "away": "Odense", "date": "2026-10-26", "league": "x"}]
    n = nesine.fetch_unlive(Ctx, js, days=2, today=datetime.date(2026, 10, 27))
    assert n == 4 and len(calls) == 2 and "date=2026-10-26" in calls[1]
    st = nesine.load_results_store(tmp_path / "sonuc.json")
    assert st["results"]["3241666"]["t"]["1"] == [2, 1] and st["results"]["3241666"]["s"] == 5


def test_unlive_days_reaches_oldest_missing_leg(tmp_path, monkeypatch):
    import json, datetime, nesine
    monkeypatch.setattr(nesine, "DATA_DIR", tmp_path)
    monkeypatch.setattr(nesine, "RESULT_FILE", tmp_path / "s.json")
    (tmp_path / "kupon_gecmisi.json").write_text(json.dumps({
        "2026-10-20": {"list": [{"legs": [{"nid": 1}]}]}, "2026-10-26": {"list": [{"legs": [{"nid": 2}]}]}}))
    (tmp_path / "s.json").write_text(json.dumps({"matches": {}, "results": {"2": {"s": 24, "t": {"1": [1, 1]}}}}))
    assert nesine.unlive_days(datetime.date(2026, 10, 29)) == 10       # 20 Ekim'deki bacağın sonucu yok
    (tmp_path / "s.json").write_text(json.dumps({"matches": {}, "results": {"1": {"s": 5, "t": {"1": [1, 0]}}, "2": {"s": 24, "t": {"1": [1, 1]}}}}))
    assert nesine.unlive_days(datetime.date(2026, 10, 29)) == 3


def test_surprise_coupons():
    hist, rows = synth(days=3)
    _env(hist, rows)
    res = bot.build_coupons(rows)
    sur = res.get("surprise") or []
    assert len(sur) <= bot.N_SURPRISE
    keys = set()
    for k in sur:
        assert k["surprise"] and len(k["legs"]) == bot.COUPON_LEGS and k["odds"] >= bot.SURPRISE_MIN_ODDS - 1e-9
        for l in k["legs"]:
            assert l["p"] >= bot.COUPON_MIN_P and l["o"] >= bot.COUPON_MIN_LEG_ODDS
            assert (l["home"], l["away"]) not in keys       # iki sürpriz kupon farklı maçlardan
            keys.add((l["home"], l["away"]))
    # tutma olasılığı, aynı oran eşiğini sağlayan en iyi normal kupondan az olmamalı değil; yalnızca alan sağlaması:
    assert all(0 < k["p"] < 1 for k in sur)


def test_surprise_saved_and_graded(tmp_path, monkeypatch):
    hist, rows = synth(days=3)
    _env(hist, rows)
    res = bot.build_coupons(rows)
    monkeypatch.setattr(bot, "COUPON_LOG", tmp_path / "k.json")
    bot.save_coupons(res)
    saved = bot._load_log()
    n = sum(len(v["list"]) for v in saved.values())
    assert n == len(res["list"]) + len(res.get("surprise") or [])
    assert sum(1 for v in saved.values() for k in v["list"] if k.get("surprise")) == len(res.get("surprise") or [])


def test_todays_coupons_are_frozen(tmp_path, monkeypatch):
    hist, rows = synth(days=3)
    _env(hist, rows)
    monkeypatch.setattr(bot, "COUPON_LOG", tmp_path / "k.json")
    res = bot.build_coupons(rows)
    day = pd.Timestamp(res["date"]).date()
    monkeypatch.setattr(bot, "today_tr", lambda: day)
    bot.save_coupons(res)
    dd = day.strftime("%Y-%m-%d")
    today_cp = [k for k in res["list"] if k["date"] == dd]
    first = [[(l["home"], l["away"], l["sel"]) for l in k["legs"]] for k in today_cp]
    # maçlar bültenden düşse bile (başlayanlar) bugünün kuponları aynı kalır
    used = {(l["home"], l["away"]) for k in today_cp for l in k["legs"]}
    rows2 = [x for x in rows if (x[0]["HomeTeam"], x[0]["AwayTeam"]) not in used]
    res2 = bot.build_coupons(rows2)
    again = [[(l["home"], l["away"], l["sel"]) for l in k["legs"]] for k in res2["list"] if k["date"] == dd]
    assert again == first
    sig = lambda L: [[(l["home"], l["away"], l["sel"], l["o"]) for l in k["legs"]] for k in (L or [])]
    assert sig(res2.get("surprise")) == sig(res.get("surprise"))
    # ikinci kayıt günün kaydını bozmaz
    bot.save_coupons(res2)
    ent = bot._load_log()[day.strftime("%Y-%m-%d")]
    assert ent["saved_on"] == day.strftime("%Y-%m-%d")
    assert len([k for k in ent["list"] if not k.get("surprise")]) == len(today_cp)


def test_odds_moves_and_payload(tmp_path):
    import json, datetime as dt
    now = dt.datetime.utcnow()
    f = tmp_path / "o.jsonl"
    rows = [
        {"t": (now - dt.timedelta(hours=9)).isoformat(), "id": 7, "odds": {"ms": [2.0, 3.0, 3.5], "ou25": [1.9, 1.8], "kg": [1.7, 2.0]}},
        {"t": (now - dt.timedelta(hours=1)).isoformat(), "id": 7, "odds": {"ms": [1.8, 3.1, 4.0], "ou25": [1.9, 1.8], "kg": [1.7, 2.0]}},
        {"t": now.isoformat(), "id": 8, "odds": {"ms": [2.0, 3.0, 3.5], "ou25": None, "kg": None}},     # tek kayıt: hareket yok
        {"t": (now - dt.timedelta(days=30)).isoformat(), "id": 9, "odds": {"ms": [2.0, 3.0, 3.5]}},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    mv = bot.odds_moves(f)
    assert set(mv) == {7} and mv[7]["ms"][0] == [2.0, 1.8] and "ou25" not in mv[7]
    assert bot.odds_moves(tmp_path / "yok.jsonl") == {}
    hist, rws = synth(days=3)
    _env(hist, rws)
    pl = bot.site_payload(rws, demo=True, perf=None, nes_info=None, yday=None, hist=hist)
    m0 = pl["matches"][0]
    assert "nid" in m0 and "mv" in m0 and len(m0["sm"]) == 6 and all(len(r) == 6 for r in m0["sm"])
    assert 0 < sum(sum(r) for r in m0["sm"]) <= 1005     # binde bir birimlerle, 6x6 kısmı


def test_euro_cup_detection():
    for ok in ("UEFA Şampiyonlar Ligi", "UEFA Avrupa Ligi", "UEFA Konferans Ligi", "Şampiyonlar Ligi", "UEFA Konferans Ligi Play-Off"):
        assert bot._is_euro_cup(ok), ok
    for no in ("UEFA Gençler Ligi", "AFC Şampiyonlar Ligi", "AFC Kupası", "CAF Şampiyonlar Ligi", "UEFA Kadınlar Şampiyonlar Ligi",
               "Şampiyonlar Ligi U19", "Konkakaf Şampiyonlar Kupası", "Türkiye Süper Lig", "Mısır Ligi Kupası", "", None, "CONCACAF Şampiyonlar Ligi"):
        assert not bot._is_euro_cup(no), no


def test_euro_flag_ignores_missing_values():
    import math
    assert bot._euro({"EURO": float("nan")}) is False      # NaN, bool() ile True olurdu
    assert bot._euro({}) is False and bot._euro({"EURO": False}) is False
    assert bot._euro({"EURO": True}) is True
    assert bot._euro(pd.Series({"Div": "E0"})) is False
    assert bot._is_euro_cup("UEFA ŞAMPİYONLAR LİGİ") and bot._is_euro_cup("UEFA KONFERANS LİGİ")
    assert not bot._is_euro_cup("UEFA GENÇLER LİGİ") and not bot._is_euro_cup("AFC ŞAMPİYONLAR LİGİ")


def test_milyoner_coupon_rules_and_freeze(tmp_path, monkeypatch):
    hist, rows = synth(days=3)
    _env(hist, rows)
    monkeypatch.setattr(bot, "COUPON_LOG", tmp_path / "k.json")
    k = bot.build_milyoner(rows)
    assert k and k["mil"] and 1 <= len(k["legs"]) <= 3 and k["odds"] >= bot.MIL_ODDS - 1e-9
    for l in k["legs"]:
        assert l["p"] >= bot.COUPON_MIN_P and l["o"] >= bot.COUPON_MIN_LEG_ODDS
        assert (l.get("mbs") or 1) <= len(k["legs"])             # MBS koşulu kupondaki maç sayısını aşmaz
    # aynı oran eşiğini sağlayan 3'lü kupondan (sürpriz mantığı) daha az olasılıklı olmamalı: en yüksek olasılığı seçer
    sur = bot.build_coupons(rows).get("surprise") or []
    if sur and sur[0]["odds"] >= bot.MIL_ODDS:
        assert k["p"] >= sur[0]["p"] - 1e-9
    day = pd.Timestamp(k["date"]).date()
    monkeypatch.setattr(bot, "today_tr", lambda: day)
    bot.save_mil(k)
    ent = bot._load_log()[k["date"]]
    assert ent["mil_saved_on"] == k["date"] and sum(1 for x in ent["list"] if x.get("mil")) == 1
    # sonradan maçlar bültenden düşse de aynı kupon geri gelir ve kayıt değişmez
    used = {(l["home"], l["away"]) for l in k["legs"]}
    rows2 = [x for x in rows if (x[0]["HomeTeam"], x[0]["AwayTeam"]) not in used]
    k2 = bot.build_milyoner(rows2)
    assert k2["frozen"] and [(l["home"], l["sel"], l["o"]) for l in k2["legs"]] == [(l["home"], l["sel"], l["o"]) for l in k["legs"]]
    k2["odds"] = 99
    bot.save_mil(k2)
    assert bot._load_log()[k["date"]]["list"][-1]["odds"] == k["odds"]


def test_mil_summary_attempts():
    def g(d, st, odds=2.0):
        return {d: [{"mil": True, "odds": odds, "status": st, "p": .5}]}
    graded = {}
    seq = ["won", "won", "lost", "won", "pending"]
    for i, st in enumerate(seq):
        graded.update(g(f"2026-10-{10 + i}", st, 2.05))
    graded["2026-10-12"].append({"mil": False, "odds": 3, "status": "won"})      # normal kupon sayılmaz
    s = bot.mil_summary(graded)
    assert [a["result"] for a in s["attempts"]] == ["lost", "running"]
    a0, a1 = s["attempts"]
    assert a0["days"] == 2 and abs(a0["bank"] - 100 * 2.05 ** 2) < 0.02
    assert a1["days"] == 1 and a1["pending"] == 1 and s["best"] == 2 and not s["reached"]
    assert (s["won"], s["lost"], s["pending"]) == (3, 1, 1)
    full = {f"2026-09-{i + 1:02d}": [{"mil": True, "odds": 2.0, "status": "won", "p": .5}] for i in range(bot.MIL_DAYS)}
    s2 = bot.mil_summary(full)
    assert s2["reached"] and s2["attempts"][0]["result"] == "goal" and s2["attempts"][0]["bank"] == 100 * 2 ** bot.MIL_DAYS


def test_mil_summary_keeps_legs():
    leg = {"home": "A", "away": "B", "sel": "MS 1", "o": 1.5, "ok": True, "score": "2-0", "time": "20:00"}
    g = {"2026-09-01": [{"mil": True, "odds": 2.1, "status": "won", "p": .6, "legs": [leg]}],
         "2026-09-02": [{"mil": True, "odds": 2.0, "status": "lost", "p": .6, "legs": [dict(leg, ok=False, score="0-1")]}]}
    a = bot.mil_summary(g)["attempts"][0]
    assert a["result"] == "lost" and a["path"][0]["legs"][0]["score"] == "2-0" and a["path"][1]["legs"][0]["ok"] is False


def test_ledger_update_and_report(tmp_path):
    def nm(ps, odds):
        return {"mtid": "1", "line": 0, "ps": ps, "outs": [{"n": i + 1, "o": o} for i, o in enumerate(odds)]}
    ms = [{"nid": 5, "date": "2026-12-01", "home": "A", "away": "B", "src": "model",
           "nm": [nm([.5, .3, .2], [2.0, 3.3, 4.0]),
                  {"mtid": "12", "line": 2.5, "ps": [.45, .55], "outs": [{"n": 1, "o": 1.95}, {"n": 2, "o": 1.85}]}]},
          {"nid": 6, "date": "2026-11-20", "home": "C", "away": "D", "src": "model", "nm": [nm([.4, .3, .3], [2.4, 3.2, 3.0])]},  # geçmiş gün: yazılmaz
          {"nid": 7, "date": "2026-12-01", "home": "E", "away": "F", "src": "model", "nm": [nm(None, [2.0, 3.0, 3.5])]}]        # modelsiz: yazılmaz
    led, n = bot.ledger_update(ms, today="2026-12-01", path=tmp_path / "l.json")
    assert set(led) == {"5"} and n == 1 and set(led["5"]) >= {"ms", "ou", "h", "a"}
    led, n2 = bot.ledger_update(ms, today="2026-12-01", path=tmp_path / "l.json")
    assert n2 == 0 and len(led) == 1
    rep = bot.ledger_report(led, {5: (2, 1, None, None)})
    assert rep["ms"]["n"] == 1 and rep["ou"]["n"] == 1
    assert abs(rep["ms"]["ll_model"] - (-math.log(.5))) < 1e-3          # ev galibiyeti, model .5 demişti
    assert rep["ou"]["ll_model"] == round(-math.log(.55), 4)           # 3 gol: Üst
    assert bot.ledger_report(led, {}) == {}


# ---- model ayarı: şut vekili, kısma, walk-forward seçimi ----
def _syn_league(rounds=110, seed=1, shots=True):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    att, dfn = rng.normal(0, .25, 20), rng.normal(0, .25, 20)
    rows = []
    for rd in range(rounds):
        perm = rng.permutation(20)
        for a in range(0, 20, 2):
            h, w = perm[a], perm[a + 1]
            lh, la = np.exp(.45 + att[h] - dfn[w]), np.exp(.25 + att[w] - dfn[h])
            r = dict(Div="E0", Date=pd.Timestamp("2025-08-01") + pd.Timedelta(days=rd * 7), HomeTeam=f"T{h}",
                     AwayTeam=f"T{w}", FTHG=rng.poisson(lh), FTAG=rng.poisson(la))
            if shots:
                hst, ast = rng.poisson(lh * 3.2), rng.poisson(la * 3.2)
                r.update(HST=hst, AST=ast, HS=hst + rng.poisson(5), AS=ast + rng.poisson(5))
            rows.append(r)
    return pd.DataFrame(rows)


def test_shot_proxy_and_shrink():
    df = _syn_league(40)
    px = bot.Model._xg_proxy(df)
    assert px is not None and abs(np.nanmean(px[0]) - df["FTHG"].mean()) < 0.15      # vekil ortalamada gollerle uyumlu
    assert bot.Model._xg_proxy(df.drop(columns=["HS", "AS", "HST", "AST"])) is None   # şut yoksa kullanılmaz
    m0 = bot.Model(shot_w=0.0).fit(df)
    m1 = bot.Model(shot_w=0.5).fit(df)
    assert m1.fit(df.drop(columns=["HS", "AS", "HST", "AST"])).knows("T1", "T2")       # şutsuz veriyle de çalışır
    h, a = "T1", "T2"
    lh_full, _ = m0.lambdas(h, a)
    m0.shrink = 0.8
    lh_shr, _ = m0.lambdas(h, a)
    assert abs(math.log(lh_shr) - m0.mu - m0.home) <= abs(math.log(lh_full) - m0.mu - m0.home) + 1e-12   # kısma ortalamaya çeker


def test_tune_writes_config(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "DATA_DIR", tmp_path)
    _syn_league(110).to_csv(tmp_path / "E0_2526.csv", index=False, date_format="%d/%m/%Y")
    out = tmp_path / "model_ayar.json"
    res = bot.cmd_tune(["E0"], out=out, n_days=100)
    assert out.exists() and res["n"] > 50
    assert res["best_loss"] <= res["default_loss"] + 1e-9           # seçilen ayar varsayılandan kötü olamaz
    assert res["used"] == (res["default_loss"] - res["best_loss"] >= bot.TUNE_MIN_GAIN)
    cfg = bot.load_model_cfg(out)
    assert (not res["used"]) or cfg.get("shot_w") is not None
    bot.load_model_cfg(tmp_path / "yok.json")                         # dosya yoksa varsayılanlar
    assert bot._CFG == {}
