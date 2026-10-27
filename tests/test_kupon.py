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
            if mk["mtid"] == 12:
                mk["outs"] = list(reversed([{**o} for o in mk["outs"]]))
                for i, o in enumerate(mk["outs"]):
                    o["n"] = i + 1
    v = _env(hist, rows)
    assert not v[12]["ok"] and v[1]["ok"]


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
