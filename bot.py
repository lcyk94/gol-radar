#!/usr/bin/env python3
"""
Futbol tahmin botu
==================
Zaman ağırlıklı Poisson modeli (Dixon-Coles düzeltmeli) ile maç sonucu,
Üst/Alt 2.5, KG Var/Yok ve skor tahmini üretir.

Veri: football-data.co.uk (ücretsiz, API anahtarı gerekmez)

Kurulum:   pip install numpy pandas scipy
Kullanım:
    python bot.py update                       # verileri indir / güncelle
    python bot.py predict                      # önümüzdeki 7 günün maçları
    python bot.py predict --days 3 --league T1 E0
    python bot.py match "Galatasaray" "Fenerbahce" --league T1
    python bot.py backtest --league T1         # modeli geçmişte sına
    python bot.py site --open                  # web sayfası üret ve tarayıcıda aç
    python bot.py demo --open                  # kurgusal takımlarla örnek sayfa
    python bot.py leagues                      # lig kodları
    python bot.py selftest                     # internetsiz kendi kendine test
"""
import argparse
import heapq
import io
import itertools
import json
import math
import sys
import urllib.request
import webbrowser
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

BASE = "https://www.football-data.co.uk"
DATA_DIR = Path(__file__).resolve().parent / "data"

LEAGUES = {
    "E0": "İngiltere Premier Lig", "E1": "İngiltere Championship",
    "E2": "İngiltere League One", "E3": "İngiltere League Two",
    "EC": "İngiltere Conference",
    "SC0": "İskoçya Premiership", "SC1": "İskoçya Championship",
    "D1": "Almanya Bundesliga", "D2": "Almanya 2. Bundesliga",
    "I1": "İtalya Serie A", "I2": "İtalya Serie B",
    "SP1": "İspanya La Liga", "SP2": "İspanya Segunda",
    "F1": "Fransa Ligue 1", "F2": "Fransa Ligue 2",
    "N1": "Hollanda Eredivisie", "B1": "Belçika Pro League",
    "P1": "Portekiz Primeira Liga", "T1": "Türkiye Süper Lig",
    "G1": "Yunanistan Super League",
}

ODDS_SETS = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"),
             ("PSH", "PSD", "PSA")]
ODDS_COLS = [c for s in ODDS_SETS for c in s]
KEEP = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

MAX_GOALS = 10
RHO = -0.06          # Dixon-Coles düşük skor düzeltmesi
HALF_LIFE = 200      # gün: eski maçların ağırlığı yarıya iner
REG = 1.0            # takım parametrelerine ridge cezası


# --------------------------------------------------------------------------
# Veri
# --------------------------------------------------------------------------
def season_codes(today, n=3):
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{(start - k) % 100:02d}{(start - k + 1) % 100:02d}"
            for k in range(n - 1, -1, -1)]


def fetch(url, dest, force):
    if dest.exists() and not force:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if data.lstrip()[:5].lower() in (b"<html", b"<!doc"):
            print(f"  ! veri yerine web sayfası geldi: {url}")
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! indirilemedi: {url} ({e})")
        return False


def update(leagues=None):
    DATA_DIR.mkdir(exist_ok=True)
    seasons = season_codes(date.today())
    current = seasons[-1]
    leagues = leagues or list(LEAGUES)
    ok = 0
    for div in leagues:
        for s in seasons:
            dest = DATA_DIR / f"{div}_{s}.csv"
            if fetch(f"{BASE}/mmz4281/{s}/{div}.csv", dest, force=(s == current)):
                ok += 1
    fx_ok = fetch(f"{BASE}/fixtures.csv", DATA_DIR / "fixtures.csv", force=True)
    print(f"{ok} sonuç dosyası hazır, fikstür: {'tamam' if fx_ok else 'İNDİRİLEMEDİ'}. "
          f"Veri klasörü: {DATA_DIR}")
    if ok == 0 or not fx_ok:
        sys.exit("Veri indirilemedi, yukarıdaki '!' satırlarına bak.")


def _read(path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")      # BOM'u temizler
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    if "Div" not in df.columns:
        if path.stem.startswith("fixtures"):
            raise ValueError("fixtures dosyasında 'Div' sütunu yok")
        df["Div"] = path.stem.split("_")[0]
    return df


def _clean(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed",
                                errors="coerce")
    for c in ("FTHG", "FTAG"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    cols = [c for c in KEEP + ODDS_COLS if c in df.columns]
    return df[cols].sort_values("Date").reset_index(drop=True)


def load_results(leagues=None):
    frames = []
    for p in sorted(DATA_DIR.glob("*_*.csv")):
        div = p.stem.split("_")[0]
        if leagues and div not in leagues:
            continue
        try:
            frames.append(_read(p))
        except Exception as e:  # noqa: BLE001
            print(f"  ! okunamadı: {p.name} ({e})")
    if not frames:
        sys.exit(f"Sonuç dosyası bulunamadı (ligler: {leagues or 'hepsi'}, klasör: {DATA_DIR}). "
                 "Önce çalıştır:  python bot.py update")
    return _clean(pd.concat(frames, ignore_index=True))


def load_fixtures():
    p = DATA_DIR / "fixtures.csv"
    if not p.exists():
        sys.exit("fixtures.csv yok. Önce çalıştır:  python bot.py update")
    df = _read(p)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed",
                                errors="coerce")
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])


def raw_odds(row):
    """Fikstürdeki 1/X/2 oranları (varsa)."""
    for h, d, a in ODDS_SETS:
        try:
            o = np.array([row[h], row[d], row[a]], dtype=float)
        except (KeyError, ValueError, TypeError):
            continue
        if np.all(np.isfinite(o)) and np.all(o > 1):
            return o
    return None


def market_probs(row):
    """Bahis oranlarından marjı temizlenmiş olasılıklar (varsa)."""
    o = raw_odds(row)
    if o is None:
        return None
    inv = 1 / o
    return inv / inv.sum()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Model:
    """log(λ_ev) = mu + ev + atak[ev] - savunma[dep]
       log(λ_dep) = mu + atak[dep] - savunma[ev]"""

    def __init__(self, half_life=HALF_LIFE, reg=REG, rho=RHO):
        self.half_life, self.reg, self.rho = half_life, reg, rho

    def fit(self, df, ref_date=None):
        ref_date = pd.Timestamp(ref_date) if ref_date is not None else df["Date"].max()
        self.teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi = df["HomeTeam"].map(idx).to_numpy()
        ai = df["AwayTeam"].map(idx).to_numpy()
        hg = df["FTHG"].to_numpy(float)
        ag = df["FTAG"].to_numpy(float)
        age = (ref_date - df["Date"]).dt.days.clip(lower=0).to_numpy(float)
        w = 0.5 ** (age / self.half_life)
        reg = self.reg

        def f(x):
            mu, home, att, dfn = x[0], x[1], x[2:2 + n], x[2 + n:]
            eh = mu + home + att[hi] - dfn[ai]
            ea = mu + att[ai] - dfn[hi]
            lh, la = np.exp(eh), np.exp(ea)
            ll = np.sum(w * (hg * eh - lh + ag * ea - la))
            ll -= reg * (att @ att + dfn @ dfn)
            rh, ra = w * (hg - lh), w * (ag - la)
            g = np.empty_like(x)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            g[2:2 + n] = (np.bincount(hi, rh, n) + np.bincount(ai, ra, n)
                          - 2 * reg * att)
            g[2 + n:] = (-np.bincount(ai, rh, n) - np.bincount(hi, ra, n)
                         - 2 * reg * dfn)
            return -ll, -g

        x0 = np.zeros(2 + 2 * n)
        x0[0] = np.log(max(np.average(np.r_[hg, ag]), 0.5))
        res = minimize(f, x0, jac=True, method="L-BFGS-B")
        self.mu, self.home = res.x[0], res.x[1]
        self.att = dict(zip(self.teams, res.x[2:2 + n]))
        self.dfn = dict(zip(self.teams, res.x[2 + n:]))
        return self

    def knows(self, *teams):
        return all(t in self.att for t in teams)

    def lambdas(self, home, away):
        lh = np.exp(self.mu + self.home + self.att[home] - self.dfn[away])
        la = np.exp(self.mu + self.att[away] - self.dfn[home])
        return lh, la

    def predict(self, home, away):
        lh, la = self.lambdas(home, away)
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        r = self.rho
        m[0, 0] *= 1 - lh * la * r
        m[0, 1] *= 1 + lh * r
        m[1, 0] *= 1 + la * r
        m[1, 1] *= 1 - r
        m /= m.sum()
        i, j = np.indices(m.shape)
        top = np.dstack(np.unravel_index(np.argsort(-m.ravel())[:3], m.shape))[0]
        return {
            "p1": float(np.tril(m, -1).sum()),
            "px": float(np.trace(m)),
            "p2": float(np.triu(m, 1).sum()),
            "over25": float(m[(i + j) >= 3].sum()),
            "btts": float(m[1:, 1:].sum()),
            "xg_h": float(lh), "xg_a": float(la),
            "scores": [(int(a), int(b), float(m[a, b])) for a, b in top],
            "matrix": m,
        }


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def backtest(df, n_test=150, step_days=7, verbose=True):
    df = df.sort_values("Date").reset_index(drop=True)
    n_test = min(n_test, len(df) // 3)
    start = df["Date"].iloc[-n_test]
    end = df["Date"].max()
    rows = []
    d = start
    while d <= end:
        train = df[df["Date"] < d]
        test = df[(df["Date"] >= d) & (df["Date"] < d + timedelta(days=step_days))]
        d += timedelta(days=step_days)
        if len(test) == 0 or len(train) < 100:
            continue
        m = Model().fit(train, ref_date=test["Date"].min())
        base = np.array([(train["FTHG"] > train["FTAG"]).mean(),
                         (train["FTHG"] == train["FTAG"]).mean(),
                         (train["FTHG"] < train["FTAG"]).mean()])
        for _, r in test.iterrows():
            if not m.knows(r["HomeTeam"], r["AwayTeam"]):
                continue
            p = m.predict(r["HomeTeam"], r["AwayTeam"])
            y = 0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2)
            rows.append((np.array([p["p1"], p["px"], p["p2"]]), base,
                         market_probs(r), y,
                         (r["FTHG"] + r["FTAG"]) >= 3, p["over25"]))
    if not rows:
        print("Backtest için yeterli veri yok.")
        return None

    def metrics(probs, ys):
        probs = np.array(probs)
        ys = np.array(ys)
        acc = (probs.argmax(1) == ys).mean()
        ll = -np.mean(np.log(np.clip(probs[np.arange(len(ys)), ys], 1e-9, 1)))
        oh = np.eye(3)[ys]
        br = np.mean(np.sum((probs - oh) ** 2, axis=1))
        return acc, ll, br

    ys = [r[3] for r in rows]
    out = {"Model": metrics([r[0] for r in rows], ys),
           "Sabit oran (taban)": metrics([r[1] for r in rows], ys)}
    mk = [(r[2], r[3]) for r in rows if r[2] is not None]
    if mk:
        out["Piyasa (oranlar)"] = metrics([a for a, _ in mk], [b for _, b in mk])
    if verbose:
        print(f"\nBacktest: {len(rows)} maç (her hafta, sadece o güne kadarki veriyle)")
        print(f"{'':22}{'İsabet':>9}{'LogLoss':>10}{'Brier':>9}")
        for k, (a, l, b) in out.items():
            print(f"{k:22}{a:9.1%}{l:10.4f}{b:9.4f}")
        ov_acc = np.mean([(r[5] > 0.5) == r[4] for r in rows])
        print(f"\nÜst/Alt 2.5 isabeti: {ov_acc:.1%}")
        print("LogLoss/Brier düşük = iyi. Piyasanın altına inmek çok zordur;")
        print("model tabanın belirgin altındaysa gerçekten bilgi taşıyor demektir.")
    return out


# --------------------------------------------------------------------------
# Çıktı
# --------------------------------------------------------------------------
def pick(p):
    probs = [p["p1"], p["px"], p["p2"]]
    k = int(np.argmax(probs))
    return ["MS1", "MSX", "MS2"][k], probs[k], k


def print_prediction(home, away, p, mkt=None):
    lab, pr, k = pick(p)
    s = p["scores"][0]
    line = (f"  {home} - {away}\n"
            f"    1/X/2: %{p['p1']*100:4.1f} / %{p['px']*100:4.1f} / %{p['p2']*100:4.1f}"
            f"   → {lab} (%{pr*100:.0f})\n"
            f"    Üst2.5: %{p['over25']*100:.0f}   KG Var: %{p['btts']*100:.0f}"
            f"   xG: {p['xg_h']:.2f}-{p['xg_a']:.2f}"
            f"   En olası skor: {s[0]}-{s[1]}")
    if mkt is not None:
        edge = p[["p1", "px", "p2"][k]] - mkt[k]
        line += (f"\n    Piyasa 1/X/2: %{mkt[0]*100:.0f} / %{mkt[1]*100:.0f} / %{mkt[2]*100:.0f}"
                 f"   (modelin seçimi piyasadan {edge*100:+.1f} puan ayrışıyor)")
    print(line)


def collect(leagues, days):
    """Önümüzdeki günlerin maçları için (satır, tahmin, piyasa) listesi."""
    fx = load_fixtures()
    today = pd.Timestamp(date.today())
    fx = fx[(fx["Date"] >= today) & (fx["Date"] <= today + timedelta(days=days))]
    if leagues:
        fx = fx[fx["Div"].isin(leagues)]
    if fx.empty:
        return [], 0
    hist = load_results(list(fx["Div"].unique()))
    models, out_rows, skipped = {}, [], 0
    for div in fx["Div"].unique():
        h = hist[hist["Div"] == div]
        if len(h) >= 50:                      # çok az veri olan ligi atla
            models[div] = Model().fit(h, ref_date=today)
    for _, r in fx.sort_values(["Date", "Div"]).iterrows():
        m = models.get(r["Div"])
        if m is None or not m.knows(r["HomeTeam"], r["AwayTeam"]):
            skipped += 1
            continue
        p = m.predict(r["HomeTeam"], r["AwayTeam"])
        out_rows.append((r, p, market_probs(r)))
    return out_rows, skipped


def cmd_predict(leagues, days):
    out_rows, skipped = collect(leagues, days)
    if not out_rows:
        print("Bu aralıkta maç bulunamadı (fixtures.csv sadece ana liglerin yakın fikstürünü içerir).")
        return
    last = None
    for r, p, mk in out_rows:
        key = (r["Date"].date(), r["Div"])
        if key != last:
            print(f"\n=== {r['Date']:%d.%m.%Y} — {LEAGUES.get(r['Div'], r['Div'])} ===")
            last = key
        print_prediction(r["HomeTeam"], r["AwayTeam"], p, mk)
    if skipped:
        print(f"\n({skipped} maç atlandı: takımın yeterli geçmiş verisi yok)")
    csv = pd.DataFrame([{
        "Tarih": r["Date"].date(), "Lig": r["Div"], "Ev": r["HomeTeam"],
        "Deplasman": r["AwayTeam"], "P1": round(p["p1"], 3), "PX": round(p["px"], 3),
        "P2": round(p["p2"], 3), "Ust25": round(p["over25"], 3),
        "KG": round(p["btts"], 3), "Skor": f"{p['scores'][0][0]}-{p['scores'][0][1]}"}
        for r, p, _ in out_rows])
    csv.to_csv("predictions.csv", index=False)
    print("\nTüm tahminler predictions.csv dosyasına yazıldı.")


# --------------------------------------------------------------------------
# Marketler ve renk sınıflaması (web sayfası için)
# --------------------------------------------------------------------------
GREEN_P = 0.60        # bu olasılık ve üstü: önerilir (yeşil)
ORANGE_P = 0.35       # bunun altı: önerilmez (kırmızı), arası riskli (turuncu)


def tier(p, odds=None):
    """g = önerilir (yeşil), o = riskli (turuncu), r = önerilmez (kırmızı).
    Renk sadece model olasılığına bağlı. Oran düşükse renk değişmez, uyarı notu eklenir."""
    t = "g" if p >= GREEN_P else ("o" if p >= ORANGE_P else "r")
    note = "oran düşük, değer yok" if (odds and t == "g" and p * odds < 0.97) else ""
    return t, note


def _num(row, *keys):
    if row is None:
        return None
    for k in keys:
        try:
            v = float(row[k])
        except (KeyError, ValueError, TypeError):
            continue
        if np.isfinite(v) and v > 1:
            return v
    return None


def build_markets(p, row=None):
    m = p["matrix"]
    i, j = np.indices(m.shape)
    g, d = i + j, i - j

    def S(label, prob, odds=None):
        t, note = tier(float(prob), odds)
        return {"l": label, "p": round(float(prob), 4),
                "o": round(float(odds), 2) if odds else None, "t": t, "n": note}

    ro = raw_odds(row) if row is not None else None
    o1, ox, o2 = ro if ro is not None else (None, None, None)
    over25 = _num(row, "Avg>2.5", "B365>2.5")
    under25 = _num(row, "Avg<2.5", "B365<2.5")
    p1, px, p2 = p["p1"], p["px"], p["p2"]

    M = [("Maç Sonucu", [S("MS 1", p1, o1), S("MS X", px, ox), S("MS 2", p2, o2)]),
         ("Çifte Şans", [S("1-X", p1 + px), S("1-2", p1 + p2), S("X-2", px + p2)])]
    for line in (1.5, 2.5, 3.5):
        over = m[g > line].sum()
        oo, uo = (over25, under25) if line == 2.5 else (None, None)
        M.append((f"Alt/Üst {line}", [S("Alt", 1 - over, uo), S("Üst", over, oo)]))
    yes = m[1:, 1:].sum()
    M.append(("Karşılıklı Gol", [S("Var", yes), S("Yok", 1 - yes)]))
    M.append(("Toplam Gol", [S("0-1", m[g <= 1].sum()), S("2-3", m[(g >= 2) & (g <= 3)].sum()),
                             S("4-5", m[(g >= 4) & (g <= 5)].sum()), S("6+", m[g >= 6].sum())]))
    M.append(("Ev Sahibi Golleri", [S("0.5 Alt", m[0, :].sum()), S("0.5 Üst", 1 - m[0, :].sum()),
                                    S("1.5 Alt", m[:2, :].sum()), S("1.5 Üst", m[2:, :].sum())]))
    M.append(("Deplasman Golleri", [S("0.5 Alt", m[:, 0].sum()), S("0.5 Üst", 1 - m[:, 0].sum()),
                                    S("1.5 Alt", m[:, :2].sum()), S("1.5 Üst", m[:, 2:].sum())]))
    M.append(("Handikaplı MS (Ev -1)", [S("1", m[d >= 2].sum()), S("X", m[d == 1].sum()),
                                        S("2", m[d <= 0].sum())]))
    M.append(("Handikaplı MS (Dep -1)", [S("1", m[d >= 0].sum()), S("X", m[d == -1].sum()),
                                         S("2", m[d <= -2].sum())]))
    combo = []
    for lab, mask in (("1", d > 0), ("X", d == 0), ("2", d < 0)):
        combo += [S(f"{lab} & Alt", m[mask & (g <= 2)].sum()),
                  S(f"{lab} & Üst", m[mask & (g >= 3)].sum())]
    M.append(("MS ve Alt/Üst 2.5", combo))
    top = np.argsort(-m.ravel())[:8]
    M.append(("Doğru Skor", [S(f"{a}-{b}", m[a, b])
                             for a, b in (np.unravel_index(k, m.shape) for k in top)]))
    return [{"name": n, "sels": s} for n, s in M]


DEMO_LEAGUES = {"DA": "Demo Lig A", "DB": "Demo Lig B"}
DEMO_TEAMS = {
    "DA": ["Kuzey FK", "Marmara SK", "Ege Spor", "Anadolu United",
           "Karadeniz AS", "Akdeniz Spor", "Trakya FK", "Toros SK"],
    "DB": ["Yıldız Gücü", "Liman Spor", "Bozkır FK", "Sahil Atletik",
           "Vadi SK", "Zirve Spor", "Kıyı FK", "Meşe Spor"],
}


def demo_rows(days=3):
    """Kurgusal takımlarla örnek veri (gerçek maç değildir)."""
    rng = np.random.default_rng(11)
    today = pd.Timestamp(date.today())
    rows = []
    for div, names in DEMO_TEAMS.items():
        n = len(names)
        att, dfn = rng.normal(0, .3, n), rng.normal(0, .3, n)
        hist = []
        for _ in range(2):
            for a in range(n):
                for b in range(n):
                    if a != b:
                        day = today - timedelta(days=int(rng.integers(5, 330)))
                        hist.append((day, names[a], names[b],
                                     rng.poisson(np.exp(.55 + att[a] - dfn[b])),
                                     rng.poisson(np.exp(.25 + att[b] - dfn[a]))))
        hdf = pd.DataFrame(hist, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        model = Model().fit(hdf, ref_date=today)
        for dd in range(days):
            order = rng.permutation(n)
            for q in range(0, n, 2):
                h, a = names[order[q]], names[order[q + 1]]
                p = model.predict(h, a)
                marg = 1.06
                pr = np.array([p["p1"], p["px"], p["p2"]]) * rng.normal(1, .05, 3)
                pr /= pr.sum()
                o = np.round(1 / (pr * marg), 2)
                po = np.clip(p["over25"] * rng.normal(1, .05), .05, .95)
                row = pd.Series({
                    "Div": div, "Date": today + timedelta(days=dd),
                    "Time": f"{int(rng.choice([13, 16, 19, 20, 21]))}:{rng.choice(['00', '30'])}",
                    "HomeTeam": h, "AwayTeam": a,
                    "B365H": o[0], "B365D": o[1], "B365A": o[2],
                    "B365>2.5": round(1 / (po * marg), 2),
                    "B365<2.5": round(1 / ((1 - po) * marg), 2)})
                rows.append((row, p, None))
    return rows


COUPON_LEGS = 4        # kupondaki maç sayısı
COUPON_MIN_ODDS = 5.0  # toplam oran en az bu kadar olmalı
COUPON_MIN_P = 0.60    # kupona girecek her seçeneğin model olasılığı en az bu kadar
N_COUPONS = 4


def _legs(r, p):
    """Gerçek bahis oranı olan seçenekler: MS 1/X/2 ve Alt/Üst 2.5.
    Diğer marketlerde gerçek oran yok, o yüzden kupona girmezler."""
    legs = []
    ro = raw_odds(r)
    if ro is not None:
        legs += [("MS 1", p["p1"], float(ro[0])), ("MS X", p["px"], float(ro[1])),
                 ("MS 2", p["p2"], float(ro[2]))]
    ov, un = _num(r, "Avg>2.5", "B365>2.5"), _num(r, "Avg<2.5", "B365<2.5")
    if ov:
        legs.append(("Üst 2.5", p["over25"], ov))
    if un:
        legs.append(("Alt 2.5", 1 - p["over25"], un))
    # sadece listede de yeşil görünen seçenekler girer (turuncu/kırmızı girmez)
    legs = [l for l in legs if l[1] >= COUPON_MIN_P and tier(l[1], l[2])[0] == "g"]
    return sorted(legs, key=lambda x: -x[1])[:3]


def build_coupons(out_rows):
    """Günün maçlarından, toplam oranı >= 5 olan ve modele göre en çok tutması
    beklenen 4 farklı 4'lü kupon. Maçlar birbirinden bağımsız varsayılır."""
    today = pd.Timestamp(date.today())
    by_day = {}
    for k, (r, p, _) in enumerate(out_rows):
        legs = _legs(r, p)
        if legs:
            by_day.setdefault(r["Date"].normalize(), []).append((k, r, legs))
    for day in sorted(d for d in by_day if d >= today):
        if len(by_day[day]) < COUPON_LEGS:
            continue
        res = _coupons_for_day(day, by_day[day])
        if res["list"]:               # kupon çıkan ilk günü göster
            return res
    return None


def _coupons_for_day(day, items):
    cands = sorted(items, key=lambda t: -t[2][0][1])[:24]

    def gen():
        for combo in itertools.combinations(cands, COUPON_LEGS):
            for pick in itertools.product(*[c[2] for c in combo]):
                od = math.prod(x[2] for x in pick)
                if od >= COUPON_MIN_ODDS:
                    yield (math.prod(x[1] for x in pick), od, combo, pick)

    best = heapq.nlargest(4000, gen(), key=lambda t: t[0])
    chosen, ids_list = [], []
    for max_shared in (2, 3):          # kuponlar birbirinden farklı olsun
        for pr, od, combo, pick in best:
            if len(chosen) >= N_COUPONS:
                break
            ids = {(c[0], x[0]) for c, x in zip(combo, pick)}
            if ids in ids_list or any(len(ids & q) > max_shared for q in ids_list):
                continue
            ids_list.append(ids)
            legs = []
            for c, x in zip(combo, pick):
                r = c[1]
                t = str(r.get("Time", "") or "")
                legs.append({"id": c[0], "time": "" if t in ("nan", "NaT") else t[:5],
                             "home": r["HomeTeam"], "away": r["AwayTeam"],
                             "league": LEAGUES.get(r["Div"], DEMO_LEAGUES.get(r["Div"], r["Div"])),
                             "sel": x[0], "o": round(x[2], 2), "p": round(x[1], 3),
                             "t": tier(x[1], x[2])[0]})
            legs.sort(key=lambda l: l["time"])
            chosen.append({"legs": legs, "odds": round(od, 2),
                           "p": round(pr, 4), "ev": round(pr * od, 2)})
    return {"date": day.strftime("%Y-%m-%d"), "list": chosen}


def site_payload(out_rows, demo=False):
    matches = []
    for k, (r, p, _) in enumerate(out_rows):
        t = str(r.get("Time", "") or "")
        matches.append({
            "id": k, "div": r["Div"],
            "league": LEAGUES.get(r["Div"], DEMO_LEAGUES.get(r["Div"], r["Div"])),
            "date": r["Date"].strftime("%Y-%m-%d"),
            "time": "" if t in ("nan", "NaT") else t[:5],
            "home": r["HomeTeam"], "away": r["AwayTeam"],
            "xg": [round(p["xg_h"], 2), round(p["xg_a"], 2)],
            "markets": build_markets(p, r)})
    return {"generated": datetime.now().strftime("%d.%m.%Y %H:%M"), "demo": demo,
            "green": GREEN_P, "orange": ORANGE_P, "matches": matches,
            "coupons": build_coupons(out_rows)}


def write_site(payload, out="tahminler.html"):
    tpl = (Path(__file__).resolve().parent / "template.html").read_text(encoding="utf-8")
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    path = Path(out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tpl.replace("/*__DATA__*/null", blob), encoding="utf-8")
    return path


def cmd_site(leagues, days, out, open_it, demo=False):
    if demo:
        rows, skipped = demo_rows(), 0
    else:
        rows, skipped = collect(leagues, days)
    if not rows:
        sys.exit("Bu aralıkta maç bulunamadı. Önce:  python bot.py update")
    path = write_site(site_payload(rows, demo), out)
    print(f"{len(rows)} maç yazıldı → {path}")
    if skipped:
        print(f"({skipped} maç atlandı: takımın yeterli geçmiş verisi yok)")
    if open_it:
        webbrowser.open(path.as_uri())


def resolve(name, teams):
    low = name.lower()
    exact = [t for t in teams if t.lower() == low]
    if exact:
        return exact[0]
    part = [t for t in teams if low in t.lower()]
    if len(part) == 1:
        return part[0]
    sys.exit(f"'{name}' için takım bulunamadı/belirsiz. Seçenekler: {', '.join(part or teams)}")


def cmd_match(home, away, div):
    hist = load_results([div])
    m = Model().fit(hist)
    h, a = resolve(home, m.teams), resolve(away, m.teams)
    print(f"\n{LEAGUES.get(div, div)}")
    print_prediction(h, a, m.predict(h, a))


# --------------------------------------------------------------------------
# Kendi kendine test (internet gerekmez)
# --------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(7)
    n = 20
    att = rng.normal(0, 0.25, n)
    dfn = rng.normal(0, 0.25, n)
    att -= att.mean()
    dfn -= dfn.mean()
    mu, home = 0.25, 0.30
    rows = []
    d0 = pd.Timestamp("2025-08-01")
    for season in range(2):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                day = d0 + timedelta(days=int(rng.integers(0, 280)) + 365 * season)
                rows.append((day, f"T{i:02d}", f"T{j:02d}",
                             rng.poisson(np.exp(mu + home + att[i] - dfn[j])),
                             rng.poisson(np.exp(mu + att[j] - dfn[i]))))
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Div"] = "XX"
    df = df.sort_values("Date").reset_index(drop=True)

    m = Model(half_life=10_000).fit(df)
    est_att = np.array([m.att[f"T{i:02d}"] for i in range(n)])
    est_def = np.array([m.dfn[f"T{i:02d}"] for i in range(n)])
    c_att = np.corrcoef(att, est_att)[0, 1]
    c_def = np.corrcoef(dfn, est_def)[0, 1]
    print(f"Atak korelasyonu: {c_att:.3f}   Savunma korelasyonu: {c_def:.3f}")
    print(f"Ev avantajı: gerçek {home:.2f}, tahmin {m.home:.2f}")
    assert c_att > 0.85 and c_def > 0.85, "parametreler geri kazanılamadı"
    assert abs(m.home - home) < 0.1, "ev avantajı hatalı"

    p = m.predict("T00", "T01")
    tot = p["p1"] + p["px"] + p["p2"]
    print(f"1+X+2 toplamı: {tot:.6f}   Üst2.5: {p['over25']:.2f}   KG: {p['btts']:.2f}")
    assert abs(tot - 1) < 1e-9
    assert 0 <= p["over25"] <= 1 and 0 <= p["btts"] <= 1

    print("\nÖrnek tahmin:")
    print_prediction("T00", "T01", p)
    res = backtest(df, n_test=300)
    assert res and res["Model"][1] < res["Sabit oran (taban)"][1], \
        "model tabandan iyi olmalı"
    print("\nSELFTEST GEÇTİ")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Futbol tahmin botu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update").add_argument("--league", nargs="*")
    p = sub.add_parser("predict")
    p.add_argument("--league", nargs="*")
    p.add_argument("--days", type=int, default=7)
    m = sub.add_parser("match")
    m.add_argument("home")
    m.add_argument("away")
    m.add_argument("--league", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--league", required=True)
    b.add_argument("--matches", type=int, default=150)
    sub.add_parser("leagues")
    sub.add_parser("selftest")
    for name in ("site", "demo"):
        s = sub.add_parser(name)
        s.add_argument("--out", default="tahminler.html")
        s.add_argument("--open", action="store_true", help="tarayıcıda aç")
        if name == "site":
            s.add_argument("--league", nargs="*")
            s.add_argument("--days", type=int, default=7)
    a = ap.parse_args()

    if a.cmd == "update":
        update(a.league)
    elif a.cmd == "predict":
        cmd_predict(a.league, a.days)
    elif a.cmd == "match":
        cmd_match(a.home, a.away, a.league)
    elif a.cmd == "backtest":
        backtest(load_results([a.league]), n_test=a.matches)
    elif a.cmd == "leagues":
        for k, v in LEAGUES.items():
            print(f"{k:5}{v}")
    elif a.cmd == "selftest":
        selftest()
    elif a.cmd == "site":
        cmd_site(a.league, a.days, a.out, a.open)
    elif a.cmd == "demo":
        cmd_site(None, 3, a.out, a.open, demo=True)


if __name__ == "__main__":
    main()
