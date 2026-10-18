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


def _now_tr():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Istanbul"))
    except Exception:  # noqa: BLE001
        return datetime.utcnow() + timedelta(hours=3)


def today_tr():
    """Bugünün tarihi, Türkiye saatiyle (Actions sunucusu UTC'dedir, gece yarısı civarında gün kayar)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Istanbul")).date()
    except Exception:  # noqa: BLE001
        return (datetime.utcnow() + timedelta(hours=3)).date()

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
ODDS_COLS = [c for s in ODDS_SETS for c in s] + ["Avg>2.5", "Avg<2.5", "B365>2.5", "B365<2.5"]
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
    seasons = season_codes(today_tr())
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
    df = df[cols].drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"])
    return df.sort_values("Date").reset_index(drop=True)


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
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    return df.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"])


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
    today = pd.Timestamp(today_tr())
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


KEYWORDS_BIG = ("SÜPER LİG", "PREMIER", "LA LIGA", "SERIE A", "BUNDESLİGA", "LIGUE 1", "EREDİVİSİE",
                "PRIMEIRA", "CHAMPIONSHIP", "1. LİG", "LALIGA", "PRO LEAGUE", "PREMIERSHIP")


def _nes_of(r):
    """Satırın Nesine verisi: (Nesine maçı ya da None, Nesine modu açık mı)."""
    try:
        return r.get("NES"), bool(r.get("NESACT", False))
    except AttributeError:
        return None, False


def attach_nesine(rows):
    """Taze Nesine verisi varsa maçlara bağlar. (satırlar, bilgi) döner; veri yoksa (satırlar, None)."""
    try:
        import nesine
    except ImportError:
        print("nesine.py bulunamadı; oranlar Avrupa sitelerinden.")
        return rows, None
    data = nesine.load(DATA_DIR / "nesine.json")
    if not data:
        print("Nesine verisi yok ya da 36 saatten eski; oranlar Avrupa sitelerinden.")
        return rows, None
    keys = [(r["Date"].date(), r["HomeTeam"], r["AwayTeam"]) for r, _, _ in rows]
    found = nesine.match_rows(keys, data["matches"])
    out, n_ok, m_nes, m_eur = [], 0, [], []
    for (r, p, mk), m in zip(rows, found):
        r = r.copy()
        r["NESACT"] = True
        r["NES"] = m
        out.append((r, p, mk))
        if m:
            n_ok += 1
            ro = raw_odds(r)
            m_nes.append(sum(1 / x for x in m["ms"]) - 1)
            if ro is not None:
                m_eur.append(float((1 / ro).sum()) - 1)
    used = {id(m) for m in found if m}
    print(f"Nesine: {len(data['matches'])} maç okundu, fikstürdeki {n_ok}/{len(rows)} maç eşleşti.")
    miss = [f"{r['Date']:%d.%m} {r['HomeTeam']} - {r['AwayTeam']}" for (r, _, _), m in zip(rows, found) if not m]
    if miss:
        print("  Nesine'de eşleşmeyen fikstür maçları:", "; ".join(miss[:25]))
    un = [m for m in data["matches"] if id(m) not in used and not m.get("skip")
          and any(k in (m.get("league") or "").upper() for k in KEYWORDS_BIG)]
    if un:
        print("  Nesine'de olup eşleşmeyen büyük lig maçları:",
              "; ".join(f"{m['date']} {m['home']} - {m['away']} ({m['league']})" for m in un[:30]))
    info = {"fetched": data["fetched"], "matched": n_ok, "total": len(rows),
            "margin_nes": float(np.mean(m_nes)) if m_nes else None,
            "margin_euro": float(np.mean(m_eur)) if m_eur else None}
    return out, info


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
GREEN_P = 0.55        # bu olasılık ve üstü: önerilir (yeşil)
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

    nes, act = _nes_of(row) if row is not None else (None, False)
    cs = (None, None, None)
    kgo = (None, None)
    if act:                               # Nesine modu: oran yalnızca Nesine'den, yoksa boş
        if nes:
            o1, ox, o2 = nes["ms"]
            under25, over25 = nes["ou25"]
            cs, kgo = nes["cs"], nes["kg"]
        else:
            o1 = ox = o2 = under25 = over25 = None
    else:
        ro = raw_odds(row) if row is not None else None
        o1, ox, o2 = ro if ro is not None else (None, None, None)
        over25 = _num(row, "Avg>2.5", "B365>2.5")
        under25 = _num(row, "Avg<2.5", "B365<2.5")
    p1, px, p2 = p["p1"], p["px"], p["p2"]

    M = [("Maç Sonucu", [S("MS 1", p1, o1), S("MS X", px, ox), S("MS 2", p2, o2)]),
         ("Çifte Şans", [S("1-X", p1 + px, cs[0]), S("1-2", p1 + p2, cs[1]), S("X-2", px + p2, cs[2])])]
    for line in (1.5, 2.5, 3.5):
        over = m[g > line].sum()
        oo, uo = (over25, under25) if line == 2.5 else (None, None)
        M.append((f"Alt/Üst {line}", [S("Alt", 1 - over, uo), S("Üst", over, oo)]))
    yes = m[1:, 1:].sum()
    M.append(("Karşılıklı Gol", [S("Var", yes, kgo[0]), S("Yok", 1 - yes, kgo[1])]))
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
    today = pd.Timestamp(today_tr())
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


COUPON_LEGS = 3        # kupondaki maç sayısı
COUPON_MIN_ODDS = 3.0  # toplam oran en az bu kadar olmalı
COUPON_MIN_P = 0.55    # kupona girecek her seçeneğin model olasılığı en az bu kadar
N_COUPONS = 4


def _legs(r, p):
    """Gerçek bahis oranı olan seçenekler: MS 1/X/2 ve Alt/Üst 2.5.
    Diğer marketlerde gerçek oran yok, o yüzden kupona girmezler."""
    nes, act = _nes_of(r)
    legs = []
    if act:
        # Nesine modu: yalnızca Nesine'de bulunan maçlar, oranlar Nesine'den.
        # MBS kupondaki en az maç sayısıdır; 4 maçlık kupona giremeyenler elenir.
        if not nes or (nes.get("mbs") or 1) > COUPON_LEGS:
            return []
        ms, (alt, ust), cs, kg = nes["ms"], nes["ou25"], nes["cs"], nes["kg"]
        p1, px, p2, ov, bt = p["p1"], p["px"], p["p2"], p["over25"], p["btts"]
        cand = [("MS 1", p1, ms[0]), ("MS X", px, ms[1]), ("MS 2", p2, ms[2]),
                ("1-X", p1 + px, cs[0]), ("1-2", p1 + p2, cs[1]), ("X-2", px + p2, cs[2]),
                ("Alt 2.5", 1 - ov, alt), ("Üst 2.5", ov, ust),
                ("KG Var", bt, kg[0]), ("KG Yok", 1 - bt, kg[1])]
        legs = [(l, pr, float(o)) for l, pr, o in cand if o]
        top_n = 4
    else:
        ro = raw_odds(r)
        if ro is not None:
            legs += [("MS 1", p["p1"], float(ro[0])), ("MS X", p["px"], float(ro[1])),
                     ("MS 2", p["p2"], float(ro[2]))]
        ov, un = _num(r, "Avg>2.5", "B365>2.5"), _num(r, "Avg<2.5", "B365<2.5")
        if ov:
            legs.append(("Üst 2.5", p["over25"], ov))
        if un:
            legs.append(("Alt 2.5", 1 - p["over25"], un))
        top_n = 3
    # sadece listede de yeşil görünen seçenekler girer (turuncu/kırmızı girmez)
    legs = [l for l in legs if l[1] >= COUPON_MIN_P and tier(l[1], l[2])[0] == "g"]
    return sorted(legs, key=lambda x: -x[1])[:top_n]


def build_coupons(out_rows):
    """Günün maçlarından, toplam oranı COUPON_MIN_ODDS ve üzeri olan ve modele göre en çok tutması
    beklenen kupon; her kupon farklı maçlardan oluşur. Maçlar birbirinden bağımsız varsayılır."""
    today = pd.Timestamp(today_tr())
    by_day = {}
    for k, (r, p, _) in enumerate(out_rows):
        legs = _legs(r, p)
        if legs:
            by_day.setdefault(r["Date"].normalize(), []).append((k, r, legs))
    nearest = None
    for day in sorted(d for d in by_day if d >= today):
        if len(by_day[day]) < COUPON_LEGS:
            continue
        res = _coupons_for_day(day, by_day[day])
        if res["list"]:               # kupon çıkan ilk günü göster
            return res
        if nearest is None:           # çıkmadıysa: güvenilir seçeneklerle ulaşılabilen en yüksek toplam oran
            best = sorted((max(l[2] for l in t[2]) for t in by_day[day]), reverse=True)[:COUPON_LEGS]
            nearest = {"date": day.strftime("%Y-%m-%d"), "list": [], "max_odds": round(math.prod(best), 2)}
    nearest = nearest or {"date": today.strftime("%Y-%m-%d"), "list": []}
    nearest["diag"] = _coupon_diag(out_rows, today)
    return nearest


def _coupon_diag(out_rows, today):
    """Kupon çıkmadığında nedenini göstermek için: kaç maç elendi, nerede."""
    d = {"matches": 0, "nesine": 0, "mbs": 0, "eligible": 0, "days": {}}
    for r, p, _ in out_rows:
        day = r["Date"].normalize()
        if day < today:
            continue
        d["matches"] += 1
        nes, act = _nes_of(r)
        if act:
            if not nes:
                continue
            d["nesine"] += 1
            if (nes.get("mbs") or 1) > COUPON_LEGS:
                d["mbs"] += 1
                continue
        if _legs(r, p):
            d["eligible"] += 1
            k = day.strftime("%Y-%m-%d")
            d["days"][k] = d["days"].get(k, 0) + 1
    return d


def _best_combo(items):
    """Verilen maçlar arasından, toplam oranı COUPON_MIN_ODDS ve üzeri olan, en yüksek ortak olasılıklı kupon."""
    cands = sorted(items, key=lambda t: -t[2][0][1])[:20]
    best = None
    for combo in itertools.combinations(cands, COUPON_LEGS):
        for pick in itertools.product(*[c[2] for c in combo]):
            od = math.prod(x[2] for x in pick)
            if od >= COUPON_MIN_ODDS:
                pr = math.prod(x[1] for x in pick)
                if best is None or pr > best[0]:
                    best = (pr, od, combo, pick)
    return best


def _coupons_for_day(day, items):
    """Her kupon farklı maçlardan oluşur: bir kupon seçilince o maçlar havuzdan çıkar."""
    def key(t):                       # maçın kimliği: tarih + iki takım (sıra numarası değil)
        r = t[1]
        return (str(r["Date"])[:10], str(r["HomeTeam"]).strip().lower(),
                str(r["AwayTeam"]).strip().lower())

    chosen, pool, seen = [], [], set()
    for t in items:                   # aynı maç iki satır olarak gelmişse biri atılır
        if key(t) not in seen:
            seen.add(key(t))
            pool.append(t)
    used_all = set()
    while len(chosen) < N_COUPONS and len(pool) >= COUPON_LEGS:
        best = _best_combo(pool)
        if best is None:
            break
        pr, od, combo, pick = best
        used = {key(c) for c in combo}
        if len(used) != COUPON_LEGS or (used & used_all):
            break                     # tekrar eden maç olamaz; olursa kuponu eklemeden dur
        used_all |= used
        pool = [t for t in pool if key(t) not in used]
        legs = []
        for c, x in zip(combo, pick):
            r = c[1]
            t = str(r.get("Time", "") or "")
            legs.append({"id": c[0], "time": "" if t in ("nan", "NaT") else t[:5],
                         "home": r["HomeTeam"], "away": r["AwayTeam"],
                         "league": LEAGUES.get(r["Div"], DEMO_LEAGUES.get(r["Div"], r["Div"])),
                         "sel": x[0], "o": round(x[2], 2), "p": round(x[1], 3),
                         "t": tier(x[1], x[2])[0],
                         "mbs": (_nes_of(r)[0] or {}).get("mbs")})
        legs.sort(key=lambda l: l["time"])
        chosen.append({"legs": legs, "odds": round(od, 2),
                       "p": round(pr, 4), "ev": round(pr * od, 2)})
    return {"date": day.strftime("%Y-%m-%d"), "list": chosen}


# --------------------------------------------------------------------------
# Önerilen kuponların kaydı ve ertesi gün sonuçlarının kontrolü
# --------------------------------------------------------------------------
COUPON_LOG = DATA_DIR / "kupon_gecmisi.json"


def _load_log():
    try:
        return json.loads(COUPON_LOG.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  dosya yok ya da bozuk: boş başla
        return {}


def save_coupons(coupons, keep_days=120):
    """Sitede gösterilen kuponları gün adıyla kaydeder (aynı gün tekrar çalışırsa üzerine yazar)."""
    if not coupons or not coupons.get("list"):
        return
    log = _load_log()
    log[coupons["date"]] = {"list": [
        {"odds": k["odds"], "p": k["p"],
         "legs": [{x: l[x] for x in ("time", "home", "away", "league", "sel", "o", "p")}
                  for l in k["legs"]]} for k in coupons["list"]]}
    for d in sorted(log)[:-keep_days]:
        del log[d]
    COUPON_LOG.parent.mkdir(parents=True, exist_ok=True)
    COUPON_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


def _grade_day(day, entry, idx):
    """Bir günün kuponlarını gerçek sonuçlarla değerlendirir. Maç ±1 gün toleransla aranır."""
    d0 = pd.Timestamp(day)
    out = []
    for k in entry["list"]:
        legs, bad, pend = [], False, False
        for l in k["legs"]:
            res = None
            for dt, hg, ag in idx.get((str(l["home"]).strip().lower(), str(l["away"]).strip().lower()), []):
                if abs((dt - d0).days) <= 1:
                    res = (int(hg), int(ag))
                    break
            ok = None if res is None else _hit(l["sel"], *res)
            bad |= ok is False
            pend |= ok is None
            legs.append({**l, "score": None if res is None else f"{res[0]}-{res[1]}", "ok": ok})
        out.append({"odds": k["odds"], "p": k["p"], "legs": legs,
                    "status": "lost" if bad else ("pending" if pend else "won")})
    return out


def yesterday_report(hist, today=None):
    """En son geçmiş kupon gününün sonuçları ve tüm kayıtlı günlerin toplam tutma sayısı."""
    log = _load_log()
    today = today or today_tr().strftime("%Y-%m-%d")
    days = sorted(d for d in log if d < today)
    if not days or hist is None or not len(hist):
        return None
    idx = {}
    for dt, h, a, hg, ag in zip(hist["Date"], hist["HomeTeam"], hist["AwayTeam"], hist["FTHG"], hist["FTAG"]):
        if pd.notna(hg) and pd.notna(ag):
            idx.setdefault((str(h).strip().lower(), str(a).strip().lower()), []).append((dt, hg, ag))
    tot = {"won": 0, "lost": 0, "pending": 0, "days": len(days)}
    graded = {}
    for d in days:
        graded[d] = _grade_day(d, log[d], idx)
        for k in graded[d]:
            tot[k["status"]] += 1
    last = days[-1]
    return {"date": last, "list": graded[last], "total": tot}


# --------------------------------------------------------------------------
# Geriye dönük başarı ölçümü
# --------------------------------------------------------------------------
def _sel_list(p, hg, ag):
    """Ana marketlerdeki seçenekler için (modelin olasılığı, gerçekleşti mi)."""
    m = p["matrix"]
    i, j = np.indices(m.shape)
    g = i + j
    tot = hg + ag
    both = hg > 0 and ag > 0
    out = [(p["p1"], hg > ag), (p["px"], hg == ag), (p["p2"], hg < ag),
           (p["p1"] + p["px"], hg >= ag), (p["p1"] + p["p2"], hg != ag),
           (p["px"] + p["p2"], hg <= ag)]
    for line in (1.5, 2.5, 3.5):
        over = float(m[g > line].sum())
        out += [(over, tot > line), (1 - over, tot < line)]
    yes = float(m[1:, 1:].sum())
    out += [(yes, both), (1 - yes, not both)]
    return out


def _hit(sel, hg, ag):
    """Seçenek (kupondaki etiketiyle) skora göre tuttu mu; bilinmeyen etiket için None."""
    t = hg + ag
    return {"MS 1": hg > ag, "MS X": hg == ag, "MS 2": hg < ag,
            "1-X": hg >= ag, "1-2": hg != ag, "X-2": hg <= ag,
            "Üst 2.5": t >= 3, "Alt 2.5": t < 3,
            "KG Var": hg > 0 and ag > 0, "KG Yok": hg == 0 or ag == 0}.get(sel)


def _leg_hit(r, sel):
    return _hit(sel, r["FTHG"], r["FTAG"])


def performance(hist, days=120, step=7):
    """Son `days` günü, her maçtan önceki veriyle tahmin edip gerçek sonuçla kıyaslar.
    Model her hafta sadece o güne kadarki maçlarla yeniden kurulur (geleceği görmez)."""
    end = hist["Date"].max()
    start = end - timedelta(days=days)
    recs = []
    for div, df in hist.groupby("Div"):
        df = df.sort_values("Date")
        d = start
        while d <= end:
            train = df[df["Date"] < d]
            test = df[(df["Date"] >= d) & (df["Date"] < d + timedelta(days=step))]
            d += timedelta(days=step)
            if test.empty or len(train) < 100:
                continue
            model = Model().fit(train, ref_date=test["Date"].min())
            base = np.array([(train["FTHG"] > train["FTAG"]).mean(),
                             (train["FTHG"] == train["FTAG"]).mean(),
                             (train["FTHG"] < train["FTAG"]).mean()])
            for _, r in test.iterrows():
                if model.knows(r["HomeTeam"], r["AwayTeam"]):
                    recs.append({"r": r, "base": base,
                                 "p": model.predict(r["HomeTeam"], r["AwayTeam"])})
    if len(recs) < 30:
        return None

    ys = np.array([0 if x["r"]["FTHG"] > x["r"]["FTAG"]
                   else (1 if x["r"]["FTHG"] == x["r"]["FTAG"] else 2) for x in recs])
    P = np.array([[x["p"]["p1"], x["p"]["px"], x["p"]["p2"]] for x in recs])
    B = np.array([x["base"] for x in recs])

    def ll(Pm, y):
        return float(-np.mean(np.log(np.clip(Pm[np.arange(len(y)), y], 1e-9, 1))))

    def acc(Pm, y):
        return float((Pm.argmax(1) == y).mean())

    perf = {"days": days, "matches": len(recs),
            "from": start.strftime("%d.%m.%Y"), "to": end.strftime("%d.%m.%Y"),
            "ms": {"acc": acc(P, ys), "ll": ll(P, ys), "base_ll": ll(B, ys)}}
    mk_idx = [k for k, x in enumerate(recs) if market_probs(x["r"]) is not None]
    if len(mk_idx) >= 30:
        MK = np.array([market_probs(recs[k]["r"]) for k in mk_idx])
        yk = ys[mk_idx]
        perf["ms"].update({"n_mkt": len(mk_idx), "mkt_acc": acc(MK, yk), "mkt_ll": ll(MK, yk),
                           "model_acc_same": acc(P[mk_idx], yk), "model_ll_same": ll(P[mk_idx], yk)})

    over_hit = np.array([(x["r"]["FTHG"] + x["r"]["FTAG"]) >= 3 for x in recs])
    over_p = np.array([x["p"]["over25"] for x in recs])
    perf["ou"] = {"acc": float(((over_p > .5) == over_hit).mean()),
                  "over_rate": float(over_hit.mean()), "model_over": float(over_p.mean())}

    sels = [s for x in recs for s in _sel_list(x["p"], x["r"]["FTHG"], x["r"]["FTAG"])]
    ps = np.array([s[0] for s in sels])
    hs = np.array([s[1] for s in sels], dtype=float)
    cal = []
    for lo, hi in ((0, ORANGE_P), (ORANGE_P, GREEN_P), (GREEN_P, .70), (.70, .80), (.80, 1.01)):
        mask = (ps >= lo) & (ps < hi)
        if mask.sum() >= 10:
            cal.append({"label": f"%{round(lo*100)}-{min(round(hi*100), 100)}" if lo else f"%{round(hi*100)} altı",
                        "t": tier(lo + 1e-9)[0], "n": int(mask.sum()),
                        "p": float(ps[mask].mean()), "hit": float(hs[mask].mean())})
    perf["cal"] = cal

    by_day = {}
    for k, x in enumerate(recs):
        legs = _legs(x["r"], x["p"])
        if legs:
            by_day.setdefault(x["r"]["Date"].normalize(), []).append((k, x["r"], legs))
    n_cp = hits = 0
    ret, exp_p, odds_l = 0.0, [], []
    for day, items in by_day.items():
        if len(items) < COUPON_LEGS:
            continue
        for cp in _coupons_for_day(day, items)["list"]:
            n_cp += 1
            exp_p.append(cp["p"])
            odds_l.append(cp["odds"])
            if all(_leg_hit(recs[l["id"]]["r"], l["sel"]) for l in cp["legs"]):
                hits += 1
                ret += cp["odds"]
    perf["coupons"] = ({"n": n_cp, "hit": hits, "exp": float(np.mean(exp_p)),
                        "avg_odds": float(np.mean(odds_l)), "roi": (ret - n_cp) / n_cp}
                       if n_cp else {"n": 0})
    return perf


def _poisson_probs(lh, la):
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    i, j = np.indices(m.shape)
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum(), m[(i + j) >= 3].sum()


def demo_history():
    """Kurgusal geçmiş sonuçlar ve piyasa oranları (sadece demo için)."""
    rng = np.random.default_rng(5)
    today = pd.Timestamp(today_tr())
    rows = []
    for div, names in DEMO_TEAMS.items():
        n = len(names)
        att, dfn = rng.normal(0, .3, n), rng.normal(0, .3, n)
        for rnd in range(1, 101):
            day = today - timedelta(days=7 * rnd)
            order = rng.permutation(n)
            for q in range(0, n, 2):
                a, b = order[q], order[q + 1]
                lh, la = np.exp(.55 + att[a] - dfn[b]), np.exp(.25 + att[b] - dfn[a])
                p1, px, p2, po = _poisson_probs(lh, la)
                pr = np.array([p1, px, p2]) * rng.normal(1, .03, 3)
                o = np.round(1 / (pr * 1.06), 2)
                rows.append({"Div": div, "Date": day, "HomeTeam": names[a], "AwayTeam": names[b],
                             "FTHG": int(rng.poisson(lh)), "FTAG": int(rng.poisson(la)),
                             "B365H": o[0], "B365D": o[1], "B365A": o[2],
                             "B365>2.5": round(1 / (po * 1.06), 2),
                             "B365<2.5": round(1 / ((1 - po) * 1.06), 2)})
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def site_payload(out_rows, demo=False, perf=None, nes_info=None, yday=None):
    matches = []
    for k, (r, p, _) in enumerate(out_rows):
        t = str(r.get("Time", "") or "")
        nes = _nes_of(r)[0]
        matches.append({
            "id": k, "div": r["Div"],
            "league": LEAGUES.get(r["Div"], DEMO_LEAGUES.get(r["Div"], r["Div"])),
            "date": r["Date"].strftime("%Y-%m-%d"),
            "time": "" if t in ("nan", "NaT") else t[:5],
            "home": r["HomeTeam"], "away": r["AwayTeam"],
            "xg": [round(p["xg_h"], 2), round(p["xg_a"], 2)],
            "nes": bool(nes), "mbs": (nes or {}).get("mbs"),
            "markets": build_markets(p, r)})
    return {"generated": datetime.combine(today_tr(), datetime.min.time()).strftime("%d.%m.%Y") + " " + _now_tr().strftime("%H:%M"), "demo": demo,
            "green": GREEN_P, "orange": ORANGE_P, "matches": matches,
            "coupon_legs": COUPON_LEGS, "min_odds": COUPON_MIN_ODDS, "n_coupons": N_COUPONS,
            "odds_source": "nesine" if nes_info else "avrupa", "nesine": nes_info,
            "coupons": build_coupons(out_rows), "yesterday": yday, "performance": perf}


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
    perf = None
    try:
        print("Geçmiş başarı hesaplanıyor...")
        perf = performance(demo_history() if demo else load_results())
        if perf is None:
            print("  (başarı hesabı için yeterli geçmiş maç yok)")
    except Exception as e:  # noqa: BLE001  başarı hesabı sayfayı bozmasın
        print(f"  ! başarı hesabı atlandı: {e}")
    nes_info = None
    if not demo:
        try:
            rows, nes_info = attach_nesine(rows)
        except Exception as e:  # noqa: BLE001  Nesine kısmı sayfayı bozmasın
            print(f"  ! Nesine verisi kullanılamadı, Avrupa oranlarıyla devam: {e}")
            rows, nes_info = [(r, p, mk) for r, p, mk in rows], None
    yday = None
    if not demo:
        try:
            yday = yesterday_report(load_results())
        except Exception as e:  # noqa: BLE001  dünkü kuponlar sayfayı bozmasın
            print(f"  ! dünkü kupon sonuçları atlandı: {e}")
    payload = site_payload(rows, demo, perf, nes_info, yday)
    if not demo:
        try:
            save_coupons(payload["coupons"])
        except Exception as e:  # noqa: BLE001
            print(f"  ! kupon kaydı yazılamadı: {e}")
    path = write_site(payload, out)
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
