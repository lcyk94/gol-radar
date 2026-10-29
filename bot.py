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
import os
import re
import time
import math
import sys
import urllib.request
import webbrowser
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pazarlar as pz
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

# football-data.co.uk "new" biçimindeki ek ligler (tek dosyada tüm sezonlar)
EXTRA = {
    "DNK": "Danimarka Süper Lig", "SWE": "İsveç Allsvenskan", "NOR": "Norveç Eliteserien",
    "FIN": "Finlandiya Veikkausliiga", "AUT": "Avusturya Bundesliga", "SWZ": "İsviçre Super League",
    "POL": "Polonya Ekstraklasa", "ROU": "Romanya Liga 1", "RUS": "Rusya Premier Lig",
    "IRL": "İrlanda Premier Lig", "BRA": "Brezilya Serie A", "ARG": "Arjantin Primera",
    "MEX": "Meksika Liga MX", "USA": "ABD MLS", "JPN": "Japonya J1 Lig", "CHN": "Çin Süper Lig",
}
LEAGUES.update(EXTRA)

ODDS_SETS = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"),
             ("PSH", "PSD", "PSA")]
ODDS_COLS = []        # yurtdışı oranları hiçbir yerde kullanılmaz; geçmiş dosyalardan yalnızca maç sonuçları alınır
KEEP = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"]

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


_FAILS = {"n": 0}      # üst üste başarısız indirme sayısı (sunucu yanıt vermiyorsa vakit kaybetme)


def fetch(url, dest, force, tries=2, timeout=45):
    """İndirir. Başarısız olursa daha önce indirilmiş kopya varsa onu kullanır (True döner)."""
    if dest.exists() and not force:
        return True
    if _FAILS["n"] >= 6:
        print(f"  ! atlandı (sunucu yanıt vermiyor): {url}")
        return dest.exists()
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if data.lstrip()[:5].lower() in (b"<html", b"<!doc"):
                print(f"  ! veri yerine web sayfası geldi: {url}")
                break
            dest.write_bytes(data)
            _FAILS["n"] = 0
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  ! indirilemedi ({k + 1}/{tries}): {url} ({e})")
    _FAILS["n"] += 1
    if dest.exists():
        print(f"    eski kopya kullanılıyor: {dest.name}")
        return True
    return False


def update(leagues=None):
    DATA_DIR.mkdir(exist_ok=True)
    seasons = season_codes(today_tr())
    current = seasons[-1]
    leagues_arg = leagues
    leagues = leagues or [d for d in LEAGUES if d not in EXTRA]
    ok = 0
    for div in leagues:
        for s in seasons:
            dest = DATA_DIR / f"{div}_{s}.csv"
            if fetch(f"{BASE}/mmz4281/{s}/{div}.csv", dest, force=(s == current)):
                ok += 1
    _FAILS["n"] = 0
    if not leagues_arg:
        for code in EXTRA:
            fetch(f"{BASE}/new/{code}.csv", DATA_DIR / f"{code}_new.csv", force=True, tries=2, timeout=45)
        _FAILS["n"] = 0
    _FAILS["n"] = 0
    print(f"{ok} sonuç dosyası hazır (yalnızca takım gücü için geçmiş sonuçlar). Veri klasörü: {DATA_DIR}")
    if ok == 0:
        sys.exit("Geçmiş sonuçlar indirilemedi, yukarıdaki '!' satırlarına bak.")


def _read(path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")      # BOM'u temizler
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    if "HomeTeam" not in df.columns and "Home" in df.columns:      # "new" biçim
        df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
    if "Div" not in df.columns:
        if path.stem.startswith("fixtures"):
            raise ValueError("fixtures dosyasında 'Div' sütunu yok")
        df["Div"] = path.stem.split("_")[0]
    return df


def _clean(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed",
                                errors="coerce")
    for c in ("FTHG", "FTAG", "HTHG", "HTAG"):
        if c in df.columns:
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
        return predict_lambdas(lh, la, self.rho)


def predict_lambdas(lh, la, rho=RHO):
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    r = rho
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


_SAME = {}


def _same(a, b):
    k = (a, b)
    if k not in _SAME:
        import nesine
        _SAME[k] = nesine.same_team(a, b)
    return _SAME[k]


def nesine_fixtures(fx, hist, today, days, leagues=None):
    """Nesine bülteninde olup fikstür dosyasında bulunmayan maçları, iki takımı da geçmiş verilerde
    tanınan liglerden fikstür satırı olarak ekler. (fikstür tablosu, eklenen sayı) döner."""
    try:
        import nesine
    except ImportError:
        return fx, 0
    data = nesine.load(DATA_DIR / "nesine.json")
    if not data:
        return fx, 0
    nes = [m for m in data["matches"] if not m.get("skip") and m.get("date")]
    nes = [m for m in nes if today.date() <= date.fromisoformat(m["date"]) <= (today + timedelta(days=days)).date()]
    keys = [(r["Date"].date(), r["HomeTeam"], r["AwayTeam"]) for _, r in fx.iterrows()]
    taken = {id(m) for m in nesine.match_rows(keys, nes) if m}
    recent = hist[hist["Date"] >= today - timedelta(days=400)]
    teams = {}
    for div in [d for d in LEAGUES if d in set(recent["Div"])]:
        if leagues and div not in leagues:
            continue
        h = recent[recent["Div"] == div]
        teams[div] = sorted(set(h["HomeTeam"]) | set(h["AwayTeam"]))
    add = []
    for m in nes:
        if id(m) in taken:
            continue
        for div, ts in teams.items():
            hh = next((t for t in ts if _same(t, m["home"])), None)
            if hh is None:
                continue
            aa = next((t for t in ts if t != hh and _same(t, m["away"])), None)
            if aa is None:
                continue
            add.append({"Div": div, "Date": pd.Timestamp(m["date"]), "HomeTeam": hh, "AwayTeam": aa, "NID": m.get("id")})
            break
    if add:
        fx = pd.concat([fx, pd.DataFrame(add)], ignore_index=True)
    return fx, len(add)


# Ligler arası maçlar (Avrupa kupaları vb.) için lig gücü farkı: log-gol birimi, İngiltere Premier Lig = 0.
# Veriden ölçülemez (ligler arası maç yok), kaba bir tahmindir; bu maçlar "tahmini" etiketlenir ve kupona girmez.
LEAGUE_STRENGTH = {"E0": 0.0, "SP1": -0.02, "D1": -0.05, "I1": -0.05, "F1": -0.12, "P1": -0.20,
                   "N1": -0.22, "B1": -0.27, "T1": -0.30, "SC0": -0.38, "G1": -0.38, "E1": -0.30,
                   "AUT": -0.38, "SWZ": -0.40, "DNK": -0.38, "NOR": -0.42, "SWE": -0.42, "POL": -0.48,
                   "RUS": -0.40, "ROU": -0.48, "FIN": -0.55, "IRL": -0.60}
_DEFAULT_STRENGTH = -0.50


def cross_predict(mh, ma, dh, da, h, a):
    sh = LEAGUE_STRENGTH.get(dh, _DEFAULT_STRENGTH)
    sa = LEAGUE_STRENGTH.get(da, _DEFAULT_STRENGTH)
    mu, home = (mh.mu + ma.mu) / 2, (mh.home + ma.home) / 2
    lh = np.exp(mu + home + mh.att[h] - ma.dfn[a] + sh - sa)
    la = np.exp(mu + ma.att[a] - mh.dfn[h] + sa - sh)
    return predict_lambdas(lh, la, RHO)


def _fit_lambdas(target, x0=(1.4, 1.1)):
    """Hedef olasılıklara (1, X, 2 ve varsa Üst 2.5) en yakın Poisson gol beklentilerini bulur."""
    def loss(x):
        q = predict_lambdas(np.exp(x[0]), np.exp(x[1]), RHO)
        e = (q["p1"] - target[0]) ** 2 + (q["px"] - target[1]) ** 2 + (q["p2"] - target[2]) ** 2
        if target[3] is not None:
            e += (q["over25"] - target[3]) ** 2
        return e

    res = minimize(loss, np.log(x0), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-8})
    lh, la = np.exp(np.clip(res.x, -2.5, 1.8))
    return predict_lambdas(lh, la, RHO)


MARKET_MARGIN = 1.20      # Nesine'nin tipik bahisçi payı (eksik oranı tamamlamak için)


def _market_targets(nm):
    """Nesine oranlarından (hedef 1-X-2 olasılıkları, Üst 2.5 olasılığı). Bir MS oranı eksikse (çok net favori
    ya da kapalı seçenek) kalan olasılık ona verilir; MS yoksa Çifte Şans oranlarından 1-X-2 çıkarılır."""
    ms = list(nm.get("ms") or [])
    pm = None
    if len(ms) == 3 and all(ms):
        inv = 1 / np.array(ms, float)
        pm = inv / inv.sum()
    elif len(ms) == 3 and sum(1 for x in ms if x) == 2:
        q = [(1 / x / MARKET_MARGIN) if x else None for x in ms]
        rest = max(0.02, min(0.96, 1 - sum(v for v in q if v)))
        pm = np.array([v if v else rest for v in q], float)
        pm /= pm.sum()
    if pm is None:
        cs = list(nm.get("cs") or [])
        if len(cs) == 3 and all(cs):
            a, b, c = (1 / np.array(cs, float))          # P(1X), P(12), P(X2) (paylı)
            k = 2 / (a + b + c)
            a, b, c = a * k, b * k, c * k
            v = np.clip(np.array([(a + b - c) / 2, (a + c - b) / 2, (b + c - a) / 2]), 0.02, 0.96)
            pm = v / v.sum()
    if pm is None:
        return None
    ou = nm.get("ou25") or []
    po = None
    if len(ou) == 2 and all(ou):
        po = (1 / ou[1]) / (1 / ou[0] + 1 / ou[1])
    return pm, po


def market_predict(nm):
    """Modeli olmayan maç: Nesine MS (ve varsa Alt/Üst 2.5) oranlarından bahisçi payı temizlenerek
    gol beklentisi çıkarılır. Bu bir piyasa olasılığıdır, modelin kendi tahmini değildir."""
    t = _market_targets(nm)
    if t is None:
        return None
    pm, po = t
    return _fit_lambdas([pm[0], pm[1], pm[2], po])


def blend_predict(pmodel, nm, w=0.5):
    """İki takımın da kendi liginde modeli varsa (Avrupa kupaları): ligler arası gücü ölçemediğimiz için
    modelin tahmini ile piyasa olasılığı yarı yarıya karıştırılır (piyasa lig farkını, kadro ve motivasyonu bilir)."""
    t = _market_targets(nm)
    if t is None:
        return None
    pm, po = t
    tgt = [w * pmodel["p1"] + (1 - w) * pm[0], w * pmodel["px"] + (1 - w) * pm[1],
           w * pmodel["p2"] + (1 - w) * pm[2],
           w * pmodel["over25"] + (1 - w) * po if po is not None else pmodel["over25"]]
    return _fit_lambdas(tgt, (max(pmodel["xg_h"], .3), max(pmodel["xg_a"], .3)))


def _is_euro_cup(name):
    n = str(name or "").upper().translate(str.maketrans("İŞÇĞÜÖ", "ISCGUO"))
    return any(k in n for k in ("SAMPIYONLAR LIGI", "AVRUPA LIGI", "KONFERANS LIGI")) and "GENC" not in n


def extra_rows(out_rows, hist_all, models, today, days, leagues=None):
    """Nesine'de olup henüz tahmini olmayan maçlar:
       1) iki takım da farklı liglerde tanınıyorsa (Avrupa kupaları): ligler arası tahmin,
       2) tanınmıyorsa: Nesine oranlarından piyasa olasılığı. İkisi de 'tahmini' etiketlenir."""
    try:
        import nesine
    except ImportError:
        return []
    data = nesine.load(DATA_DIR / "nesine.json")
    if not data or leagues:
        return []
    lo, hi = today.date(), (today + timedelta(days=days)).date()
    # kadın / genç / yedek takım maçları dahil: onlar modele sokulmaz (isim karışmasın), yalnızca piyasa olasılığı gösterilir
    nes = [m for m in data["matches"] if m.get("date")
           and lo <= date.fromisoformat(m["date"]) <= hi]
    keys = [(r["Date"].date(), r["HomeTeam"], r["AwayTeam"]) for r, _, _ in out_rows]
    taken = {id(m) for m in nesine.match_rows(keys, nes) if m}
    recent = hist_all[hist_all["Date"] >= today - timedelta(days=400)].sort_values("Date")
    latest, tots = {}, {}
    for dv, h, a, dt, hg, ag in zip(recent["Div"], recent["HomeTeam"], recent["AwayTeam"], recent["Date"],
                                    recent["FTHG"], recent["FTAG"]):
        if dv in LEAGUES:
            latest[h] = (dv, dt)
            latest[a] = (dv, dt)
            tots.setdefault(h, []).append(float(hg + ag))
            tots.setdefault(a, []).append(float(hg + ag))

    def tempo(t, k=8, base=2.7):
        """Takımın son 12 maçındaki ortalama toplam gol, genel ortalamaya doğru çekilmiş (az maç = az güven)."""
        x = tots.get(t, [])[-12:]
        return (sum(x) + k * base) / (len(x) + k)
    by_canon = {}
    for t, (dv, _) in latest.items():
        by_canon.setdefault(nesine.canon(t), (dv, t))
    names = list(latest)
    memo = {}

    def find(name):
        if name in memo:
            return memo[name]
        hit = by_canon.get(nesine.canon(name))
        if hit is None:
            for t in names:
                if _same(t, name):
                    hit = (latest[t][0], t)
                    break
        memo[name] = hit
        return hit

    def model(div):
        if div not in models:
            h = hist_all[hist_all["Div"] == div]
            models[div] = Model().fit(h, ref_date=today) if len(h) >= 50 else None
        return models[div]

    book = None                       # ClubElo kullanılmıyor (otomatik indirmeye izin vermiyor)
    out, n_cross, n_mkt, n_elo, miss, unm = [], 0, 0, 0, [], []
    for m in nes:
        if id(m) in taken:
            continue
        hh, aa = (None, None) if m.get("skip") else (find(m["home"]), find(m["away"]))
        src, p, hn, an = None, None, m["home"], m["away"]
        if book is not None:
            eh = book.lookup(hh[1] if hh else m["home"], DIV_COUNTRY.get(hh[0]) if hh else None)
            ea = book.lookup(aa[1] if aa else m["away"], DIV_COUNTRY.get(aa[0]) if aa else None)
            if eh and ea and eh[0] != ea[0]:
                total = 2.65
                if hh and aa:
                    total = (tempo(hh[1]) + tempo(aa[1])) / 2          # iki takımın gol temposu
                    mh, ma = model(hh[0]), model(aa[0])
                    if mh and ma and mh.knows(hh[1]) and ma.knows(aa[1]):
                        q = cross_predict(mh, ma, hh[0], aa[0], hh[1], aa[1])   # hücum/savunma gücünden gelen beklenti
                        total = (total + (q["xg_h"] + q["xg_a"])) / 2
                    total = float(np.clip(total, 2.0, 3.4))
                p, src = elo_predict(eh[3], ea[3], total), "elo"
            elif (eh or ea) and len(miss) < 25:
                miss.append(f"{m['home']} - {m['away']} ({'ev' if not eh else 'dep'} Elo'da bulunamadı)")
        if p is None and hh and aa and hh[1] != aa[1]:
            mh, ma = model(hh[0]), model(aa[0])
            if mh and ma and mh.knows(hh[1]) and ma.knows(aa[1]):
                pc = cross_predict(mh, ma, hh[0], aa[0], hh[1], aa[1])
                pb = blend_predict(pc, m)
                if pb is not None:
                    p, src = pb, "blend"
                else:
                    p, src = pc, "cross"
                hn, an = hh[1], aa[1]
        if p is None:
            p = market_predict(m)
            src = "market"
            if len(unm) < 60 and not m.get("skip"):
                unm.append(f"{m.get('league')} | {m['home']} - {m['away']} | ev {'var' if hh else 'YOK'}, dep {'var' if aa else 'YOK'}")
        if p is None:                       # ne model ne MS oranı var: genel ortalama; maç bültende ise listede olmalı
            p, src = predict_lambdas(1.45, 1.15, RHO), "prior"
        n_cross += src == "cross"
        n_mkt += src == "market"
        n_elo += src == "blend"
        r = pd.Series({"Div": "NES", "Date": pd.Timestamp(m["date"]), "Time": m.get("time") or "",
                       "HomeTeam": hn, "AwayTeam": an, "SRC": src, "NID": m.get("id"), "LeagueName": m.get("league") or "",
                       "EURO": _is_euro_cup(m.get("league"))})
        out.append((r, p, None))
    print(f"Nesine'den eklenen: ligler arası model+piyasa {n_elo}, ligler arası yalnız model {n_cross}, yalnızca piyasa olasılığı {n_mkt} maç.")
    if unm:
        print("  Modeli kurulamayan (geçmiş verisinde takımı bulunamayan) maçlar:")
        for u in unm:
            print("   MODELSİZ", u)
    if miss:
        print("  Bir takımı Elo'da bulunan, diğeri bulunamayan maçlar:", "; ".join(miss))
    return out


def collect(leagues, days):
    """Önümüzdeki günlerin maçları için (satır, tahmin, piyasa) listesi."""
    today = pd.Timestamp(today_tr())
    # Maç listesi YALNIZCA Nesine bülteninden gelir (fikstür dosyası kullanılmaz)
    fx = pd.DataFrame({"Div": pd.Series(dtype=str), "Date": pd.Series(dtype="datetime64[ns]"),
                       "HomeTeam": pd.Series(dtype=str), "AwayTeam": pd.Series(dtype=str)})
    hist_all = load_results()
    fx, n_add = nesine_fixtures(fx, hist_all, today, days, leagues)
    print(f"Nesine bülteninden, geçmiş verisi olan liglerde modelle eşleşen maç: {n_add}.")
    hist = hist_all[hist_all["Div"].isin(list(fx["Div"].unique()))]
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
    try:
        out_rows += extra_rows(out_rows, hist_all, models, today, days, leagues)
    except Exception as e:  # noqa: BLE001  ek maçlar ana listeyi bozmasın
        print(f"  ! Nesine'den ek maçlar eklenemedi: {e}")
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
    by_id = {m.get("id"): m for m in data["matches"] if m.get("id") is not None}
    for i, (r, _, _) in enumerate(rows):          # önce Nesine maç numarası: takım adı farkı (Leipzig/RB Leipzig) önemsiz
        if r.get("NID") in by_id:
            found[i] = by_id[r["NID"]]
    miss_idx = [i for i, (f, (r, _, _)) in enumerate(zip(found, rows)) if f is None and r.get("SRC") == "market"]
    if miss_idx:                          # genç/kadın/yedek maçlar yalnızca piyasa satırlarına bağlanır
        f2 = nesine.match_rows([keys[i] for i in miss_idx], [m for m in data["matches"] if m.get("skip")],
                               include_skip=True)
        for i, m2 in zip(miss_idx, f2):
            found[i] = m2
    out, n_ok, m_nes, m_eur = [], 0, [], []
    for (r, p, mk), m in zip(rows, found):
        r = r.copy()
        r["NESACT"] = True
        r["NES"] = m
        out.append((r, p, mk))
        if m:
            n_ok += 1
            ro = raw_odds(r)
            if all(m["ms"]):
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
    from collections import Counter
    nd = Counter(m["date"] for m in data["matches"] if not m.get("skip"))
    fd = Counter(r["Date"].strftime("%Y-%m-%d") for r, _, _ in rows)
    print("  Nesine bülteninde günlere göre maç:", dict(sorted(nd.items())))
    print("  Fikstürde günlere göre maç:", dict(sorted(fd.items())))
    info = {"nes_days": dict(sorted(nd.items())), "fix_days": dict(sorted(fd.items())),
            "fetched": data["fetched"], "matched": n_ok, "total": len(rows),
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

    src = row.get("SRC") if row is not None else None

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
    idx = {}                              # Nesine'nin (bahis numarası, çizgi) -> {seçenek no: oran}
    if act and nes:
        for mk_ in nes.get("markets") or []:
            idx[(mk_.get("mtid"), round(float(mk_.get("line") or 0), 2))] = {
                o_["n"]: o_["o"] for o_ in mk_.get("outs") or [] if o_.get("o")}

    def NO(mtid, line, n):
        """Nesine'nin bu bahis/çizgi/seçenek için açtığı oran; yoksa None (uydurulmaz)."""
        return idx.get((mtid, round(float(line), 2)), {}).get(n)

    M = [("Maç Sonucu", [S("MS 1", p1, o1), S("MS X", px, ox), S("MS 2", p2, o2)]),
         ("Çifte Şans", [S("1-X", p1 + px, cs[0]), S("1-2", p1 + p2, cs[1]), S("X-2", px + p2, cs[2])])]
    for line in (1.5, 2.5, 3.5):
        over = m[g > line].sum()
        oo, uo = (over25, under25) if line == 2.5 else ((NO(11, 1.5, 2), NO(11, 1.5, 1)) if line == 1.5
                                                       else (NO(13, 3.5, 2), NO(13, 3.5, 1)))
        M.append((f"Alt/Üst {line}", [S("Alt", 1 - over, uo), S("Üst", over, oo)]))
    yes = m[1:, 1:].sum()
    M.append(("Karşılıklı Gol", [S("Var", yes, kgo[0]), S("Yok", 1 - yes, kgo[1])]))
    # Bahis numaraları Nesine'nin kendi sayfa tablosundan doğrulandı (43 Toplam Gol Aralığı, 20 Ev Sahibi 1,5 Gol,
    # 29 Deplasman 1,5 Gol, 268 Handikaplı MS, 343 Maç Sonucu & 2,5 Alt/Üst). 0.5 çizgileri için numara doğrulanmadı.
    M.append(("Toplam Gol", [S("0-1", m[g <= 1].sum(), NO(43, 0, 1)), S("2-3", m[(g >= 2) & (g <= 3)].sum(), NO(43, 0, 2)),
                             S("4-5", m[(g >= 4) & (g <= 5)].sum(), NO(43, 0, 3)), S("6+", m[g >= 6].sum(), NO(43, 0, 4))]))
    M.append(("Ev Sahibi Golleri", [S("0.5 Alt", m[0, :].sum()), S("0.5 Üst", 1 - m[0, :].sum()),
                                    S("1.5 Alt", m[:2, :].sum(), NO(20, 1.5, 1)), S("1.5 Üst", m[2:, :].sum(), NO(20, 1.5, 2))]))
    M.append(("Deplasman Golleri", [S("0.5 Alt", m[:, 0].sum()), S("0.5 Üst", 1 - m[:, 0].sum()),
                                    S("1.5 Alt", m[:, :2].sum(), NO(29, 1.5, 1)), S("1.5 Üst", m[:, 2:].sum(), NO(29, 1.5, 2))]))
    M.append(("Handikaplı MS (Ev -1)", [S("1", m[d >= 2].sum(), NO(268, -1, 1)), S("X", m[d == 1].sum(), NO(268, -1, 2)),
                                        S("2", m[d <= 0].sum(), NO(268, -1, 3))]))
    M.append(("Handikaplı MS (Dep -1)", [S("1", m[d >= 0].sum(), NO(268, 1, 1)), S("X", m[d == -1].sum(), NO(268, 1, 2)),
                                         S("2", m[d <= -2].sum(), NO(268, 1, 3))]))
    combo = []
    for lab, mask, ia, iu in (("1", d > 0, 1, 4), ("X", d == 0, 2, 5), ("2", d < 0, 3, 6)):
        combo += [S(f"{lab} & Alt", m[mask & (g <= 2)].sum(), NO(343, 2.5, ia)),
                  S(f"{lab} & Üst", m[mask & (g >= 3)].sum(), NO(343, 2.5, iu))]
    M.append(("MS ve Alt/Üst 2.5", combo))
    top = np.argsort(-m.ravel())[:8]
    M.append(("Doğru Skor", [S(f"{a}-{b}", m[a, b])
                             for a, b in (np.unravel_index(k, m.shape) for k in top)]))
    res = [{"name": n, "sels": s} for n, s in M]
    if src in ("elo", "market", "cross", "prior"):
        # yalnızca 1-X-2 (ve Alt/Üst) bilgisi var; gol sayısına bağlı diğer marketler tahmini sayılır
        for k in res:
            if k["name"] not in ("Maç Sonucu", "Çifte Şans"):
                for x in k["sels"]:
                    if x["t"] == "g":
                        x["t"], x["n"] = "o", "tahmini: yalnızca 1-X-2 verisi var"
    return res


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
COUPON_MIN_LEG_ODDS = 1.20   # kupondaki tek bir seçeneğin en düşük oranı (daha düşüğü gereksiz risk)
COUPON_MIN_ODDS = 3.0  # toplam oran en az bu kadar olmalı
COUPON_MIN_P = 0.55    # kupona girecek her seçeneğin model olasılığı en az bu kadar
MIN_EDGE = 0.10        # modelin olasılığı sıradan bir lig maçındakinden en az 10 puan yüksek değilse seçenek kupona girmez
FAM_PEN = 0.10         # aynı pazar ailesi önceki kuponlarda kullanıldıkça seçim puanından düşülen ceza (çeşitlilik)
DUP_PEN = 0.30         # aynı kuponda aynı pazar ailesinin ikinci kez kullanılması cezası
N_COUPONS = 4
BASIC_FAMILIES = ("MS", "ÇŞ", "KG", "AÜ")   # başarı ölçümü yoksa yalnızca bunlar kullanılır

# Kupon seçiminin bağlamı (site üretilirken cmd_site doldurur): ilk yarı gol payları, pazar doğrulaması, pazar ailesi
# kapısı, maç bağlamı (form/tarz), lig ortalamaları.
ENV = {"shares": {}, "valid": None, "gate": None, "ctxf": None, "avg": {}, "cx": {}, "bx": {}}


def _league_avgs(hist):
    """Lig başına ortalama ev/deplasman golü (kıyas tabanı: 'sıradan bir maç')."""
    out = {}
    if hist is None or not len(hist):
        return out
    out[None] = (float(hist["FTHG"].mean()), float(hist["FTAG"].mean()))
    for div, x in hist.groupby("Div"):
        if len(x) >= 100:
            out[div] = (float(x["FTHG"].mean()), float(x["FTAG"].mean()))
    return out


def _ctx_of(p, div):
    key = id(p)
    if key not in ENV["cx"]:
        ENV["cx"][key] = (p, pz.Ctx(p, pz.shares_for(ENV["shares"], div)))
    return ENV["cx"][key][1]


def _base_of(div):
    """O ligin ortalama maçı için aynı olasılık hesabı: bir seçeneğin 'sıradan' olasılığı."""
    key = div if div in ENV["avg"] else None
    if key not in ENV["bx"]:
        a = ENV["avg"].get(key) or ENV["avg"].get(None)
        if not a:
            return None
        ENV["bx"][key] = pz.Ctx(predict_lambdas(a[0], a[1], RHO), pz.shares_for(ENV["shares"], div))
    return ENV["bx"][key]


_ALT = re.compile(r"\bAlt\b")


def _legs(r, p):
    """Maçta Nesine'nin açtığı BÜTÜN bahis türlerinden (pazarlar.py'de adı ve seçenek sırası doğrulananlar) kupon adayları.
    Sıralama oranla değil botun analiziyle yapılır: 'edge' = modelin olasılığının, o ligin sıradan bir maçındaki aynı
    seçeneğin olasılığına göre kaç puan yukarıda olduğu, p - b (zaten neredeyse kesin olan sıradan seçenekler böylece öne geçmez). Oran yalnızca kupon kuralları için süzgeçtir
    (1.20 altı elenir, toplam ≥3 aranır); hangi seçeneğin seçileceğini belirlemez.
    Aday: (etiket, olasılık, oran, ayrıntı sözlüğü)."""
    nes, act = _nes_of(r)
    if not act or not nes or not nes.get("markets"):
        return []                      # Nesine verisi yoksa kupon yok; yurtdışı oran kullanılmaz
    if (nes.get("mbs") or 1) > COUPON_LEGS:
        return []
    src = r.get("SRC")
    euro = bool(r.get("EURO"))
    if src in ("cross", "prior") or (src == "market" and not euro):   # botun kendi tahmini olmayan maçlar kupona girmez
        return []
    cx = _ctx_of(p, r.get("Div"))
    bx = _base_of(r.get("Div"))
    if bx is None:
        return []
    valid, gate = ENV["valid"], ENV["gate"]
    legs = []
    for mk in nes["markets"]:
        try:
            mtid = int(mk.get("mtid"))
        except (TypeError, ValueError):
            continue
        if mtid not in pz.SPECS:
            continue
        if src == "market" and mtid not in (1, 3):      # yalnızca piyasa olasılığı olan maçta başka pazara girilmez
            continue
        if valid is not None and not (valid.get(mtid) or {}).get("ok"):
            continue
        if mk.get("mbs") and mk["mbs"] > COUPON_LEGS:
            continue
        line = float(mk.get("line") or 0)
        name, labels, fam = pz.info(mtid, line)
        if (gate is None and fam not in BASIC_FAMILIES) or (gate is not None and fam not in gate):
            continue
        outs = mk.get("outs") or []
        ps = pz.probs(cx, mtid, line)
        if ps is None or [o.get("n") for o in outs] != list(range(1, len(ps) + 1)):
            continue                                   # seçenek sırası bilinenle uyuşmuyorsa pazar atlanır
        bs = pz.probs(bx, mtid, line) or [None] * len(ps)
        for i, o in enumerate(outs):
            od, pr, b, lab = o.get("o"), ps[i], bs[i], labels[i]
            if not od or od < COUPON_MIN_LEG_ODDS or _ALT.search(lab):   # Alt seçimi kuponlara girmez
                continue
            if pr < COUPON_MIN_P or tier(pr, od)[0] != "g" or b is None or b >= 0.97:
                continue
            if src in ("blend", "market") and pr < 0.60:
                continue
            edge = pr - b
            if edge < MIN_EDGE:
                continue
            legs.append((lab, float(pr), float(od), {"mtid": mtid, "line": line, "idx": i, "mk": name, "fam": fam,
                                                     "base": float(b), "edge": float(edge)}))
    legs.sort(key=lambda x: -x[3]["edge"])
    per, out = {}, []
    for l in legs:                                      # bir pazar ailesinden en çok 3 seçenek tutulur
        if per.get(l[3]["fam"], 0) < 3:
            per[l[3]["fam"]] = per.get(l[3]["fam"], 0) + 1
            out.append(l)
    return out[:14]


def build_coupons(out_rows):
    """Günün maçlarından, toplam oranı COUPON_MIN_ODDS ve üzeri olan kuponlar. Her kupon farklı maçlardan oluşur;
    seçimi botun analiz puanı (edge) belirler."""
    today = pd.Timestamp(today_tr())
    by_day = {}
    for k, (r, p, _) in enumerate(out_rows):
        legs = _legs(r, p)
        if legs:
            by_day.setdefault(r["Date"].normalize(), []).append((k, r, legs, p))
    nearest = None
    from collections import Counter as _Cn
    for d_, its in sorted(by_day.items()):
        if d_ >= today:
            print(f"  Kupon adayı maç {d_:%Y-%m-%d}: {len(its)} (kaynak: {dict(_Cn(t[1].get('SRC') for t in its))}, avrupa kupası: {sum(bool(t[1].get('EURO')) for t in its)})")
    got, fam_use = [], {}                 # bugünden başlayarak, 4 kupon dolana kadar sonraki günlerin maçlarına bakılır
    for day in sorted(d for d in by_day if d >= today):
        if len(got) >= N_COUPONS:
            break
        if len(by_day[day]) < COUPON_LEGS:
            continue
        res = _coupons_for_day(day, by_day[day], N_COUPONS - len(got), fam_use)
        print(f"  {day:%Y-%m-%d}: kurulan kupon {len(res['list'])}")
        got += res["list"]
        if not res["list"] and nearest is None and not got:   # çıkmadıysa: ulaşılabilen en yüksek toplam oran
            best = sorted((max(l[2] for l in t[2]) for t in by_day[day]), reverse=True)[:COUPON_LEGS]
            nearest = {"date": day.strftime("%Y-%m-%d"), "list": [], "max_odds": round(math.prod(best), 2)}
    if got:
        print("  Kuponlarda kullanılan pazar aileleri:", dict(sorted(fam_use.items(), key=lambda x: -x[1])))
        return {"date": got[0]["date"], "list": got}
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


EURO_TOL = 0.6          # Avrupa kupası önceliğinde kabul edilen en düşük göreli puan


def _pareto(legs):
    """Aynı maçta hem analiz puanı hem oranı başka bir seçenekten kötü olan seçenekleri eler; en çok 6 seçenek kalır."""
    out, best_e = [], -1.0
    for x in sorted(legs, key=lambda x: -x[2]):          # orana göre azalan
        if x[3]["edge"] > best_e + 1e-12:
            out.append(x)
            best_e = x[3]["edge"]
    if len(out) > 6:
        out = sorted(out, key=lambda x: -x[3]["edge"])[:5] + [max(out, key=lambda x: x[2])]
        out = list({id(x): x for x in out}.values())
    return out


def _best_combo(items, pool_size=45, fam_use=None):
    """Verilen maçlar arasından, toplam oranı COUPON_MIN_ODDS ve üzeri olan, analiz puanı (edge toplamı) en yüksek kupon.
    Aynı pazar ailesi kuponlarda tekrarlandıkça puan düşer (çeşitlilik), böylece her kupon çifte şans olmaz."""
    fam_use = fam_use or {}
    cands = [(t, _pareto(t[2])) for t in items if t[2]]
    euro = lambda t: t[1] is not None and t[1].get("EURO") == True  # noqa: E712
    eu = [c for c in cands if euro(c[0])]
    cands = [c for c in cands if not euro(c[0])]
    cands.sort(key=lambda c: -max(x[3]["edge"] for x in c[1]))
    by_e = cands[:pool_size - 10]
    rest = sorted(cands[pool_size - 10:], key=lambda c: -max(x[2] for x in c[1]))[:10]
    cands = by_e + rest + eu
    best_by = {}                      # Avrupa kupası maçı sayısına göre en iyi kupon
    maxo = [max(x[2] for x in c[1]) for c in cands]
    maxe = [max(x[3]["edge"] for x in c[1]) for c in cands]
    for i, j, k in itertools.combinations(range(len(cands)), 3):
        if maxo[i] * maxo[j] * maxo[k] < COUPON_MIN_ODDS:
            continue
        ne = euro(cands[i][0]) + euro(cands[j][0]) + euro(cands[k][0])
        cur = best_by.get(ne)
        if cur is not None and maxe[i] + maxe[j] + maxe[k] <= cur[0]:
            continue                  # bu üçlü en iyi puanı bile geçemez
        for pick in itertools.product(cands[i][1], cands[j][1], cands[k][1]):
            od = pick[0][2] * pick[1][2] * pick[2][2]
            if od < COUPON_MIN_ODDS:
                continue
            fams = [x[3]["fam"] for x in pick]
            sc = sum(x[3]["edge"] for x in pick) - FAM_PEN * sum(fam_use.get(f, 0) for f in fams) \
                - DUP_PEN * (len(fams) - len(set(fams)))
            cur = best_by.get(ne)
            if cur is None or sc > cur[0]:
                best_by[ne] = (sc, od, (cands[i][0], cands[j][0], cands[k][0]), pick, ne)
    if not best_by:
        return None
    top = max(v[0] for v in best_by.values())
    # Avrupa kupası maçları ön planda: puanı en iyinin %60'ından az düşmüyorsa en çok kupa maçı içeren seçilir
    ok = [v for v in best_by.values() if v[0] >= top - (1 - EURO_TOL) * abs(top)]   # puan eksiye düşebilir (çeşitlilik cezası); en iyi her zaman kalır
    return max(ok, key=lambda v: (v[4], v[0]))


def _pct(x):
    return f"%{round(100 * x)}"


def _why(r, p, meta, lab, pr):
    """Seçimin gerekçesi: yalnızca eldeki verilerden (model beklentisi, son maçlar, ikili maçlar) cümleler. Veri yoksa o
    cümle yazılmaz; hiçbir şey uydurulmaz."""
    home, away = r["HomeTeam"], r["AwayTeam"]
    fam = meta["fam"]
    out = [f"Model beklentisi: {home} {p['xg_h']:.2f}, {away} {p['xg_a']:.2f} gol. Bu seçenek {_pct(pr)} olasılıkla "
           f"gerçekleşiyor; ligin sıradan bir maçında benzer seçenek {_pct(meta['base'])}."]
    cf = ENV.get("ctxf")
    c = cf(home, away) if cf else None
    if c:
        def form(name, f):
            if not f:
                return None
            w, d, l = (sum(1 for x in f if x["r"] == k) for k in "GBM")
            gf = sum(int(x["s"].split("-")[0]) for x in f)
            ga = sum(int(x["s"].split("-")[1]) for x in f)
            return f"{name} son {len(f)} maçta {w}G {d}B {l}M ({gf}-{ga} gol)"
        fh, fa = form(home, c.get("fh")), form(away, c.get("fa"))
        if fh or fa:
            out.append("Form: " + "; ".join(x for x in (fh, fa) if x) + ".")
        sh, sa = c.get("sh"), c.get("sa")
        goal_fams = ("AÜ", "TG", "TAKIM", "MS&AÜ", "İY AÜ", "İY TG", "YARI", "KG", "MS&KG", "İY MS&KG")
        if sh and sa:
            if fam in goal_fams:
                out.append(f"Gol temposu (son {min(sh['n'], sa['n'])}+ maç): {home} maçlarında ortalama {sh['gf'] + sh['ga']:.1f}, "
                           f"{away} maçlarında {sa['gf'] + sa['ga']:.1f} gol; 2.5 üst oranı %{round(sh['ov'] * 100)} / %{round(sa['ov'] * 100)}, "
                           f"iki takım da gol atma oranı %{round(sh['bt'] * 100)} / %{round(sa['bt'] * 100)}.")
            else:
                out.append(f"Hücum/savunma (son {min(sh['n'], sa['n'])}+ maç): {home} {sh['gf']:.1f} atıyor, {sh['ga']:.1f} yiyor; "
                           f"{away} {sa['gf']:.1f} atıyor, {sa['ga']:.1f} yiyor.")
        if c.get("h2h"):
            out.append("Son ikili maçlar: " + ", ".join(f"{x['h']} {x['s']} {x['a']}" for x in c["h2h"][:3]) + ".")
    if fam in ("İY", "İY ÇŞ", "İY AÜ", "İY TG", "İY/MS", "İY MS&KG", "YARI", "2Y", "2Y ÇŞ"):
        sh = pz.shares_for(ENV["shares"], r.get("Div"))
        if sh:
            out.append(f"Yarı hesabı: ligde goller ilk yarıda ev sahibinde %{round(sh[0] * 100)}, deplasmanda %{round(sh[1] * 100)} "
                       f"oranında atılıyor (geçmiş maçların yarı skorlarından).")
    src = r.get("SRC")
    if src == "blend":
        out.append("Avrupa kupası: takımlar farklı liglerde olduğu için olasılığın yarısı modelden, yarısı Nesine'nin kendi "
                   "olasılığından geliyor; bu maçta karar oranlardan tam bağımsız değil.")
    return out


def _coupons_for_day(day, items, limit=None, fam_use=None):
    """Her kupon farklı maçlardan oluşur: bir kupon seçilince o maçlar havuzdan çıkar."""
    fam_use = fam_use if fam_use is not None else {}

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
    while len(chosen) < (limit or N_COUPONS) and len(pool) >= COUPON_LEGS:
        best = _best_combo(pool, fam_use=fam_use)
        if best is None:
            break
        sc, od, combo, pick = best[:4]
        used = {key(c) for c in combo}
        if len(used) != COUPON_LEGS or (used & used_all):
            break                     # tekrar eden maç olamaz; olursa kuponu eklemeden dur
        used_all |= used
        pool = [t for t in pool if key(t) not in used]
        legs, pr = [], 1.0
        for c, x in zip(combo, pick):
            r = c[1]
            t = str(r.get("Time", "") or "")
            m = x[3]
            pr *= x[1]
            fam_use[m["fam"]] = fam_use.get(m["fam"], 0) + 1
            legs.append({"id": c[0], "time": "" if t in ("nan", "NaT") else t[:5],
                         "home": r["HomeTeam"], "away": r["AwayTeam"],
                         "league": r.get("LeagueName") or LEAGUES.get(r["Div"], DEMO_LEAGUES.get(r["Div"], "Lig adı Nesine'de yok" if r["Div"] == "NES" else r["Div"])),
                         "sel": x[0], "mk": m["mk"], "o": round(x[2], 2), "p": round(x[1], 3), "base": round(m["base"], 3),
                         "edge": round(m["edge"], 3), "mtid": m["mtid"], "line": m["line"], "idx": m["idx"], "fam": m["fam"],
                         "t": tier(x[1], x[2])[0], "src": r.get("SRC") or "model", "nid": _nid(r.get("NID")),
                         "mbs": (_nes_of(r)[0] or {}).get("mbs"),
                         "why": _why(r, c[3], m, x[0], x[1]) if len(c) > 3 else []})
        legs.sort(key=lambda l: l["time"])
        chosen.append({"legs": legs, "odds": round(od, 2), "date": day.strftime("%Y-%m-%d"),
                       "p": round(pr, 4), "ev": round(pr * od, 2), "score": round(sc, 3)})
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
    by_day = {}
    for k in coupons["list"]:             # her kupon kendi maç gününün altına kaydedilir
        by_day.setdefault(k.get("date") or coupons["date"], []).append(
            {"odds": k["odds"], "p": k["p"],
             "legs": [{x: l[x] for x in ("time", "home", "away", "league", "sel", "o", "p", "nid", "mk", "mtid", "line", "idx", "fam", "base", "edge", "why")
                              if x in l}
                      for l in k["legs"]]})
    for d_, lst in by_day.items():
        log[d_] = {"list": lst}
    for d in sorted(log)[:-keep_days]:
        del log[d]
    COUPON_LOG.parent.mkdir(parents=True, exist_ok=True)
    COUPON_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


def _nid(v):
    """Nesine maç numarası (pandas'ta NaN olabilir)."""
    try:
        return int(v) if v is not None and not pd.isna(v) else None
    except (TypeError, ValueError):
        return None


def nesine_results():
    """Nesine sonuç akışından ({maç no: (ev, dep, ilk yarı ev, ilk yarı dep)}, {(ev, dep adı): [(tarih, ev, dep)]}).
    Yalnızca 'bitti' (S=5) ve maç skoru (T=1) olan kayıtlar. İlk yarı skoru T=2'dir ve tutarlıysa (her iki takımda da
    maç skorunu aşmıyorsa) alınır; yoksa None. S/T anlamı logdaki canlı örneklerden çıkarılan varsayımdır (S: 2=1. yarı,
    3=devre arası, 4=2. yarı, 5=bitti); doğrulama için nesine.py sonuç örneklerine bakın."""
    try:
        import nesine
        st = nesine.load_results_store()
    except Exception:                                    # noqa: BLE001
        return {}, {}
    byid, byname = {}, {}
    for k, r in (st.get("results") or {}).items():
        t = r.get("t") or {}
        sc = t.get("1")
        m = (st.get("matches") or {}).get(k)
        if r.get("s") not in nesine.FINAL_STATES or not sc or not m:
            continue
        ht = t.get("2")
        if not (ht and ht[0] <= sc[0] and ht[1] <= sc[1]):
            ht = None
        byid[int(k)] = (sc[0], sc[1], None if ht is None else ht[0], None if ht is None else ht[1])
        if m.get("d"):
            byname.setdefault((str(m["h"]).strip().lower(), str(m["a"]).strip().lower()), []).append(
                (pd.Timestamp(m["d"]), sc[0], sc[1]))
    return byid, byname


def _grade_day(day, entry, idx, byid=None):
    """Bir günün kuponlarını gerçek sonuçlarla değerlendirir. Maç ±1 gün toleransla aranır."""
    d0 = pd.Timestamp(day)
    out = []
    for k in entry["list"]:
        legs, bad, pend = [], False, False
        for l in k["legs"]:
            res = None
            nid = l.get("nid")
            if byid and nid is not None and int(nid) in byid:      # Nesine maç numarasıyla kesin eşleşme
                res = tuple(None if x is None else int(x) for x in byid[int(nid)])
            for dt, hg, ag, h1, a1 in ([] if res else idx.get((str(l["home"]).strip().lower(), str(l["away"]).strip().lower()), [])):
                if abs((dt - d0).days) <= 1:
                    res = (int(hg), int(ag), None if h1 is None or pd.isna(h1) else int(h1), None if a1 is None or pd.isna(a1) else int(a1))
                    break
            if res is None:
                ok = None
            elif l.get("mtid") is not None:                       # yeni kayıt: bahis türü ve seçenek sırasıyla
                ok = pz.outcome(int(l["mtid"]), float(l.get("line") or 0), int(l["idx"]), res[0], res[1], res[2], res[3])
            else:                                                 # eski kayıt: etiketle
                ok = _hit(l["sel"], res[0], res[1])
            bad |= ok is False
            pend |= ok is None
            legs.append({**l, "score": None if res is None else f"{res[0]}-{res[1]}", "ok": ok})
        out.append({"odds": k["odds"], "p": k["p"], "legs": legs,
                    "status": "lost" if bad else ("pending" if pend else "won")})
    return out


# football-data sonuçları 1-2 gün geç yayımlar; bekleyen maçlar için ESPN'in herkese açık skor
# listesine bakılır (resmi olmayan bir uç nokta; ulaşılamazsa sessizce atlanır).
ESPN = {"E0": "eng.1", "E1": "eng.2", "E2": "eng.3", "E3": "eng.4", "SC0": "sco.1", "SC1": "sco.2",
        "D1": "ger.1", "D2": "ger.2", "I1": "ita.1", "I2": "ita.2", "SP1": "esp.1", "SP2": "esp.2",
        "F1": "fra.1", "F2": "fra.2", "N1": "ned.1", "B1": "bel.1", "P1": "por.1", "T1": "tur.1",
        "G1": "gre.1", "DNK": "den.1", "SWE": "swe.1", "NOR": "nor.1", "FIN": "fin.1", "AUT": "aut.1",
        "SWZ": "sui.1", "POL": "pol.1", "ROU": "rou.1", "RUS": "rus.1", "IRL": "irl.1", "BRA": "bra.1",
        "ARG": "arg.1", "MEX": "mex.1", "USA": "usa.1", "JPN": "jpn.1", "CHN": "chn.1"}
_NAME2DIV = {v: k for k, v in LEAGUES.items()}


def _http_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


NES_SLUG = [("FRANSA LIGUE 1", "fra.1"), ("FRANSA LIGUE 2", "fra.2"), ("DANIMARKA SUPER", "den.1"), ("NORVEC ELITESERIEN", "nor.1"),
            ("BELCIKA PRO", "bel.1"), ("ARJANTIN PRIMERA", "arg.1"), ("ISPANYA LA LIGA", "esp.1"), ("ISPANYA LALIGA", "esp.1"),
            ("ISPANYA SEGUNDA", "esp.2"), ("INGILTERE PREMIER", "eng.1"), ("INGILTERE CHAMPIONSHIP", "eng.2"),
            ("LEAGUE ONE", "eng.3"), ("LEAGUE TWO", "eng.4"), ("TURKIYE SUPER", "tur.1"), ("ALMANYA BUNDESLIGA", "ger.1"),
            ("ITALYA SERIE A", "ita.1"), ("ITALYA SERIE B", "ita.2"), ("HOLLANDA EREDIVISIE", "ned.1"), ("PORTEKIZ", "por.1"),
            ("RUSYA PREMIER", "rus.1"), ("ISVEC ALLSVENSKAN", "swe.1"), ("ISVICRE SUPER", "sui.1"), ("AVUSTURYA BUNDESLIGA", "aut.1"),
            ("YUNANISTAN SUPER", "gre.1"), ("POLONYA EKSTRAKLASA", "pol.1"), ("ROMANYA LIGA 1", "rou.1"),
            ("SKOCYA PREMIERSHIP", "sco.1"), ("AFC ASYA SAMPIYONLAR", "afc.champions"), ("LIBERTADORES", "conmebol.libertadores"),
            ("SUDAMERIKA KUPASI", "conmebol.sudamericana"), ("MEKSIKA", "mex.1"), ("BREZILYA SERIE A", "bra.1"), ("ABD MLS", "usa.1")]


def _espn_slug(league):
    """Kupon bacağının lig adından ESPN lig kodu (Nesine'nin Türkçe lig adları ve football-data adları)."""
    div = _NAME2DIV.get(league)
    if div in ESPN:
        return ESPN[div]
    n = str(league or "").upper().translate(str.maketrans("İŞÇĞÜÖ", "ISCGUO"))
    if "GENC" in n or "KADIN" in n or "RFEF" in n:
        return None
    if "SAMPIYONLAR LIGI" in n and "AFC" not in n:
        return "uefa.champions"
    if "AVRUPA LIGI" in n:
        return "uefa.europa"
    if "KONFERANS" in n:
        return "uefa.europa.conf"
    for key, slug in NES_SLUG:
        if key in n:
            return slug
    return None


def _espn_events(data):
    """ESPN skor listesinden biten maçlar: [(ev adları, dep adları, ev golü, dep golü, ilk yarı ev, ilk yarı dep, ad metni)]."""
    out = []
    for ev in data.get("events", []):
        try:
            comp = ev["competitions"][0]
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            names = {}
            for side in ("home", "away"):
                t = teams[side]["team"]
                names[side] = [t.get(k) for k in ("displayName", "shortDisplayName", "name", "location") if t.get(k)]
            label = f"{names['home'][0]} - {names['away'][0]}"
            if not bool(comp["status"]["type"].get("completed")):
                out.append((None, None, None, None, None, None, label + " (bitmedi)"))
                continue
            hg, ag = int(teams["home"]["score"]), int(teams["away"]["score"])
            h1 = a1 = None
            try:
                lh, la = teams["home"].get("linescores") or [], teams["away"].get("linescores") or []
                if len(lh) >= 2 and len(la) >= 2:           # periyot gollerinin toplamı maç skoruna eşitse ilk periyot ilk yarıdır
                    x, y = int(float(lh[0]["value"])), int(float(la[0]["value"]))
                    if sum(int(float(v["value"])) for v in lh) == hg and sum(int(float(v["value"])) for v in la) == ag and x <= hg and y <= ag:
                        h1, a1 = x, y
            except (KeyError, ValueError, TypeError):
                pass
            out.append((names["home"], names["away"], hg, ag, h1, a1, label))
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return out


def espn_scores(pending, max_calls=60):
    """pending: [(gün 'YYYY-MM-DD', kupon bacağı)]. Biten maçlar için {(ev, dep): (tarih, ev golü, dep golü, İY ev, İY dep)}.
    Önce ESPN'in o günün TÜM futbol maçlarını veren listesine bakılır (lig eşlemesi gerekmez, iki takım adı da tutmalı);
    olmazsa lig kodu bilinenler için lig listesine. Yalnızca skor alınır; oran ya da fikstür alınmaz."""
    try:
        import nesine
    except ImportError:
        return {}
    out, calls, fails, cache = {}, 0, 0, {}

    def get(url, key):
        nonlocal calls, fails
        if key in cache:
            return cache[key]
        if calls >= max_calls or fails >= 5:
            return []
        calls += 1
        try:
            cache[key] = _espn_events(_http_json(url, timeout=20))
        except Exception as e:  # noqa: BLE001
            fails += 1
            cache[key] = []
            print(f"  ! ESPN okunamadı ({key[0]} {key[1]}): {e}")
        return cache[key]

    def match(day, l, events):
        d0 = pd.Timestamp(day)
        for h, a, hg, ag, h1, a1, label in events:
            if h is None:
                continue
            if any(nesine.same_team(l["home"], n) for n in h) and any(nesine.same_team(l["away"], n) for n in a):
                out[(str(l["home"]).strip().lower(), str(l["away"]).strip().lower())] = (d0, hg, ag, h1, a1)
                return True
        return False

    for day, l in pending:
        d0 = pd.Timestamp(day)
        for dd in (d0, d0 - timedelta(days=1), d0 + timedelta(days=1)):
            ev = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard?dates={dd:%Y%m%d}&limit=1000",
                     ("all", f"{dd:%Y%m%d}"))
            if match(day, l, ev):
                break
        else:
            slug = _espn_slug(l.get("league"))
            if slug:
                for dd in (d0, d0 - timedelta(days=1), d0 + timedelta(days=1)):
                    ev = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={dd:%Y%m%d}",
                             (slug, f"{dd:%Y%m%d}"))
                    if match(day, l, ev):
                        break
    for day, l in pending:
        if (str(l["home"]).strip().lower(), str(l["away"]).strip().lower()) not in out:
            cand = []
            for key, evs in cache.items():                    # takımlardan birine benzeyen ESPN maçları (ad farkını görmek için)
                for h, a, hg, ag, h1, a1, label in evs:
                    nm = (h or []) + (a or [])
                    if any(nesine.same_team(l["home"], n) or nesine.same_team(l["away"], n) for n in nm) and label not in cand:
                        cand.append(f"{label} [{key[0]} {key[1]}]")
            print(f"  ESPN'de sonuç bulunamadı: {l['home']} - {l['away']} ({l.get('league')}, {day}, lig kodu: {_espn_slug(l.get('league'))}). "
                  f"Benzer ESPN maçları: {'; '.join(cand[:5]) or 'yok'}")
    n_all = sum(len(v) for k, v in cache.items() if k[0] == "all")
    print(f"  ESPN: {calls} istek, günlük listelerde toplam {n_all} maç, bulunan sonuç {len(out)} / {len(pending)}")
    return out


BACKUP_FILE = DATA_DIR / "skor_yedek.json"       # ESPN'den bulunan skorlar kalıcı tutulur (tekrar sorulmaz)


def _load_backup():
    try:
        return json.loads(BACKUP_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_backup(d):
    BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    cut = (date.today() - timedelta(days=60)).isoformat()
    d = {k: v for k, v in d.items() if k.split("|")[0] >= cut}
    BACKUP_FILE.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def yesterday_report(hist, today=None, online=False):
    """En son geçmiş kupon gününün sonuçları ve tüm kayıtlı günlerin toplam tutma sayısı."""
    log = _load_log()
    today = today or today_tr().strftime("%Y-%m-%d")
    days = sorted(d for d in log if d < today)
    if not days or hist is None or not len(hist):
        return None
    idx = {}
    hh = hist["HTHG"] if "HTHG" in hist.columns else [None] * len(hist)
    ha = hist["HTAG"] if "HTAG" in hist.columns else [None] * len(hist)
    for dt, h, a, hg, ag, h1, a1 in zip(hist["Date"], hist["HomeTeam"], hist["AwayTeam"], hist["FTHG"], hist["FTAG"], hh, ha):
        if pd.notna(hg) and pd.notna(ag):
            idx.setdefault((str(h).strip().lower(), str(a).strip().lower()), []).append((dt, hg, ag, h1, a1))
    byid, byname = nesine_results()
    for key, v in byname.items():
        idx.setdefault(key, []).extend((dt, hg, ag, None, None) for dt, hg, ag in v)
    backup = _load_backup()
    for k_, v in backup.items():                    # daha önce ESPN'den bulunan skorlar
        day_, h_, a_ = k_.split("|", 2)
        idx.setdefault((h_, a_), []).append((pd.Timestamp(day_), v[0], v[1], v[2], v[3]))
    graded = {d: _grade_day(d, log[d], idx, byid) for d in days}
    if online:
        pend = [(d, l) for d in days[-14:] for k, g in zip(log[d]["list"], graded[d])
                for l, gl in zip(k["legs"], g["legs"]) if gl["ok"] is None]
        if pend:
            try:
                found = espn_scores(pend)
            except Exception as e:  # noqa: BLE001  yedek kaynak sayfayı bozmasın
                print(f"  ! ESPN yedeği atlandı: {e}")
                found = {}
            print(f"Sonucu bekleyen {len(pend)} kupon maçından ESPN'de bulunan: {len(found)}")
            for (h_, a_), (d0, hg, ag, h1, a1) in found.items():
                idx.setdefault((h_, a_), []).append((d0, hg, ag, h1, a1))
                backup[f"{d0:%Y-%m-%d}|{h_}|{a_}"] = [hg, ag, h1, a1]
            if found:
                try:
                    _save_backup(backup)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! skor yedeği yazılamadı: {e}")
            graded = {d: _grade_day(d, log[d], idx, byid) for d in days}
    try:                                  # sonucu gelmeyen bacakların nedeni (log için)
        import nesine
        st = nesine.load_results_store()
        for d in days[-2:]:
            for k, g in zip(log[d]["list"], graded[d]):
                for l, gl in zip(k["legs"], g["legs"]):
                    if gl["ok"] is None:
                        nid = l.get("nid")
                        rec = (st.get("results") or {}).get(str(nid)) if nid is not None else None
                        why = ("kupon kaydında maç no yok" if nid is None else
                               "Nesine kaydında maç yok" if str(nid) not in (st.get("matches") or {}) else
                               "akışta hiç görülmedi" if rec is None else
                               f"akışta son durum S={rec.get('s')} skor={(rec.get('t') or {}).get('1')}")
                        if l.get("mtid") in (None,) or not pz.SPECS.get(int(l["mtid"]), {}).get("half"):
                            pass
                        print(f"  SONUÇ-BEKLEYEN {d} {l['home']} - {l['away']} [{l['sel']}] nid={nid}: {why}")
    except Exception as e:  # noqa: BLE001
        print("  (bekleyen sonuç nedenleri yazılamadı:", e, ")")
    tot = {"won": 0, "lost": 0, "pending": 0, "days": len(days)}
    for d in days:
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


STD_MARKETS = [(1, 0), (3, 0), (38, 0), (11, 1.5), (12, 2.5), (13, 3.5), (155, 4.5), (14, 1.5), (209, 0.5), (20, 1.5), (29, 1.5),
               (43, 0), (297, 0), (268, -1), (268, 1), (268, -2), (268, 2), (342, 1.5), (343, 2.5), (272, 3.5), (272, 4.5),
               (414, 0), (7, 0), (8, 0), (9, 0), (10, 0), (5, 0), (48, 0), (332, 0), (416, 0), (295, 0), (296, 0), (586, 0),
               (587, 0), (584, 0), (585, 0), (591, 0), (592, 0), (589, 0), (590, 0)]
FAM_MIN_N = 40         # bir pazar ailesi kuponlara girebilmek için geçmişte en az bu kadar yeşil seçenek üretmiş olmalı
FAM_TOL = 0.05         # ... ve gerçekleşme oranı, modelin verdiği ortalama olasılığın en fazla bu kadar altında kalmalı


def family_calibration(recs):
    """Yürüyen (geleceği görmeyen) geçmiş tahminlerde her pazar ailesinin yeşil (≥%55) seçeneklerinin gerçekte ne kadar
    tuttuğu. Model bir aileyi aşırı güvenle öngörüyorsa (tutma oranı verdiği olasılığın FAM_TOL'dan fazla altında) o aile
    kuponlara girmez; yeterince veri yoksa da girmez."""
    acc = {}
    for x in recs:
        r = x["r"]
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        h1 = r["HTHG"] if "HTHG" in r.index and pd.notna(r["HTHG"]) else None
        a1 = r["HTAG"] if "HTAG" in r.index and pd.notna(r["HTAG"]) else None
        if h1 is not None and a1 is not None and (h1 > hg or a1 > ag):
            h1 = a1 = None
        cx = pz.Ctx(x["p"], x.get("sh"))
        for mtid, line in STD_MARKETS:
            ps = pz.probs(cx, mtid, line)
            if ps is None:
                continue
            fam = pz.info(mtid, line)[2]
            for i, pi in enumerate(ps):
                if pi < GREEN_P:
                    continue
                ok = pz.outcome(mtid, line, i, hg, ag, None if h1 is None else int(h1), None if a1 is None else int(a1))
                if ok is None:
                    continue
                a = acc.setdefault(fam, [0, 0.0, 0])
                a[0] += 1
                a[1] += pi
                a[2] += bool(ok)
    out = []
    for fam, (n, sp, h) in sorted(acc.items()):
        pm, hit = sp / n, h / n
        out.append({"fam": fam, "n": n, "p": round(pm, 3), "hit": round(hit, 3),
                    "ok": bool(n >= FAM_MIN_N and hit >= pm - FAM_TOL)})
    return out


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
            sh = pz.shares_for(pz.half_shares(train, min_n=150), None)
            for _, r in test.iterrows():
                if model.knows(r["HomeTeam"], r["AwayTeam"]):
                    recs.append({"r": r, "base": base, "sh": sh,
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
    try:
        perf["fam"] = family_calibration(recs)
    except Exception as e:  # noqa: BLE001  pazar ailesi ölçümü sayfayı bozmasın
        print(f"  ! pazar ailesi ölçümü atlandı: {e}")
        perf["fam"] = None

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


def _valid_ok(mtid):
    v = ENV.get("valid")
    return v is None or bool((v.get(mtid) or {}).get("ok"))


def nes_markets(nes, p, src=None, div=None):
    """Nesine'de bu maç için açık olan bütün pazarlar (sitede listelenir). Modelin olasılığı yalnızca pazarlar.py'de
    adı/seçenek sırası Nesine tablosundan doğrulanan ve çalıştırma sırasında oranlara karşı sınavı geçen pazarlarda gösterilir."""
    if not nes or not nes.get("markets"):
        return None
    out = []
    cx = _ctx_of(p, div)
    for mk in nes["markets"]:
        ps = None
        name, labels = mk.get("name"), mk.get("labels")
        try:
            mtid = int(mk.get("mtid"))
        except (TypeError, ValueError):
            mtid = None
        if mtid in pz.SPECS and src not in ("market", "prior") and _valid_ok(mtid):
            line = float(mk.get("line") or 0)
            ps = pz.probs(cx, mtid, line)
            outs = mk.get("outs") or []
            if ps is None or [o.get("n") for o in outs] != list(range(1, len(ps) + 1)):
                ps = None
            else:
                pn, pl, _ = pz.info(mtid, line)
                name = name or pn
                labels = labels or pl
        out.append({"name": name, "mtid": mk.get("mtid"), "line": mk.get("line", 0), "mbs": mk.get("mbs"),
                    "main": mk.get("main"), "labels": labels, "outs": mk.get("outs"),
                    "ps": [round(float(x), 4) for x in ps] if ps else None})
    out.sort(key=lambda k: (0 if k["ps"] else 1, k["mtid"] or 0, k["line"] or 0))
    return out


def match_context(hist):
    """Maç panelinde gösterilen bilgi: son 5 maç formu ve aralarındaki son karşılaşmalar.
    Olasılıkları değiştirmez (form zaten modelde zaman ağırlığıyla yer alıyor)."""
    if hist is None or not len(hist):
        return lambda h, a: None
    h2 = hist.sort_values("Date")
    team, pair = {}, {}
    for dt, h, a, hg, ag in zip(h2["Date"], h2["HomeTeam"], h2["AwayTeam"], h2["FTHG"], h2["FTAG"]):
        hg, ag = int(hg), int(ag)
        team.setdefault(h, []).append((dt, a, hg, ag, True))
        team.setdefault(a, []).append((dt, h, ag, hg, False))
        pair.setdefault(frozenset((h, a)), []).append((dt, h, a, hg, ag))

    def ctx(h, a):
        def form(t):
            return [{"d": dt.strftime("%d.%m.%y"), "o": o, "s": f"{gf}-{ga}", "ev": ev,
                     "r": "G" if gf > ga else ("B" if gf == ga else "M")}
                    for dt, o, gf, ga, ev in team.get(t, [])[-5:]][::-1]
        def style(t):
            x = team.get(t, [])[-10:]
            if len(x) < 5:
                return None
            gf = sum(e[2] for e in x) / len(x)
            ga = sum(e[3] for e in x) / len(x)
            ov = sum(e[2] + e[3] >= 3 for e in x) / len(x)
            bt = sum(e[2] > 0 and e[3] > 0 for e in x) / len(x)
            tot = gf + ga
            lab = "bol gollü" if tot >= 3.1 else ("kapalı, az gollü" if tot <= 2.2 else "dengeli")
            return {"gf": round(gf, 2), "ga": round(ga, 2), "ov": round(ov, 2), "bt": round(bt, 2), "lab": lab, "n": len(x)}
        fh, fa = form(h), form(a)
        m = [{"d": dt.strftime("%d.%m.%y"), "h": x, "a": y, "s": f"{hg}-{ag}"}
             for dt, x, y, hg, ag in pair.get(frozenset((h, a)), [])[-5:]][::-1]
        if not fh and not fa and not m:
            return None
        return {"fh": fh, "fa": fa, "h2h": m, "sh": style(h), "sa": style(a)}
    return ctx


def site_payload(out_rows, demo=False, perf=None, nes_info=None, yday=None, hist=None):
    ctxf = match_context(hist)
    matches = []
    for k, (r, p, _) in enumerate(out_rows):
        t = str(r.get("Time", "") or "")
        nes = _nes_of(r)[0]
        matches.append({
            "id": k, "div": r["Div"],
            "league": (r.get("LeagueName") or LEAGUES.get(r["Div"], DEMO_LEAGUES.get(r["Div"], "Lig adı Nesine'de yok" if r["Div"] == "NES" else r["Div"]))),
            "src": r.get("SRC") or "model", "euro": bool(r.get("EURO")),
            "date": r["Date"].strftime("%Y-%m-%d"),
            "time": "" if t in ("nan", "NaT") else t[:5],
            "home": r["HomeTeam"], "away": r["AwayTeam"],
            "xg": [round(p["xg_h"], 2), round(p["xg_a"], 2)],
            "nes": bool(nes), "mbs": (nes or {}).get("mbs"),
            "ctx": ctxf(r["HomeTeam"], r["AwayTeam"]),
            "nm": nes_markets(nes, p, r.get("SRC"), r.get("Div")),
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


def setup_env(rows, hist, perf):
    """Kupon seçimi ve pazar paneli için bağlamı kurar: yarı gol payları, lig ortalamaları, form bilgisi, pazar eşlemesinin
    doğrulaması ve pazar ailesi kapısı. Döner: doğrulama sonucu (sayfada gösterilir)."""
    ENV.update({"shares": pz.half_shares(hist), "avg": _league_avgs(hist), "ctxf": match_context(hist),
                "cx": {}, "bx": {}, "valid": None, "gate": None})
    fam = (perf or {}).get("fam")
    if fam:
        ENV["gate"] = {f["fam"] for f in fam if f["ok"]}
        print("  Pazar ailesi kapısı (geçmişte yeşil seçeneklerin tutma oranı):",
              "; ".join(f"{f['fam']} n={f['n']} p={f['p']:.2f} tuttu={f['hit']:.2f} {'✓' if f['ok'] else '✗'}" for f in fam))
    else:
        print("  Pazar ailesi ölçümü yok; kuponlarda yalnızca temel pazarlar:", ", ".join(BASIC_FAMILIES))
    samples = []
    for r, p, _ in rows:
        nes, act = _nes_of(r)
        if act and nes and nes.get("markets") and r.get("SRC") in (None, "model"):   # gerçek veride model satırlarının SRC alanı boş olabilir
            samples.append((_ctx_of(p, r.get("Div")), nes["markets"]))
    if not samples:
        return None
    v = pz.validate(samples)
    ENV["valid"] = v
    good = sorted(k for k, x in v.items() if x["ok"])
    bad = {k: x for k, x in v.items() if not x["ok"]}
    print(f"  Pazar doğrulaması ({len(samples)} model maçı): geçen {len(good)} bahis türü {good}")
    if bad:
        print("  Doğrulanamayan (kuponlara girmez):", "; ".join(f"{k} r={x['r']} n={x['n']} fark={x['mad']}" for k, x in sorted(bad.items())))
    return v


def cmd_site(leagues, days, out, open_it, demo=False):
    if demo:
        rows, skipped = demo_rows(), 0
    else:
        rows, skipped = collect(leagues, days)
    if not rows:
        sys.exit("Nesine maç listesi yok ya da 36 saatten eski (ya da maçlar aralığın dışında). "
                 "Site güncellenmedi; önceki başarılı sürüm yayında kalır. Önce:  python nesine.py fetch")
    perf = None
    hist_all = demo_history() if demo else load_results()
    try:
        print("Geçmiş başarı hesaplanıyor...")
        perf = performance(hist_all)
        if perf is None:
            print("  (başarı hesabı için yeterli geçmiş maç yok)")
    except Exception as e:  # noqa: BLE001  başarı hesabı sayfayı bozmasın
        print(f"  ! başarı hesabı atlandı: {e}")
    nes_info = None
    if not demo:
        rows, nes_info = attach_nesine(rows)           # hata olursa site güncellenmez; yurtdışı oranına dönülmez
        if nes_info is None:
            sys.exit("Nesine verisi kullanılamadı; site güncellenmedi (yurtdışı oranlarına dönülmüyor).")
    valid = None
    try:
        valid = setup_env(rows, hist_all, perf)
    except Exception as e:  # noqa: BLE001
        print(f"  ! pazar bağlamı kurulamadı, kuponlar yalnızca temel pazarlarla: {e}")
    yday = None
    if not demo:
        try:
            yday = yesterday_report(hist_all, online=os.environ.get("GOLRADAR_ESPN") != "0")
        except Exception as e:  # noqa: BLE001  dünkü kuponlar sayfayı bozmasın
            print(f"  ! dünkü kupon sonuçları atlandı: {e}")
    payload = site_payload(rows, demo, perf, nes_info, yday, None if demo else hist_all)
    payload["pazar_dogrulama"] = None if not valid else {str(k): x for k, x in valid.items()}
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
