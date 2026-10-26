"""Nesine'nin açtığı bahis türleri için modelin olasılıkları.

Her bahis türü (MTID) için seçenek sırası Nesine'nin kendi bahis tablosundan (CCAll.min.js / Bulletin.min.js,
log satırları "TANIM <no> -> ad | {seçenek no: ad}") alınmıştır. Burada ne seçeneğin ne olduğu tek bir yerde,
bir koşul (yüklem) olarak tanımlanır; aynı koşul hem olasılığı hesaplamada hem de oynanmış maçın tutup
tutmadığını bulmada kullanılır, ikisi birbirinden ayrılamaz.

Yalnızca adı ve seçenek sırası Nesine tablosunda olan bahisler vardır; tabloda olmayan bahis için olasılık uydurulmaz.
Her çalıştırmada ayrıca `validate` ile tabloyla eşleme, modelin kendi maçlarında Nesine oranlarına karşı sınanır;
sınavı geçemeyen bahis türü kuponlara girmez (oran yalnızca bu eşlemeyi doğrulamak için kullanılır, seçimi etkilemez).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

K = 8                       # her yarıda en çok 7 gol hesaplanır (kuyruk ihmal edilebilir)
_L = np.logical_and


def fmt(x):
    return f"{x:g}".replace(".", ",")


# ---------------------------------------------------------------- koşullar
# Her koşul (h1, a1, h2, a2) alır: ev/deplasman 1. ve 2. yarı golleri. Tam maç golleri H=h1+h2, A=a1+a2.
def _ms3(sel):
    return {"1": lambda h1, a1, h2, a2: (h1 + h2) > (a1 + a2),
            "X": lambda h1, a1, h2, a2: (h1 + h2) == (a1 + a2),
            "2": lambda h1, a1, h2, a2: (h1 + h2) < (a1 + a2)}[sel]


def _iy3(sel):
    return {"1": lambda h1, a1, h2, a2: h1 > a1, "X": lambda h1, a1, h2, a2: h1 == a1,
            "2": lambda h1, a1, h2, a2: h1 < a1}[sel]


def _y2_3(sel):
    return {"1": lambda h1, a1, h2, a2: h2 > a2, "X": lambda h1, a1, h2, a2: h2 == a2,
            "2": lambda h1, a1, h2, a2: h2 < a2}[sel]


def _or(f, g):
    return lambda *a: np.logical_or(f(*a), g(*a))


def _and(f, g):
    return lambda *a: _L(f(*a), g(*a))


def _not(f):
    return lambda *a: np.logical_not(f(*a))


def _tot(h1, a1, h2, a2):
    return h1 + h2 + a1 + a2


def _kg(h1, a1, h2, a2):
    return _L((h1 + h2) > 0, (a1 + a2) > 0)


def _under(line):
    return lambda h1, a1, h2, a2: _tot(h1, a1, h2, a2) < line


def _over(line):
    return lambda h1, a1, h2, a2: _tot(h1, a1, h2, a2) > line


SPECS: dict[int, dict] = {}


def _reg(mtid, fam, half, name, labels, preds):
    SPECS[mtid] = {"fam": fam, "half": half, "name": name, "labels": labels, "preds": preds}


_reg(1, "MS", False, lambda L: "Maç Sonucu", lambda L: ["MS 1", "MS X", "MS 2"],
     lambda L: [_ms3("1"), _ms3("X"), _ms3("2")])
_reg(3, "ÇŞ", False, lambda L: "Çifte Şans", lambda L: ["ÇŞ 1-X", "ÇŞ 1-2", "ÇŞ X-2"],
     lambda L: [lambda h1, a1, h2, a2: (h1 + h2) >= (a1 + a2), lambda h1, a1, h2, a2: (h1 + h2) != (a1 + a2),
                lambda h1, a1, h2, a2: (h1 + h2) <= (a1 + a2)])
_reg(38, "KG", False, lambda L: "Karşılıklı Gol", lambda L: ["KG Var", "KG Yok"],
     lambda L: [_kg, _not(_kg)])
for _m in (11, 12, 13, 155):
    _reg(_m, "AÜ", False, lambda L: f"Alt/Üst {fmt(L)}", lambda L: [f"{fmt(L)} Alt", f"{fmt(L)} Üst"],
         lambda L: [_under(L), _over(L)])
for _m in (14, 209):
    _reg(_m, "İY AÜ", True, lambda L: f"1. Yarı Alt/Üst {fmt(L)}", lambda L: [f"İY {fmt(L)} Alt", f"İY {fmt(L)} Üst"],
         lambda L: [lambda h1, a1, h2, a2: (h1 + a1) < L, lambda h1, a1, h2, a2: (h1 + a1) > L])
_reg(20, "TAKIM", False, lambda L: f"Ev Sahibi {fmt(L)} Gol Alt/Üst", lambda L: [f"Ev {fmt(L)} Alt", f"Ev {fmt(L)} Üst"],
     lambda L: [lambda h1, a1, h2, a2: (h1 + h2) < L, lambda h1, a1, h2, a2: (h1 + h2) > L])
_reg(29, "TAKIM", False, lambda L: f"Deplasman {fmt(L)} Gol Alt/Üst", lambda L: [f"Dep {fmt(L)} Alt", f"Dep {fmt(L)} Üst"],
     lambda L: [lambda h1, a1, h2, a2: (a1 + a2) < L, lambda h1, a1, h2, a2: (a1 + a2) > L])
_reg(43, "TG", False, lambda L: "Toplam Gol Aralığı", lambda L: ["0-1 Gol", "2-3 Gol", "4-5 Gol", "6+ Gol"],
     lambda L: [lambda *a: _tot(*a) <= 1, lambda *a: _L(_tot(*a) >= 2, _tot(*a) <= 3),
                lambda *a: _L(_tot(*a) >= 4, _tot(*a) <= 5), lambda *a: _tot(*a) >= 6])
_reg(297, "TG", False, lambda L: "Toplam Gol Sayısı", lambda L: ["0 Gol", "1 Gol", "2 Gol", "3 Gol", "4 Gol", "5 Gol", "6+ Gol"],
     lambda L: [(lambda n: (lambda *a: _tot(*a) == n))(n) for n in range(6)] + [lambda *a: _tot(*a) >= 6])


def _hms(L):
    adj = lambda h1, a1, h2, a2: (h1 + h2) - (a1 + a2) + L          # çizgi ev sahibine eklenir (−1: ev sahibi 1 gol geride başlar)
    return [lambda *a: adj(*a) > 0, lambda *a: adj(*a) == 0, lambda *a: adj(*a) < 0]


_reg(268, "HMS", False, lambda L: f"Handikaplı MS (ev {L:+g})", lambda L: [f"HMS {L:+g} · 1", f"HMS {L:+g} · X", f"HMS {L:+g} · 2"], _hms)


def _mso(L):
    out = []
    for ou in (_under(L), _over(L)):
        for s in ("1", "X", "2"):
            out.append(_and(_ms3(s), ou))
    return out


for _m in (342, 343, 272):
    _reg(_m, "MS&AÜ", False, lambda L: f"Maç Sonucu & {fmt(L)} Alt/Üst",
         lambda L: [f"MS1 & {fmt(L)} Alt", f"MSX & {fmt(L)} Alt", f"MS2 & {fmt(L)} Alt",
                    f"MS1 & {fmt(L)} Üst", f"MSX & {fmt(L)} Üst", f"MS2 & {fmt(L)} Üst"], _mso)
_reg(414, "MS&KG", False, lambda L: "Maç Sonucu & Karşılıklı Gol",
     lambda L: ["MS1 & KG Var", "MS1 & KG Yok", "MSX & KG Var", "MSX & KG Yok", "MS2 & KG Var", "MS2 & KG Yok"],
     lambda L: [f(s) for s in ("1", "X", "2") for f in ((lambda s: _and(_ms3(s), _kg)), (lambda s: _and(_ms3(s), _not(_kg))))])
_reg(7, "İY", True, lambda L: "1. Yarı Sonucu", lambda L: ["İY 1", "İY X", "İY 2"], lambda L: [_iy3("1"), _iy3("X"), _iy3("2")])
_reg(8, "İY ÇŞ", True, lambda L: "1. Yarı Çifte Şans", lambda L: ["İY 1-X", "İY 1-2", "İY X-2"],
     lambda L: [_or(_iy3("1"), _iy3("X")), _or(_iy3("1"), _iy3("2")), _or(_iy3("X"), _iy3("2"))])
_reg(9, "2Y", True, lambda L: "2. Yarı Sonucu", lambda L: ["2Y 1", "2Y X", "2Y 2"], lambda L: [_y2_3("1"), _y2_3("X"), _y2_3("2")])
_reg(10, "2Y ÇŞ", True, lambda L: "2. Yarı Çifte Şans", lambda L: ["2Y 1-X", "2Y 1-2", "2Y X-2"],
     lambda L: [_or(_y2_3("1"), _y2_3("X")), _or(_y2_3("1"), _y2_3("2")), _or(_y2_3("X"), _y2_3("2"))])
_reg(5, "İY/MS", True, lambda L: "İlk Yarı / Maç Sonucu", lambda L: [f"İY/MS {a}/{b}" for a in "1X2" for b in "1X2"],
     lambda L: [_and(_iy3(a), _ms3(b)) for a in "1X2" for b in "1X2"])
_reg(48, "YARI", True, lambda L: "En Çok Gol Olacak Yarı", lambda L: ["En çok gol 1. yarı", "Yarılar eşit", "En çok gol 2. yarı"],
     lambda L: [lambda h1, a1, h2, a2: (h1 + a1) > (h2 + a2), lambda h1, a1, h2, a2: (h1 + a1) == (h2 + a2),
                lambda h1, a1, h2, a2: (h1 + a1) < (h2 + a2)])
_reg(332, "İY TG", True, lambda L: "1. Yarı Gol Sayısı", lambda L: ["İY 0 gol", "İY 1 gol", "İY 2 gol", "İY 3 gol", "İY 4+ gol"],
     lambda L: [(lambda n: (lambda h1, a1, h2, a2: (h1 + a1) == n))(n) for n in range(4)] + [lambda h1, a1, h2, a2: (h1 + a1) >= 4])
_reg(416, "İY MS&KG", True, lambda L: "1. Yarı Sonucu & 1. Yarı KG",
     lambda L: ["İY 1 & KG Var", "İY 1 & KG Yok", "İY X & KG Var", "İY X & KG Yok", "İY 2 & KG Var", "İY 2 & KG Yok"],
     lambda L: [f(s) for s in ("1", "X", "2") for f in (
         (lambda s: _and(_iy3(s), lambda h1, a1, h2, a2: _L(h1 > 0, a1 > 0))),
         (lambda s: _and(_iy3(s), lambda h1, a1, h2, a2: np.logical_not(_L(h1 > 0, a1 > 0)))))])
_reg(295, "YARI", True, lambda L: "Ev Sahibi İki Yarıda da Gol", lambda L: ["Ev iki yarıda da gol atar", "Ev iki yarıda da atamaz"],
     lambda L: [lambda h1, a1, h2, a2: _L(h1 > 0, h2 > 0), lambda h1, a1, h2, a2: np.logical_not(_L(h1 > 0, h2 > 0))])
_reg(296, "YARI", True, lambda L: "Deplasman İki Yarıda da Gol", lambda L: ["Dep iki yarıda da gol atar", "Dep iki yarıda da atamaz"],
     lambda L: [lambda h1, a1, h2, a2: _L(a1 > 0, a2 > 0), lambda h1, a1, h2, a2: np.logical_not(_L(a1 > 0, a2 > 0))])
_reg(586, "YARI", True, lambda L: "Ev Sahibi Hangi Yarıda Daha Çok Gol Atar", lambda L: ["Ev: 1. yarıda daha çok", "Ev: yarılar eşit", "Ev: 2. yarıda daha çok"],
     lambda L: [lambda h1, a1, h2, a2: h1 > h2, lambda h1, a1, h2, a2: h1 == h2, lambda h1, a1, h2, a2: h1 < h2])
_reg(587, "YARI", True, lambda L: "Deplasman Hangi Yarıda Daha Çok Gol Atar", lambda L: ["Dep: 1. yarıda daha çok", "Dep: yarılar eşit", "Dep: 2. yarıda daha çok"],
     lambda L: [lambda h1, a1, h2, a2: a1 > a2, lambda h1, a1, h2, a2: a1 == a2, lambda h1, a1, h2, a2: a1 < a2])


def _wins_any(home):
    f = (lambda h1, a1, h2, a2: np.logical_or(h1 > a1, h2 > a2)) if home else (lambda h1, a1, h2, a2: np.logical_or(a1 > h1, a2 > h2))
    return [f, _not(f)]


def _wins_both(home):
    f = (lambda h1, a1, h2, a2: _L(h1 > a1, h2 > a2)) if home else (lambda h1, a1, h2, a2: _L(a1 > h1, a2 > h2))
    return [f, _not(f)]


_reg(584, "YARI", True, lambda L: "Ev Sahibi Herhangi Bir Yarıyı Kazanır", lambda L: ["Ev bir yarıyı kazanır", "Ev hiçbir yarıyı kazanamaz"], lambda L: _wins_any(True))
_reg(585, "YARI", True, lambda L: "Deplasman Herhangi Bir Yarıyı Kazanır", lambda L: ["Dep bir yarıyı kazanır", "Dep hiçbir yarıyı kazanamaz"], lambda L: _wins_any(False))
_reg(591, "YARI", True, lambda L: "Ev Sahibi Her İki Yarıyı da Kazanır", lambda L: ["Ev iki yarıyı da kazanır", "Ev iki yarıyı da kazanamaz"], lambda L: _wins_both(True))
_reg(592, "YARI", True, lambda L: "Deplasman Her İki Yarıyı da Kazanır", lambda L: ["Dep iki yarıyı da kazanır", "Dep iki yarıyı da kazanamaz"], lambda L: _wins_both(False))
_reg(589, "MS", False, lambda L: "Ev Sahibi Gol Yemeden Kazanır", lambda L: ["Ev gol yemeden kazanır", "Ev gol yemeden kazanamaz"],
     lambda L: [lambda h1, a1, h2, a2: _L((h1 + h2) > (a1 + a2), (a1 + a2) == 0),
                lambda h1, a1, h2, a2: np.logical_not(_L((h1 + h2) > (a1 + a2), (a1 + a2) == 0))])
_reg(590, "MS", False, lambda L: "Deplasman Gol Yemeden Kazanır", lambda L: ["Dep gol yemeden kazanır", "Dep gol yemeden kazanamaz"],
     lambda L: [lambda h1, a1, h2, a2: _L((a1 + a2) > (h1 + h2), (h1 + h2) == 0),
                lambda h1, a1, h2, a2: np.logical_not(_L((a1 + a2) > (h1 + h2), (h1 + h2) == 0))])


# ---------------------------------------------------------------- olasılık
class Ctx:
    """Bir maçın olasılık hesabı için gereken her şey: tam maç skor matrisi (Dixon-Coles) ve, ilk yarı pazarları için,
    yarılara bölünmüş Poisson. shares=(ev, dep) ilk yarıda atılan gol payı; verilmediyse yarı pazarları hesaplanmaz."""

    def __init__(self, p, shares=None):
        self.m = p["matrix"]
        self.lh, self.la = float(p["xg_h"]), float(p["xg_a"])
        self.sh, self.sa = shares if shares else (None, None)
        self._ft = np.indices(self.m.shape)
        self._half = None

    @property
    def half_ok(self):
        return self.sh is not None

    def half(self):
        if self._half is None:
            g = np.arange(K)
            P1 = np.outer(poisson.pmf(g, self.lh * self.sh), poisson.pmf(g, self.la * self.sa))
            P2 = np.outer(poisson.pmf(g, self.lh * (1 - self.sh)), poisson.pmf(g, self.la * (1 - self.sa)))
            J = P1[:, :, None, None] * P2[None, None, :, :]
            self._half = (J / J.sum(), np.indices((K, K, K, K)))
        return self._half


def probs(ctx: Ctx, mtid, line=0.0):
    """Seçeneklerin (Nesine sırasıyla) model olasılıkları, ya da bu pazar tanımlı/hesaplanabilir değilse None."""
    sp = SPECS.get(mtid)
    if sp is None:
        return None
    if sp["half"]:
        if not ctx.half_ok:
            return None
        J, (h1, a1, h2, a2) = ctx.half()
        return [float(J[np.asarray(f(h1, a1, h2, a2), bool)].sum()) for f in sp["preds"](line)]
    i, j = ctx._ft
    z = np.zeros_like(i)
    return [float(ctx.m[np.asarray(f(i, j, z, z), bool)].sum()) for f in sp["preds"](line)]


def info(mtid, line=0.0):
    sp = SPECS.get(mtid)
    if sp is None:
        return None
    return sp["name"](line), sp["labels"](line), sp["fam"]


def outcome(mtid, line, idx, hg, ag, h1=None, a1=None):
    """Oynanmış maçta seçenek (0'dan başlayan sıra) tuttu mu: True/False; yarı skoru gerekip yoksa None."""
    sp = SPECS.get(mtid)
    if sp is None:
        return None
    fs = sp["preds"](line)
    if not 0 <= idx < len(fs):
        return None
    if sp["half"]:
        if h1 is None or a1 is None:
            return None
        return bool(fs[idx](h1, a1, hg - h1, ag - a1))
    return bool(fs[idx](hg, ag, 0, 0))


# ---------------------------------------------------------------- yarı gol payı
def half_shares(hist, min_n=300):
    """İlk yarıda atılan gol payı (ev, deplasman), football-data'daki HTHG/HTAG sütunlarından: lig başına, lig
    yetersizse tüm ligler. Veri yoksa None (o zaman ilk yarı pazarları hesaplanmaz, pay uydurulmaz)."""
    out = {}
    if hist is None or not len(hist) or "HTHG" not in hist.columns or "HTAG" not in hist.columns:
        return out
    d = hist.dropna(subset=["HTHG", "HTAG", "FTHG", "FTAG"])
    d = d[(d["HTHG"] <= d["FTHG"]) & (d["HTAG"] <= d["FTAG"])]

    def sh(x):
        if len(x) < min_n or x["FTHG"].sum() == 0 or x["FTAG"].sum() == 0:
            return None
        return float(x["HTHG"].sum() / x["FTHG"].sum()), float(x["HTAG"].sum() / x["FTAG"].sum())

    g = sh(d)
    if g:
        out[None] = g
    for div, x in d.groupby("Div"):
        s = sh(x)
        if s:
            out[div] = s
    return out


def shares_for(table, div):
    return table.get(div) or table.get(None)


# ---------------------------------------------------------------- doğrulama
def validate(samples, min_pairs=24, min_r=0.80, max_mad=0.10):
    """samples: [(Ctx, Nesine pazar listesi)] (yalnızca modelin kendi tahmini olan maçlar). Her pazar türü için model
    olasılığı ile Nesine'nin bahisçi payından arındırılmış olasılığı karşılaştırılır; eşleme yanlışsa (seçenek sırası ya da
    anlam) korelasyon düşer. Dönen: {mtid: {"ok": bool, "r": korelasyon, "n": çift sayısı, "mad": ortalama fark}}."""
    acc: dict = {}
    for ctx, mks in samples:
        for mk in mks or []:
            mtid = mk.get("mtid")
            if mtid not in SPECS:
                continue
            odds = [o.get("o") for o in mk.get("outs") or []]
            if not odds or not all(odds):
                continue
            try:
                pm = probs(ctx, mtid, float(mk.get("line") or 0))
            except Exception:                                   # noqa: BLE001
                pm = None
            if pm is None or len(pm) != len(odds):
                continue
            inv = 1 / np.array(odds, float)
            tot = sum(pm)
            if tot < 0.97:                      # hesaplanan seçenekler pazarın tamamını kapsamıyor
                continue
            q = inv / inv.sum() * tot           # çifte şans gibi örtüşen pazarlarda toplam 1 değil 2'dir
            a = acc.setdefault(mtid, ([], []))
            a[0].extend(pm)
            a[1].extend(q.tolist())
    res = {}
    for mtid, (x, y) in acc.items():
        x, y = np.array(x), np.array(y)
        n = len(x)
        r = float(np.corrcoef(x, y)[0, 1]) if n > 2 and x.std() > 0 and y.std() > 0 else 0.0
        mad = float(np.abs(x - y).mean())
        res[mtid] = {"ok": bool(n >= min_pairs and r >= min_r and mad <= max_mad), "r": round(r, 3), "n": n, "mad": round(mad, 3)}
    return res
