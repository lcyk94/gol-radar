#!/usr/bin/env python3
"""
Nesine iddaa bülteni okuyucu
============================
fetch            Nesine sayfasını başsız tarayıcıyla açar, oranları data/nesine.json'a yazar.
parse dosya.txt  Kaydedilmiş sayfa metnini ayrıştırır (deneme için).

Sayfa her satırı ayrı gösteriyor. Bir lig bloğu şöyle başlar:
    LİG ADI / tarih etiketi / "1 X 2 Alt Üst HND 1 X 2 1-X 1-2 X-2 Var Yok +"
ardından her maç için:
    saat / "Ev - Deplasman" / MBS / 15 değer (oran ya da "-") / ek market sayısı
Engel (captcha, 403) çıkarsa durur, engeli aşmaya çalışmaz.
"""
import difflib
import json
import os
import time
import re
import sys
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

URL = os.environ.get("NESINE_URL", "https://www.nesine.com/iddaa?et=1&le=2")
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT = DATA_DIR / "nesine.json"
MAX_AGE_H = 36

HEADER = ["1", "X", "2", "Alt", "Üst", "HND", "1", "X", "2", "1-X", "1-2", "X-2", "Var", "Yok", "+"]
TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
ODDS_RE = re.compile(r"^\d{1,4}[.,]\d{1,2}$")
HND_RE = re.compile(r"^-?\d+:-?\d+$")
INT_RE = re.compile(r"^\d{1,4}$")
BLOCK_WORDS = ("captcha", "access denied", "erişim reddedildi", "are you a robot",
               "verify you are human", "attention required", "forbidden")

_TR = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
                     "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c"})


def _ascii(s):
    s = unicodedata.normalize("NFKD", str(s).translate(_TR))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def istanbul_today():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Istanbul")).date()
    except Exception:  # noqa: BLE001
        return (datetime.now(timezone.utc) + timedelta(hours=3)).date()


# --------------------------------------------------------------------------
# Ayrıştırma
# --------------------------------------------------------------------------
MONTHS = {"ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5, "haziran": 6, "temmuz": 7,
          "agustos": 8, "eylul": 9, "ekim": 10, "kasim": 11, "aralik": 12}


def parse_label(label, today):
    """'Bugün', 'Yarın', '19.10.2026', '19 Ekim' gibi etiketleri tarihe çevirir; bilinmiyorsa None."""
    if not label:
        return None
    l = _ascii(label).strip()
    if l.startswith("bugun"):
        return today
    if l.startswith("yarin"):
        return today + timedelta(days=1)
    m = re.search(r"(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?", l)
    if m:
        d, mo, y = int(m[1]), int(m[2]), m[3]
    else:
        m = re.search(r"(\d{1,2})\s+([a-z]+)", l)
        if not m or m[2] not in MONTHS:
            return None
        d, mo, y = int(m[1]), MONTHS[m[2]], None
    try:
        if y:
            y = int(y) + (2000 if int(y) < 100 else 0)
            return date(y, mo, d)
        cand = date(today.year, mo, d)
        return cand if cand >= today - timedelta(days=60) else date(today.year + 1, mo, d)
    except ValueError:
        return None


SKIP_RE = re.compile(r"\((k|w|y)\)|\bu-?(1[5-9]|2[0-3])\b|\b(ii|iii)\b|\bkadin|\bwomen|\byedek|\brezerv|\bcastilla\b")


def split_name(name):
    parts = name.split(" - ")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    if len(parts) > 2:
        k = min(range(1, len(parts)),
                key=lambda k: abs(len(" - ".join(parts[:k])) - len(" - ".join(parts[k:]))))
        return " - ".join(parts[:k]).strip(), " - ".join(parts[k:]).strip()
    return None


def _odds(tok):
    if tok == "-":
        return None
    if ODDS_RE.match(tok):
        return float(tok.replace(",", "."))
    raise ValueError(tok)


def _row(lines, i):
    """lines[i] saat satırıdır. Başarılıysa (maç sözlüğü, sonraki satır) döner."""
    n = len(lines)
    if i + 2 >= n:
        return None
    names = split_name(lines[i + 1])
    if not names:
        return None
    j = i + 2
    mbs = None
    if INT_RE.match(lines[j]):
        mbs, j = int(lines[j]), j + 1
    vals = lines[j:j + 15]
    if len(vals) < 15:
        return None
    try:
        o = []
        for k, v in enumerate(vals):
            if k == 5:
                if v != "-" and not HND_RE.match(v):
                    raise ValueError(v)
                o.append(None if v == "-" else v)
            elif k == 14:
                if not INT_RE.match(v):
                    raise ValueError(v)
                o.append(int(v))
            else:
                o.append(_odds(v))
    except ValueError:
        return None
    home, away = names
    low = _ascii(home + " | " + away)
    return ({
        "time": lines[i], "home": home, "away": away, "mbs": mbs,
        "skip": bool(SKIP_RE.search(low)),
        "ms": o[0:3], "ou25": o[3:5],
        "hnd": {"line": o[5], "odds": o[6:9]},
        "cs": o[9:12], "kg": o[12:14], "n_markets": o[14],
    }, j + 15)


def parse_bulletin(text, today=None, state=None):
    """Sayfa metninden maç listesi çıkarır. (maçlar, ayrıştırılamayan örnekler) döner.
    `state` verilirse (sözlük), önceki kaydırma adımından kalan lig/tarih bilgisiyle başlar
    ve sonunda güncellenir; böylece listenin ortasından başlayan metin parçaları da okunur."""
    today = today or istanbul_today()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out, bad = [], []
    st = state if state is not None else {}
    league, label, in_block = st.get("league"), st.get("label"), st.get("in_block", False)
    i, n = 0, len(lines)
    while i < n:
        if lines[i:i + 15] == HEADER:
            label = lines[i - 1] if i >= 1 else None
            league = lines[i - 2] if i >= 2 else None
            in_block = True
            i += 15
            continue
        if in_block and TIME_RE.match(lines[i]) and i + 1 < n and " - " in lines[i + 1]:
            res = _row(lines, i)
            if res:
                m, nxt = res
                d = parse_label(label, today)
                m.update({"league": league, "label": label, "date": d.isoformat() if d else None})
                out.append(m)
                i = nxt
                continue
            bad.append(" | ".join(lines[i:i + 8]))
        i += 1
    st.update(league=league, label=label, in_block=in_block)
    return out, bad


# --------------------------------------------------------------------------
# Takım adı eşleştirme
# --------------------------------------------------------------------------
STOP = {"fc", "cf", "sc", "sk", "ac", "as", "ssc", "afc", "fk", "bk", "sv", "vfl", "vfb", "rc", "ud",
        "cd", "rcd", "ca", "us", "ss", "spor", "kulubu", "club", "calcio", "de", "la", "le", "fsv", "tsv"}

GROUPS = [
    ["manchester city", "man city"], ["manchester united", "man united", "man utd", "manchester utd"],
    ["tottenham", "tottenham hotspur", "spurs"], ["wolverhampton", "wolves", "wolverhampton wanderers"],
    ["newcastle", "newcastle united", "newcastle utd"],
    ["nottingham forest", "nottm forest", "nott m forest", "nottingham"],
    ["west ham", "west ham united"], ["brighton", "brighton hove albion", "brighton and hove albion"],
    ["leeds", "leeds united"], ["leicester", "leicester city"], ["sheffield united", "sheffield utd"],
    ["west brom", "west bromwich albion", "west bromwich"], ["crystal palace", "c palace"],
    ["atletico madrid", "ath madrid", "atl madrid", "atletico de madrid", "atletico"],
    ["athletic bilbao", "ath bilbao", "athletic club", "athletic"],
    ["real sociedad", "sociedad"], ["real betis", "betis"], ["celta vigo", "celta"],
    ["real valladolid", "valladolid"], ["rayo vallecano", "vallecano"], ["espanyol", "espanol"],
    ["inter", "inter milan", "internazionale", "inter milano"], ["milan", "ac milan"],
    ["verona", "hellas verona"],
    ["bayern munich", "bayern munchen", "bayern munih", "bayern"],
    ["borussia dortmund", "dortmund"], ["bayer leverkusen", "leverkusen"],
    ["borussia monchengladbach", "m gladbach", "mgladbach", "gladbach", "monchengladbach",
     "b monchengladbach", "borussia m gladbach"],
    ["eintracht frankfurt", "ein frankfurt", "frankfurt"], ["rb leipzig", "leipzig"],
    ["koln", "cologne"], ["union berlin", "un berlin"], ["st pauli"], ["werder bremen", "bremen"],
    ["paris saint germain", "paris sg", "psg", "paris st germain"],
    ["marseille", "olympique marsilya", "olympique marseille"],
    ["lyon", "olympique lyon", "olympique lyonnais"], ["saint etienne", "st etienne"],
    ["buyuksehyr", "istanbul basaksehir", "basaksehir", "medipol basaksehir", "i basaksehir"],
    ["rizespor", "caykur rizespor"], ["genclerbirligi", "genclerbirligi ankara"],
    ["karagumruk", "fatih karagumruk"], ["goztepe", "goztep"], ["eyupspor", "eyup"],
    ["alanyaspor", "alanya"], ["gaziantep", "gaziantep fk"], ["adana demirspor", "a demirspor"],
    ["sporting cp", "sporting lisbon", "sporting lisboa", "sporting", "lisbon"],
    ["psv", "psv eindhoven"], ["porto"], ["benfica", "sl benfica"],
    ["club brugge", "brugge", "club bruges"], ["anderlecht", "rsc anderlecht"],
]


TOKEN_SYN = {                # Nesine'nin Türkçe/kısaltılmış şehir adları -> geçmiş verilerdeki yazılış
    "marsilya": "marseille", "kopenhag": "copenhagen", "munih": "munchen", "prag": "praha", "lizbon": "lisbon",
    "viyana": "wien", "varsova": "warszawa", "bukres": "bucuresti", "moskova": "moskva", "selanik": "thessaloniki",
    "anvers": "antwerp", "sofya": "sofia", "belgrad": "beograd", "kiev": "kyiv", "brüksel": "brussels",
    "bruksel": "brussels", "atina": "athens", "roma": "roma", "milano": "milan", "torino": "turin",
}


def norm(s):
    s = _ascii(s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [TOKEN_SYN.get(t, t) for t in s.split() if t not in STOP and not t.isdigit()]
    if len(toks) > 1:                                  # "B. Leverkusen", "U. S. Gilloise": tek harflik baş harfler atılır
        toks = [t for t in toks if len(t) > 1] or toks
    return " ".join(toks)


_CANON = {}
for _g in GROUPS:
    for _v in _g:
        _CANON[norm(_v)] = norm(_g[0])


def canon(s):
    n = norm(s)
    return _CANON.get(n, n)


def same_team(a, b):
    na, nb = canon(a), canon(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = set(na.split()), set(nb.split())
    short, long_ = (ta, tb) if len(na) <= len(nb) else (tb, ta)
    if len(min(na, nb, key=len)) >= 5 and short <= long_:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.86


def match_rows(rows, nes, include_skip=False):
    """rows: [(tarih(date), ev, deplasman)]. Her satır için eşleşen Nesine maçı (ya da None) döner.
    Ev ve deplasman takımı birlikte eşleşmeli, tarih en fazla 1 gün farklı olabilir."""
    used, out = set(), []
    for d, h, a in rows:
        hit = None
        for k, m in enumerate(nes):
            if k in used or (m.get("skip") and not include_skip):
                continue
            if d and m.get("date") and abs((date.fromisoformat(m["date"]) - d).days) > 1:
                continue
            if same_team(h, m["home"]) and same_team(a, m["away"]):
                hit = k
                break
        if hit is not None:
            used.add(hit)
        out.append(nes[hit] if hit is not None else None)
    return out


def load(path=OUT, max_age_h=MAX_AGE_H):
    """Kayıtlı Nesine verisini döner; yoksa ya da çok eskiyse None."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        age = datetime.now(timezone.utc) - datetime.fromisoformat(d["fetched"])
        if age > timedelta(hours=max_age_h) or not d.get("matches"):
            return None
        return d
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Sayfayı açma ve özet
# --------------------------------------------------------------------------
def summarize(matches, bad, status=None, title=None):
    from collections import Counter
    print(f"HTTP durumu: {status} | Başlık: {title}")
    ok = [m for m in matches if not m["skip"]]
    print(f"Ayrıştırılan maç: {len(matches)} (kadın/genç/yedek diye atlananlar: {len(matches) - len(ok)})")
    print(f"Ayrıştırılamayan satır grubu: {len(bad)}")
    print("Tarih etiketleri:", dict(Counter(f"{m['label']} -> {m['date']}" for m in matches)))
    print("Lig sayısı:", len({m['league'] for m in matches}))
    print("MBS dağılımı:", dict(Counter(m["mbs"] for m in matches)))
    print("---- İlk 12 maç ----")
    for m in matches[:12]:
        print(f"{m['date']} {m['time']} | {m['league']} | {m['home']} - {m['away']} | MBS {m['mbs']} | "
              f"MS {m['ms']} | Alt/Üst {m['ou25']} | KG {m['kg']} | ÇŞ {m['cs']}")
    if bad:
        print("---- Ayrıştırılamayan örnekler (ilk 6) ----")
        for b in bad[:6]:
            print("  ", b)


def _merge(store, matches):
    for m in matches:
        store[(m["date"], m["home"], m["away"])] = m


_SCROLLABLES_JS = """() => [...document.querySelectorAll('*')].filter(e => e.scrollHeight > e.clientHeight + 200 &&
  /(auto|scroll)/.test(getComputedStyle(e).overflowY)).slice(0, 6)
  .map(e => e.tagName + '.' + String(e.className).slice(0, 40) + ' ' + e.scrollHeight + '/' + e.clientHeight)"""

_SCROLL_ALL_JS = """() => {
  window.scrollBy(0, 1400);
  for (const e of document.querySelectorAll('*')) {
    if (e.scrollHeight > e.clientHeight + 200 && /(auto|scroll)/.test(getComputedStyle(e).overflowY))
      e.scrollTop += Math.max(600, e.clientHeight * 0.8);
  }
}"""


def _header_count(text):
    """Sayfa üstündeki 'FUTBOL <sayı>' sekmesinden toplam maç sayısı (bulunamazsa None)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for k in range(len(lines) - 1):
        if lines[k].upper() == "FUTBOL" and lines[k + 1].isdigit():
            return int(lines[k + 1])
    return None


_DAY_TXT = re.compile(
    r"^(yarın|"
    r"(pzt|sal|çar|per|cum|cmt|paz|pazartesi|salı|çarşamba|perşembe|cuma|cumartesi|pazar)\.?(\s+\d{1,2})?|"
    r"\d{1,2}\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)"
    r"(\s+(pazartesi|salı|çarşamba|perşembe|cuma|cumartesi|pazar))?)$", re.I)

_CLICKABLES_JS = """() => [...new Set([...document.querySelectorAll('button, a, li, [role=tab], [role=button], span, div')]
  .filter(e => e.children.length <= 1 && e.offsetParent !== null)
  .map(e => (e.innerText || '').trim().replace(/\\s+/g, ' ')).filter(t => t && t.length <= 24))]"""


def _day_tabs(page, store, state):
    """Gün sekmeleri (Yarın, 20 Ekim, Salı ...) varsa tek tek tıklayıp o günün maçlarını da toplar."""
    try:
        texts = page.evaluate(_CLICKABLES_JS)
    except Exception as e:                                   # noqa: BLE001
        print("Gün sekmeleri aranamadı:", e)
        return
    cands = [t for t in texts if _DAY_TXT.match(t) and t.lower() != "bugün"][:8]
    print("Tıklanabilir görünen kısa metinler (ilk 60):", texts[:60])
    print("Gün sekmesi adayları:", cands)
    for t in cands:
        try:
            before = len(store)
            page.get_by_text(t, exact=True).first.click(timeout=4000)
            page.wait_for_timeout(2500)
            idle = 0
            for _ in range(40):
                n0 = len(store)
                text = page.inner_text("body")
                ms, _bad = parse_bulletin(text, state=state)
                _merge(store, ms)
                idle = idle + 1 if len(store) == n0 else 0
                if idle >= 3:
                    break
                page.mouse.move(700, 1200)
                page.mouse.wheel(0, 1400)
                page.evaluate(_SCROLL_ALL_JS)
                page.wait_for_timeout(900)
            print(f"  '{t}' tıklandı: {len(store) - before} yeni maç")
        except Exception as e:                               # noqa: BLE001
            print(f"  '{t}' tıklanamadı: {str(e)[:80]}")


def _find_events(obj):
    """JSON içinde en uzun 'takım - takım' adlı kayıt listesini bulur."""
    best = []
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            if len(x) > len(best) and x and all(isinstance(i, dict) for i in x[:20]):
                hit = sum(any(isinstance(v, str) and " - " in v for v in i.values()) for i in x[:50])
                if hit >= min(len(x), 50) * 0.8:
                    best = x
            stack.extend(i for i in x[:200] if isinstance(i, (dict, list)))
    return best


def _explain_prebulten(body, needles):
    """Tam bülten JSON'unun yapısını logla gösterir (ayrıştırıcıyı bu yapıya göre yazmak için)."""
    if not body:
        print("getprebultenfull yanıtı yakalanamadı.")
        return
    obj = json.loads(body)
    print(f"getprebultenfull: {len(body)} bayt, üst düzey tür: {type(obj).__name__}")
    if isinstance(obj, dict):
        print("  üst düzey anahtarlar:", {k: (type(v).__name__, len(v) if hasattr(v, '__len__') else v)
                                         for k, v in list(obj.items())[:15]})
        for k, v in obj.items():
            if isinstance(v, dict):
                print(f"  '{k}' anahtarları:", {kk: (type(vv).__name__, len(vv) if hasattr(vv, '__len__') else vv)
                                                 for kk, vv in list(v.items())[:15]})
    ev = _find_events(obj)
    print(f"  bulunan maç kaydı sayısı: {len(ev)}")
    if ev:
        print("  ilk kaydın alanları:", {k: (type(v).__name__ if isinstance(v, (list, dict)) else v)
                                         for k, v in ev[0].items()})
    for nd in needles:
        for e in ev:
            if nd.lower() in json.dumps(e, ensure_ascii=False).lower():
                print(f"  --- '{nd}' içeren kayıt ---")
                print("  ", json.dumps(e, ensure_ascii=False)[:3500])
                break


# --------------------------------------------------------------------------
# Tam bülten verisi (sayfanın kendi indirdiği JSON) -> maç listesi
# --------------------------------------------------------------------------
def _ev_names(e):
    if isinstance(e.get("HN"), str) and isinstance(e.get("AN"), str):
        return e["HN"].strip(), e["AN"].strip()
    for k in ("ENN", "N", "EN", "NM"):
        v = e.get(k)
        if isinstance(v, str) and " - " in v:
            return split_name(v)
    return None


def _market_table(e):
    """Maçın pazarları: {(pazar türü MTID, çizgi): (MBS, [oranlar, seçenek sırasına göre])}.
    Aynı kimlikle birden fazla pazar varsa o kimlik belirsiz sayılıp tabloya konmaz."""
    out, dup = {}, set()
    for mk in e.get("MA") or []:
        if not isinstance(mk, dict):
            continue
        oca = [o for o in (mk.get("OCA") or []) if isinstance(o, dict)]
        oca.sort(key=lambda o: o.get("N", 0))
        odds = []
        for o in oca:
            v = o.get("O")
            odds.append(float(v) if isinstance(v, (int, float)) and v > 1.0 else None)
        # Pazar türü MTID'dir (1=Maç Sonucu, 3=Çifte Şans, 38=KG, 12=Alt/Üst ...); 'NO' her maçta farklıdır.
        key = (mk.get("MTID", mk.get("MT")), float(mk.get("SOV") or 0))
        if key in out:
            dup.add(key)
        out[key] = (mk.get("MBS"), odds)
    for k in dup:
        out.pop(k, None)
    return out


KNOWN_MARKETS = {            # canlı veriyle ekrandaki oranlara karşı doğrulanmış pazarlar (anahtar: "MTID|çizgi")
    "1|0": ("Maç Sonucu", ["1", "X", "2"]),
    "3|0": ("Çifte Şans", ["1-X", "1-2", "X-2"]),
    "38|0": ("Karşılıklı Gol", ["Var", "Yok"]),
    "12|2.5": ("Alt/Üst 2,5", ["Alt", "Üst"]),
}
NAMES_FILE = DATA_DIR / "pazar_adlari.json"      # otomatik öğrenilen pazar adları (anahtar: "MTID|çizgi")


def load_market_names(path=None):
    try:
        return json.loads(Path(path or NAMES_FILE).read_text(encoding="utf-8"))
    except Exception:                                # noqa: BLE001
        return {}



def parse_market_defs(ccall=None, bulletin=None):
    """Nesine'nin kendi sayfa kodundaki bahis tablosunu okur. Anahtar: bahis numarası (MTID).
    CCAll.min.js:  OutcomeGroups:{1:{<mtid>:{Title:"..",CouponTitle:"..",Value:[{id:1,title:".."}],BultenValue:[{id:1,title:".."}]}}}
    Bulletin.min.js: {title:"..",key:"<mtid>",liveKey:"..",outcomeMasks:[{title:"..",outcomeID:"1"}]}
    Dönen: {"def|<mtid>": {"name": başlık, "labels": {seçenek numarası: ad}}}. Bulunamayan numara için ad uydurulmaz."""
    defs = {}
    if ccall:
        i = ccall.find("OutcomeGroups:{")
        blk = ccall[i:] if i >= 0 else ""
        # yalnızca ilk spor (futbol=1) bloğu: ikinci spor bloğu "},2:{" ile başlar
        j = blk.find("}},2:{")
        blk = blk[:j + 1] if j > 0 else blk
        for m in re.finditer(r'(\d+):\{Title:"([^"]*)",CouponTitle:"[^"]*",Value:\[(.*?)\](?:,BultenValue:\[(.*?)\])?\}', blk):
            mtid, title, val, bval = m.groups()
            src = bval or val
            labels = {int(a): b for a, b in re.findall(r'\{id:(\d+),title:"([^"]*)"', src)}
            if title:
                defs.setdefault(f"def|{mtid}", {"name": title, "labels": labels})   # ilk tanım futbola ait; sonrakiler başka spor
    if bulletin:
        for m in re.finditer(r'\{title:"([^"]+)",key:"(\d+)"(?:,liveKey:"[^"]*")?(?:,specialSelector:"[^"]*")?,outcomeMasks:\[(.*?)\]\}', bulletin):
            title, mtid, masks = m.groups()
            labels = {int(a): b for b, a in re.findall(r'\{title:"([^"]*)",outcomeID:"(\d+)"', masks)}
            defs.setdefault(f"def|{mtid}", {"name": title, "labels": labels})
    return defs


def _markets_full(e, names=None):
    """Maçta açık olan BÜTÜN pazarlar: sabit bir listeyle sınırlanmaz; yeni pazar türleri kendiliğinden gelir.
    Adı bilinmeyen pazar 'name': None kalır (ad uydurulmaz)."""
    names = names or {}
    out = []
    for mk in e.get("MA") or []:
        if not isinstance(mk, dict):
            continue
        oca = sorted([o for o in (mk.get("OCA") or []) if isinstance(o, dict)], key=lambda o: o.get("N", 0))
        outs = []
        for o in oca:
            v = o.get("O")
            d = {"n": o.get("N"), "o": float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1.0 else None}
            if o.get("ON"):
                d["on"] = str(o["ON"])
            outs.append(d)
        if not outs:
            continue
        mtid = mk.get("MTID", mk.get("MT"))
        try:
            line = float(mk.get("SOV") or 0)
        except (TypeError, ValueError):
            line = 0.0
        key = f"{mtid}|{line:g}"
        known = KNOWN_MARKETS.get(key)
        labels = known[1] if known and len(known[1]) == len(outs) else None
        nm_ = known[0] if known else names.get(key)
        d_ = names.get(f"def|{mtid}")
        if not known and isinstance(d_, dict) and d_.get("name"):    # Nesine'nin kendi tablosu
            nm_ = d_["name"]
            lt = f"{line:g}".replace(".", ",")
            if "{{handicap}}" in nm_ or "{{SOV}}" in nm_:            # şablon: çizgi değeri adın içine girer
                nm_ = nm_.replace("{{handicap}}", lt).replace("{{SOV}}", lt)
            elif re.fullmatch(r"\d+,\d+", nm_.strip()) and any("Alt" in str(v) for v in d_.get("labels", {}).values()):
                nm_ = nm_.strip() + " Alt/Üst"                       # "1,5" -> "1,5 Alt/Üst" (seçenek adları 'Alt/Üst' diyor)
            elif line and lt not in nm_:
                nm_ += f" ({lt})"
            lab = [d_.get("labels", {}).get(o["n"]) for o in outs]
            ok_ = all(lab) and len(set(lab)) == len(lab) and not ("Evet" in lab and len(lab) > 2)
            if ok_:                                                  # tutarsız etiketler (ör. 'Evet' + 6 seçenek) kullanılmaz
                labels = [re.sub(r"\{\{(handicap|SOV)\}\}", lt, x) for x in lab]
        out.append({"id": mk.get("ID"), "mtid": mtid, "mst": mk.get("MST"), "line": line, "mbs": mk.get("MBS"),
                    "status": mk.get("MS"), "main": bool(mk.get("BM")),
                    "name": nm_, "labels": labels, "outs": outs})
    return out


def _sane(role, vals):
    """Oranlar mantıklı mı? (oranların terslerinin toplamı bahisçi payı kadar 1'den büyük olmalı)"""
    if any(v is None for v in vals):
        return True
    t = sum(1 / v for v in vals)
    lo, hi = {"ms": (0.97, 1.6), "ou25": (0.97, 1.6), "kg": (0.97, 1.6), "cs": (1.8, 3.3)}[role]
    return lo <= t <= hi


_ROLES = (("ms", 3), ("ou25", 2), ("kg", 2), ("cs", 3))


def _calibrate(events, dom):
    """Ekrandaki (doğru bildiğimiz) oranlarla JSON'u karşılaştırıp hangi pazarın MS, Alt/Üst 2.5, KG, ÇŞ
    olduğunu ve seçenek sırasını öğrenir. Böylece pazar numaralarını tahmin etmemiz gerekmez."""
    import itertools
    from collections import Counter
    by_name = {}
    for e in events:
        nm = _ev_names(e)
        if nm:
            by_name.setdefault((nm[0].lower(), nm[1].lower(), e.get("T")), e)
    pairs = []
    for m in dom:
        e = by_name.get((m["home"].lower(), m["away"].lower(), m["time"]))
        if e is not None:
            pairs.append((m, e))
    sig, types = {}, Counter(e.get("TYPE") for _, e in pairs)
    for role, n in _ROLES:
        votes, elig = Counter(), 0
        for m, e in pairs:
            ref = m[role] if role != "ou25" else m["ou25"]
            ref = list(ref)
            if sum(v is not None for v in ref) < (n - 1 if n == 3 else 1):
                continue
            elig += 1
            for key, (_, odds) in _market_table(e).items():
                if len(odds) != n:
                    continue
                for perm in itertools.permutations(range(n)):
                    cand = [odds[p] for p in perm]
                    if all(r is None or (c is not None and abs(r - c) < 0.02) for r, c in zip(ref, cand)):
                        votes[(key, perm)] += 1
        if votes:
            (key, perm), v = votes.most_common(1)[0]
            if v >= 3 and v >= elig * 0.3:
                sig[role] = (key, perm)
                print(f"  kalibrasyon {role}: pazar {key}, sıra {perm}, uyum {v}/{elig}")
                continue
        print(f"  kalibrasyon {role}: bulunamadı (en iyi {votes.most_common(1)[:1]}, uygun {elig})")
    return sig, (types.most_common(1)[0][0] if types else None), len(pairs)


FIXED_SIG = {                # logda 3 ayrı çalıştırmada ekranla birebir doğrulandı: (pazar, çizgi), seçenek sırası
    "ms": ((1, 0.0), (0, 1, 2)), "ou25": ((12, 2.5), (0, 1)), "kg": ((38, 0.0), (0, 1)), "cs": ((3, 0.0), (0, 1, 2))}
FULL_SCAN = os.environ.get("NESINE_FULL") == "1"     # varsayılan hafif mod: Nesine'ye tek sayfa yüklemesi, tarama yok
RAW_FILE = DATA_DIR / "nesine_ham.json"


def import_raw(path=None, out=None):
    """Tarayıcıdan elle indirilen ham bülten dosyasını (data/nesine_ham.json) data/nesine.json'a çevirir.
    GitHub'ın sunucusundan Nesine'ye erişilemediğinde, kullanıcının kendi tarayıcısından alınan veri kullanılır."""
    path, out = Path(path or RAW_FILE), Path(out or OUT)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        body = raw["prebulten"]
        body = body if isinstance(body, str) else json.dumps(body)
        ts = raw.get("fetched") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    except Exception as e:                               # noqa: BLE001
        print("Ham dosya okunamadı:", str(e)[:120])
        return 1
    names = load_market_names()
    print("  ham dosya alanları:", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in raw.items()})
    if raw.get("sayfa"):
        print("  tarayıcı sayfasındaki sayılar (FUTBOL vb.):", raw["sayfa"])
    if raw.get("delta"):
        d_ = raw["delta"]
        print("  delta yanıtı:", (d_[:300] if isinstance(d_, str) else d_))
    js = matches_from_prebulten(body, None, names)
    if not js:
        print("Ham dosyadan maç çıkmadı.")
        return 3
    lc = {m["league_id"]: m["league"] for m in js if m.get("league") and m.get("league_id") is not None}
    for m in js:
        if not m.get("league"):
            m["league"] = lc.get(m.get("league_id"))
    try:
        res = raw.get("results")
        if res:
            update_result_store(js, results_from_feed(res if isinstance(res, (str, bytes)) else json.dumps(res)))
    except Exception as e:                               # noqa: BLE001
        print("  sonuç akışı işlenemedi:", str(e)[:100])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fetched": ts, "url": "elle yüklenen tarayıcı verisi", "source": "nesine.com",
                               "matches": js}, ensure_ascii=False), encoding="utf-8")
    print(f"Ham dosyadan {len(js)} maç yazıldı → {out} (veri zamanı: {ts})")
    try:
        append_history(js, ts, out.parent / "oran_gecmisi.jsonl")
    except Exception as e:                               # noqa: BLE001
        print("  oran geçmişi yazılamadı:", str(e)[:100])
    return 0


def matches_from_prebulten(body, dom_matches, names=None):
    """Sayfanın indirdiği tam bülten JSON'undan tüm futbol maçlarını çıkarır. Hata olursa [] döner."""
    obj = json.loads(body)
    sg = obj.get("sg") if isinstance(obj, dict) else None
    events = sg.get("EA") if isinstance(sg, dict) and isinstance(sg.get("EA"), list) else _find_events(obj)
    events = [e for e in events if isinstance(e, dict)]
    league_of = {}
    for L in (sg.get("LA") if isinstance(sg, dict) else None) or []:
        if isinstance(L, dict) and L.get("LID") is not None and isinstance(L.get("N"), str):
            league_of[L["LID"]] = L["N"].strip()          # gerçek yapı: {"LID": 2, "N": "ABD Major Lig Playoff", "CC": "US"}
    if dom_matches is None:        # ekran okunamadı (elle yüklenen ham dosya): canlı verilerle doğrulanmış sabit eşleme
        sig, ftype, npairs = dict(FIXED_SIG), 1, 0
        print(f"  JSON'da {len(events)} kayıt; sabit pazar eşlemesi kullanıldı (MS=1, ÇŞ=3, KG=38, Alt/Üst 2,5=12)")
    else:
        sig, ftype, npairs = _calibrate(events, dom_matches)
        print(f"  JSON'da {len(events)} kayıt; ekranla eşleşen {npairs}; futbol TYPE değeri: {ftype}")
    if "ms" not in sig or ftype is None:
        print("  Kalibrasyon yapılamadı (MS pazarı bulunamadı), JSON kullanılmadı.")
        return []
    out = []
    seen_ids = set()
    from collections import Counter as _Ct
    drop = _Ct()
    print("  kayıtların TYPE dağılımı:", dict(_Ct(e.get("TYPE") for e in events).most_common(8)),
          "| sürümler:", {k: (sg or {}).get(k) for k in ("eventVersion", "marketVersion", "oddVersion")} if isinstance(sg, dict) else None)
    for e in events:
        if e.get("TYPE") != ftype:
            continue
        if e.get("C") is not None:
            if e.get("C") in seen_ids:                   # aynı maç iki kez gelirse tek kayıt
                drop["tekrar"] += 1
                continue
            seen_ids.add(e.get("C"))
        nm = _ev_names(e)
        d = str(e.get("D") or "")
        if not nm:
            drop["takım adı yok"] += 1
            continue
        if not re.match(r"^\d{2}\.\d{2}\.\d{4}$", d):
            drop["tarih okunamadı"] += 1
            continue
        tab = _market_table(e)
        vals = {}
        for role, n in _ROLES:
            if role in sig:
                key, perm = sig[role]
                odds = tab.get(key, (None, []))[1]
                v = [odds[p] if len(odds) == n else None for p in perm]
                vals[role] = v if _sane(role, v) else [None] * n
            else:
                vals[role] = [None] * n
        ms_key = sig["ms"][0]
        iso = f"{d[6:]}-{d[3:5]}-{d[:2]}"
        low = _ascii(nm[0] + " | " + nm[1])
        mk_full = _markets_full(e, names)
        out.append({"id": e.get("C"), "ev": e.get("EV"), "start_ms": e.get("ESD"), "league_id": e.get("LC"),
                    "source": "nesine.com/getprebultenfull", "markets": mk_full,
                    "time": e.get("T"), "home": nm[0], "away": nm[1], "mbs": tab[ms_key][0] if ms_key in tab else None,
                    "skip": bool(SKIP_RE.search(low)), "ms": vals["ms"], "ou25": vals["ou25"],
                    "hnd": {"line": None, "odds": [None, None, None]}, "cs": vals["cs"], "kg": vals["kg"],
                    "n_markets": len(mk_full), "league": league_of.get(e.get("LC")),
                    "label": "JSON", "date": iso})
    print(f"  futbol maçı: {len(out)} | atlananlar: {dict(drop)} | günlere göre: {dict(sorted(_Ct(m['date'] for m in out).items()))}"
          f" | MS oranı hiç olmayan: {sum(1 for m in out if all(v is None for v in m['ms']))}")
    return out



RESULT_FILE = DATA_DIR / "nesine_sonuc.json"


def results_from_feed(body):
    """GetLiveBetResultsWithVersion yanıtından {maç no: {"s": durum, "t": {tür: [ev, dep]}}}.
    Yapı: {"d":[{"C": maç no, "S": durum, "ES":[{"T": tür, "H": ev, "A": dep}, ...]}]}. T=1'in 'maç skoru' olduğu
    ve S=5'in 'bitti' demek olduğu (S=4 ikinci yarı, 3 devre arası, 2 ilk yarı) logdaki örneklerden ÇIKARILAN varsayımdır; doğrulama için fetch() örnekleri loga yazar."""
    try:
        obj = json.loads(body if isinstance(body, str) else body.decode("utf-8", "replace"))
    except Exception:                                    # noqa: BLE001
        return {}
    out = {}
    for r in (obj.get("d") if isinstance(obj, dict) else None) or []:
        if not isinstance(r, dict) or r.get("C") is None:
            continue
        sc = {}
        for e in r.get("ES") or []:
            if isinstance(e, dict) and isinstance(e.get("H"), int) and isinstance(e.get("A"), int) and e.get("T") is not None:
                sc.setdefault(e["T"], [e["H"], e["A"]])
        out[r["C"]] = {"s": r.get("S"), "t": sc}
    return out


def update_result_store(js, feed, path=None, today=None):
    """Bültendeki maçların adlarını (maç no ile) ve yayındaki sonuçları birikimli dosyada tutar: maç oynanınca bültenden
    düşer, adı ancak burada kalır. 25 günden eski kayıtlar silinir."""
    path = Path(path or RESULT_FILE)
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        st = {}
    ms, rs = st.get("matches", {}), st.get("results", {})
    for m in js:
        if m.get("id") is not None:
            ms[str(m["id"])] = {"h": m["home"], "a": m["away"], "d": m.get("date"), "lg": m.get("league")}
    for k, v in (feed or {}).items():
        if str(k) in ms:
            rs[str(k)] = {"s": v["s"], "t": {str(t): x for t, x in v["t"].items()}}
    cut = (date.today() - timedelta(days=25)).isoformat()
    ms = {k: v for k, v in ms.items() if (v.get("d") or "9") >= cut}
    rs = {k: v for k, v in rs.items() if k in ms}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"matches": ms, "results": rs}, ensure_ascii=False), encoding="utf-8")
    return ms, rs


def load_results_store(path=None):
    try:
        return json.loads(Path(path or RESULT_FILE).read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {"matches": {}, "results": {}}


HIST_FILE = DATA_DIR / "oran_gecmisi.jsonl"
_OUT_LABELS = {"1", "X", "2", "Alt", "Üst", "Var", "Yok", "1-X", "1-2", "X-2", "Evet", "Hayır", "Ev Sahibi", "Deplasman",
               "Beraberlik", "Tümü", "Daha fazla"}


def append_history(matches, ts, path=None):
    """Oranlar değiştikçe zaman damgalı kayıt ekler (eski kaydın üzerine yazılmaz): açılış oranı, son oran, değişim."""
    path = Path(path or HIST_FILE)
    last = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines()[-30000:]:
            try:
                r = json.loads(line)
                last[r["id"]] = r["odds"]
            except Exception:                            # noqa: BLE001  bozuk satır atlanır
                continue
    rows = []
    for m in matches:
        mid = m.get("id")
        if mid is None:
            continue
        odds = {k: m.get(k) for k in ("ms", "ou25", "kg", "cs")}
        if last.get(mid) != odds:
            rows.append(json.dumps({"t": ts, "id": mid, "d": m.get("date"), "h": m.get("home"), "a": m.get("away"),
                                    "odds": odds, "source": "nesine"}, ensure_ascii=False))
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
    return len(rows)


def _num(t):
    try:
        return float(str(t).strip().replace(",", "."))
    except ValueError:
        return None


def _seq_positions(flat, odds):
    """Sayfa satırlarında, verilen oran dizisinin (sırayla, arada etiketlerle) geçtiği başlangıç satırları."""
    for i, l in enumerate(flat):
        v = _num(l)
        if v is None or abs(v - odds[0]) > 1e-9:
            continue
        j, ok = i, True
        for o in odds[1:]:
            k, hit = j + 1, False
            while k < min(j + 6, len(flat)):
                w = _num(flat[k])
                if w is not None and abs(w - o) < 1e-9:
                    hit, j = True, k
                    break
                k += 1
            if not hit:
                ok = False
                break
        if ok:
            yield i


def _titles_from_text(text, ev):
    """Bir maçın açılmış bahis paneli metninden {pazar anahtarı: başlık}. Başlık, oran dizisinin hemen öncesindeki
    etiket olmayan satırdır. Kesin değildir; çağıran taraf birden fazla maçta tutarlılık arar."""
    flat = [l.strip() for l in text.splitlines() if l.strip()]
    got = {}
    for mk in ev.get("markets") or []:
        odds = [o["o"] for o in mk["outs"]]
        if len(odds) < 2 or any(o is None for o in odds):
            continue
        key = f"{mk['mtid']}|{mk['line']:g}"
        for i in _seq_positions(flat, odds):
            for k in range(i - 1, max(i - 9, -1), -1):
                t = flat[k]
                if _num(t) is not None or t in _OUT_LABELS or len(t) < 4 or len(t) > 45:
                    continue
                got.setdefault(key, []).append(t)
                break
    return got


def learn_market_names(page, js_matches, budget_s=75):
    """Sayfadaki bazı maçların bahis panelini açıp pazar adlarını öğrenmeye çalışır. Bir ad ancak en az 3 maçta aynı
    pazar için aynı başlıkla bulunursa kabul edilir. Başarısız olursa hiçbir şey eklenmez (ad uydurulmaz)."""
    from collections import Counter
    t0 = time.time()
    votes = {}
    cands = sorted([m for m in js_matches if len(m.get("markets") or []) >= 25 and not m.get("skip")],
                   key=lambda m: -len(m["markets"]))[:8]
    shown = False
    for m in cands:
        if time.time() - t0 > budget_s:
            break
        try:
            before = len(page.inner_text("body"))
            loc = page.get_by_text(f"{m['home']} - {m['away']}", exact=True).first
            if loc.count() == 0:
                loc = page.get_by_text(m["home"], exact=True).first
            if loc.count() == 0:
                if not shown:
                    print(f"  pazar adı öğrenme: satır bulunamadı ({m['home']} - {m['away']})")
                continue
            loc.scroll_into_view_if_needed(timeout=3000)
            loc.click(timeout=3000)
            page.wait_for_timeout(1800)
            text = page.inner_text("body")
            if not shown:
                shown = True
                print(f"  pazar adı öğrenme: panel açıldı mı? sayfa metni {before} -> {len(text)} karakter")
                i = text.find(m["home"])
                print("  panel metni örneği:", repr(text[max(i, 0):max(i, 0) + 700]))
            for key, ts in _titles_from_text(text, m).items():
                votes.setdefault(key, Counter()).update(ts)
            page.keyboard.press("Escape")
        except Exception as e:                           # noqa: BLE001
            print(f"  pazar adı öğrenme atlandı ({m['home']}): {str(e)[:80]}")
    learned = {}
    for key, c in votes.items():
        title, n = c.most_common(1)[0]
        if n >= 3 and n >= 0.7 * sum(c.values()):
            learned[key] = title
    print(f"  öğrenilen pazar adı: {len(learned)} (oy verilen pazar: {len(votes)})")
    return learned



def _log_extras(raw):
    """Sonraki geliştirmeler için: pazar adlarının ve maç sonuçlarının nerede durduğunu loga yazar."""
    obj = json.loads(raw["prebulten"])
    sg = obj.get("sg", {})
    print("  sg.BMV:", json.dumps(sg.get("BMV"), ensure_ascii=False)[:500])
    la = [x for x in (sg.get("LA") or []) if isinstance(x, dict)][:2]
    print("  sg.LA örnek:", json.dumps(la, ensure_ascii=False)[:500])
    nb = ((obj.get("nsn") or {}).get("nsnBultenNames") or [])[:3]
    print("  nsnBultenNames örnek:", json.dumps(nb, ensure_ascii=False)[:400])
    for u_, ln_, ctxs in raw.get("scan", []):
        print(f"  PAZAR-ADI-TARAMASI {u_} ({ln_} karakter)")
        for c_ in ctxs:
            print("    ...", repr(c_))
    if not raw.get("scan"):
        print("  PAZAR-ADI-TARAMASI: 'Çifte Şans' yazan dosya yakalanmadı")
    seo = raw.get("seo")
    if seo:
        t = seo.decode("utf-8", "replace")
        i = t.find("Maç Sonucu")
        print(f"  GetSeoConfig ({len(t)} bayt): 'Maç Sonucu' konumu {i}; çevresi:", repr(t[max(i - 300, 0):i + 500]) if i >= 0 else "yok")
    res = raw.get("results")
    if res:
        t = res.decode("utf-8", "replace")
        print(f"  GetLiveBetResults ({len(t)} bayt) başı:", repr(t[:900]))

def _log_sample_markets(js):
    """En çok pazarı olan maçın ham pazarlarını loga yazar: ekran görüntüsüyle oran oran eşleştirip ad çıkarmak için."""
    if not js:
        return
    m = max(js, key=lambda x: len(x.get("markets") or []))
    print(f"  ÖRNEK-MAÇ: {m['home']} - {m['away']} ({m.get('league')}) {m.get('date')} {m.get('time')} | pazar: {len(m['markets'])}")
    for mk in m["markets"]:
        print(f"  PZ {mk['mtid']}|{mk['line']:g} -> {[o['o'] for o in mk['outs']]}")


def fetch(url=URL, out=OUT):
    from playwright.sync_api import sync_playwright
    store, bad_all = {}, []
    with sync_playwright() as p:
        kw = {}
        if os.environ.get("CHROMIUM_PATH"):              # sadece yerel test için
            kw["executable_path"] = os.environ["CHROMIUM_PATH"]
        browser = p.chromium.launch(**kw)
        ctx = browser.new_context(locale="tr-TR", timezone_id="Europe/Istanbul",
                                  viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        api = []                                         # sayfanın kendi veri istekleri (sadece teşhis için)

        raw = {}                                         # sayfanın indirdiği tam bülten verisi

        def _scan(r, ct):
            """Pazar adlarının sayfanın kendi dosyalarında (JS/JSON) durup durmadığını arar; yalnızca loglar."""
            try:
                if "nesine" not in r.url or not any(k in ct for k in ("javascript", "json", "text/plain")):
                    return
                t = r.body().decode("utf-8", "replace")
                if "OutcomeGroups:{" in t:
                    raw["ccall"] = t
                if 'outcomeMasks:[{title:' in t and 'key:"268"' in t:
                    raw["bulletin"] = t
                hits = [m_.start() for m_ in re.finditer(r"[ÇC]ifte [ŞS]ans", t)][:2]
                if hits and len(raw.setdefault("scan", [])) < 6:
                    raw["scan"].append((r.url[:150], len(t), [t[max(i - 350, 0):i + 450] for i in hits]))
            except Exception:                            # noqa: BLE001
                pass

        def _on_resp(r):
            try:
                ct = r.headers.get("content-type", "")
                _scan(r, ct)
                if "json" in ct and "nesine" in r.url:
                    body = r.body()
                    api.append((r.status, r.url[:140], len(body)))
                    low = r.url.lower()
                    if "getprebultenfull" in low:
                        raw["prebulten"] = body
                    elif "getseoconfig" in low:
                        raw["seo"] = body
                    elif "getlivebetresults" in low:
                        raw["results"] = body
            except Exception:                            # noqa: BLE001
                pass
        page.on("response", _on_resp)
        try:
            resp, last_err = None, None
            for attempt in range(1, 4):                  # site bazen yavaş: 3 deneme, her seferinde daha uzun bekleme
                try:
                    resp = page.goto(url, wait_until="commit", timeout=45000 + attempt * 30000)
                    break
                except Exception as e:                   # noqa: BLE001
                    last_err = e
                    print(f"  sayfa açılışı {attempt}. deneme başarısız: {str(e)[:90]}")
                    page.wait_for_timeout(5000 * attempt)
            if resp is None:
                raise last_err
            status = resp.status
            try:
                page.wait_for_load_state("domcontentloaded", timeout=60000)
                page.wait_for_load_state("networkidle", timeout=30000)
            except Exception:                            # noqa: BLE001  sürekli istek atan sayfa
                pass
            page.wait_for_timeout(6000)
            title = page.title()
            text = page.inner_text("body")
            if (status and status >= 400) or any(w in text.lower() for w in BLOCK_WORDS):
                print(f"Site isteği engelledi (HTTP {status}). Devam edilmiyor.")
                return 2
            print("Sayfa üstündeki sayılar (CANLI / FUTBOL ...):", _header_count(text))
            state = {}
            ms, bad = parse_bulletin(text, state=state)
            _merge(store, ms)
            bad_all = bad
            # Gerçek bir kullanıcı gibi yavaş yavaş aşağı kaydır; liste kaydırdıkça yükleniyor.
            t0, idle, step = time.time(), 0, 0
            for step in range(1, 701 if FULL_SCAN else 3):          # hafif mod: sayfa bir kez açılır, kaydırma yok
                pos = page.evaluate("[Math.round(window.scrollY), document.documentElement.scrollHeight, innerHeight]")
                before = len(store)
                page.mouse.move(640, 400)
                page.mouse.wheel(0, 450)
                page.evaluate(_SCROLL_ALL_JS)
                page.wait_for_timeout(700)
                text = page.inner_text("body")
                ms, bad = parse_bulletin(text, state=state)
                _merge(store, ms)
                bad_all = bad or bad_all
                at_bottom = pos[0] + pos[2] >= pos[1] - 5
                idle = idle + 1 if (len(store) == before and at_bottom) else 0
                if step % 25 == 1:
                    print(f"  adım {step}: kaydırma {pos[0]}/{pos[1]} px, toplam maç {len(store)}")
                if at_bottom and len(store) == before:   # dipte yeni içerik gelsin diye biraz bekle
                    page.wait_for_timeout(1500)
                if idle >= 6 or time.time() - t0 > 360:
                    break
            print(f"Kaydırma adımı: {step} | toplam benzersiz maç: {len(store)} | süre: {int(time.time() - t0)} sn")
            tail = [l.strip() for l in text.splitlines() if l.strip()][-12:]
            print("Sayfa metninin son satırları:", tail)
            print("Sayfanın JSON istekleri (durum, adres, bayt):")
            for a_ in sorted(set(api), key=lambda x: -x[2])[:15]:
                print("  ", a_)
            try:
                if raw.get("prebulten"):
                    dom = list(store.values())
                    names = load_market_names()
                    try:
                        defs = parse_market_defs(raw.get("ccall"), raw.get("bulletin"))
                        print(f"  Nesine bahis tablosu: {len(defs)} bahis tanımı (ccall {'var' if raw.get('ccall') else 'yok'}, bulletin {'var' if raw.get('bulletin') else 'yok'})")
                        for k_, v_ in sorted(defs.items(), key=lambda kv: int(kv[0].split('|')[1]))[:400]:
                            print(f"  TANIM {k_[4:]} -> {v_['name']} | {v_['labels']}")
                        if defs:
                            names.update(defs)
                            NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
                            NAMES_FILE.write_text(json.dumps(names, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
                    except Exception as e:               # noqa: BLE001
                        print("  bahis tablosu okunamadı:", str(e)[:100])
                    js = matches_from_prebulten(raw["prebulten"], dom, names)
                    print(f"Tam bülten JSON'undan okunan futbol maçı: {len(js)} (ekrandan: {len(dom)})")
                    if not js and dom:                   # çalışmadıysa yapıyı ayrıntılı logla
                        _explain_prebulten(raw["prebulten"], [dom[0]["home"], dom[-1]["home"]])
                    for m in js:                         # JSON tam ve kayıpsız: ekrandaki aynı maçın yerine geçer
                        old = store.get((m["date"], m["home"], m["away"]))
                        if not m.get("league") and old and old.get("league"):
                            m["league"] = old["league"]      # lig adı JSON'da yoksa ekrandaki bültenden
                        store[(m["date"], m["home"], m["away"])] = m
                    lc_name = {}                         # aynı lig kodunu taşıyan maçlardan lig adı tamamlanır
                    for m in js:
                        if m.get("league") and m.get("league_id") is not None:
                            lc_name.setdefault(m["league_id"], m["league"])
                    for m in js:
                        if not m.get("league") and m.get("league_id") in lc_name:
                            m["league"] = lc_name[m["league_id"]]
                    print("  lig adı bulunamayan maç:", sum(1 for m in js if not m.get("league")))
                    try:
                        feed = results_from_feed(raw["results"]) if raw.get("results") else {}
                        ms_, rs_ = update_result_store(js, feed)
                        known = [i for i in feed if str(i) in ms_]
                        from collections import Counter as _C
                        print(f"  SONUÇ-AKIŞI: akışta {len(feed)} maç, bizim bültenle eşleşen {len(known)}; kayıtlı maç {len(ms_)}, sonuçlu {len(rs_)}")
                        print("  SONUÇ-AKIŞI durum dağılımı (S):", dict(_C(feed[i]["s"] for i in known)))
                        for i in known[:6]:
                            mm = ms_[str(i)]
                            print(f"  SONUÇ-ÖRNEK {mm['d']} {mm['h']} - {mm['a']} | S={feed[i]['s']} | skorlar(T:[ev,dep])={feed[i]['t']}")
                    except Exception as e:               # noqa: BLE001
                        print("  sonuç akışı işlenemedi:", str(e)[:100])
                    try:
                        _log_sample_markets(js)
                    except Exception as e:               # noqa: BLE001
                        print("  örnek maç yazılamadı:", str(e)[:100])
                    try:
                        _log_extras(raw)
                    except Exception as e:               # noqa: BLE001
                        print("  ek tanı atlandı:", str(e)[:100])
                    try:
                        learned = learn_market_names(page, js) if FULL_SCAN else {}
                        if learned:
                            names.update(learned)
                            NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
                            NAMES_FILE.write_text(json.dumps(names, ensure_ascii=False, indent=1, sort_keys=True),
                                                  encoding="utf-8")
                            print("  öğrenilen adlar:", learned)
                    except Exception as e:               # noqa: BLE001
                        print("  pazar adı öğrenme hatası:", str(e)[:100])
                else:
                    print("getprebultenfull yanıtı yakalanamadı.")
            except Exception as e:                       # noqa: BLE001
                print("Tam bülten JSON'u okunamadı:", e)
            if FULL_SCAN:
                _day_tabs(page, store, state)
        except Exception as e:                           # noqa: BLE001
            print(f"Sayfa açılamadı: {e}")
            return 1
        finally:
            browser.close()
    matches = list(store.values())
    summarize(matches, bad_all, status, title)
    if not matches:
        print("Hiç maç ayrıştırılamadı, dosya yazılmadı.")
        return 3
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.write_text(json.dumps({
        "fetched": ts, "url": url, "source": "nesine.com", "matches": matches}, ensure_ascii=False), encoding="utf-8")
    print(f"Yazıldı: {out}")
    try:
        n = append_history(matches, ts, out.parent / "oran_gecmisi.jsonl")
        print(f"Oran geçmişine eklenen değişen/yeni kayıt: {n}")
    except Exception as e:                               # noqa: BLE001
        print("Oran geçmişi yazılamadı:", str(e)[:100])
    return 0


def probe(url):
    """Başka bir bülten sayfasının yapısını görmek için: sayfa metninin başını ve sonunu yazdırır."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        kw = {}
        if os.environ.get("CHROMIUM_PATH"):
            kw["executable_path"] = os.environ["CHROMIUM_PATH"]
        browser = p.chromium.launch(**kw)
        try:
            ctx = browser.new_context(locale="tr-TR", timezone_id="Europe/Istanbul",
                                      viewport={"width": 1400, "height": 2400})
            page = ctx.new_page()
            bodies = []

            def _on(resp_):
                try:
                    ct = resp_.headers.get("content-type", "")
                    if "json" in ct and resp_.status == 200:
                        bodies.append((resp_.url, resp_.text()))
                except Exception:                        # noqa: BLE001
                    pass
            page.on("response", _on)
            resp = page.goto(url, wait_until="domcontentloaded", timeout=60000)
            status = resp.status if resp else None
            try:
                page.wait_for_load_state("networkidle", timeout=30000)
            except Exception:                            # noqa: BLE001
                pass
            page.wait_for_timeout(6000)
            for _ in range(8):                           # tembel yüklenen maçlar için kaydır
                page.mouse.wheel(0, 1500)
                page.wait_for_timeout(500)
            text = page.inner_text("body")
            print("---- JSON yanıtları (adres, bayt) ----")
            for u_, b_ in sorted(bodies, key=lambda x: -len(x[1]))[:12]:
                print("  ", len(b_), u_[:160])
            for u_, b_ in sorted(bodies, key=lambda x: -len(x[1]))[:2]:
                print("---- en büyük JSON örneği:", u_[:160])
                print(b_[:2500])
            print(f"HTTP {status} | başlık: {page.title()} | metin uzunluğu: {len(text)}")
            if (status and status >= 400) or any(w in text.lower() for w in BLOCK_WORDS):
                print("Site isteği engelledi. Devam edilmiyor.")
                return 2
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            print(f"Satır sayısı: {len(lines)}")
            print("---- İlk 150 satır ----")
            for l in lines[:150]:
                print("  ", l[:120])
            print("---- Son 30 satır ----")
            for l in lines[-30:]:
                print("  ", l[:120])
        except Exception as e:                           # noqa: BLE001
            print(f"Sayfa açılamadı: {e}")
            return 1
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        sys.exit(fetch())
    elif cmd == "import":
        sys.exit(import_raw(sys.argv[2] if len(sys.argv) > 2 else None))
    elif cmd == "probe" and len(sys.argv) > 2:
        sys.exit(probe(sys.argv[2]))
    elif cmd == "parse" and len(sys.argv) > 2:
        ms, bd = parse_bulletin(Path(sys.argv[2]).read_text(encoding="utf-8"))
        summarize(ms, bd)
    else:
        sys.exit(__doc__)
