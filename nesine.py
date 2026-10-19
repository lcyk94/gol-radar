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
    ["sporting cp", "sporting lisbon", "sporting lisboa", "sporting"],
    ["psv", "psv eindhoven"], ["porto"], ["benfica", "sl benfica"],
    ["club brugge", "brugge", "club bruges"], ["anderlecht", "rsc anderlecht"],
]


def norm(s):
    s = _ascii(s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(t for t in s.split() if t not in STOP and not t.isdigit())


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


def match_rows(rows, nes):
    """rows: [(tarih(date), ev, deplasman)]. Her satır için eşleşen Nesine maçı (ya da None) döner.
    Ev ve deplasman takımı birlikte eşleşmeli, tarih en fazla 1 gün farklı olabilir."""
    used, out = set(), []
    for d, h, a in rows:
        hit = None
        for k, m in enumerate(nes):
            if k in used or m.get("skip"):
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

        def _on_resp(r):
            try:
                ct = r.headers.get("content-type", "")
                if "json" in ct and "nesine" in r.url:
                    api.append((r.status, r.url[:140], len(r.body())))
            except Exception:                            # noqa: BLE001
                pass
        page.on("response", _on_resp)
        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=60000)
            status = resp.status if resp else None
            try:
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
            for step in range(1, 701):
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
    out.write_text(json.dumps({
        "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": url, "matches": matches}, ensure_ascii=False), encoding="utf-8")
    print(f"Yazıldı: {out}")
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
            resp = page.goto(url, wait_until="domcontentloaded", timeout=60000)
            status = resp.status if resp else None
            try:
                page.wait_for_load_state("networkidle", timeout=30000)
            except Exception:                            # noqa: BLE001
                pass
            page.wait_for_timeout(6000)
            text = page.inner_text("body")
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
    elif cmd == "probe" and len(sys.argv) > 2:
        sys.exit(probe(sys.argv[2]))
    elif cmd == "parse" and len(sys.argv) > 2:
        ms, bd = parse_bulletin(Path(sys.argv[2]).read_text(encoding="utf-8"))
        summarize(ms, bd)
    else:
        sys.exit(__doc__)
