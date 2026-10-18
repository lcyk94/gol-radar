#!/usr/bin/env python3
"""
Nesine keşif denemesi (sadece bir kez çalıştırılır, elle).

Sayfayı normal bir tarayıcı gibi BİR kez açar, ekranda görünen metni günlüğe yazar.
Amaç: oranların burada okunup okunamadığını görmek.
Engel (captcha, 403, "erişim reddedildi") çıkarsa durur. Engeli aşmaya çalışmaz.
"""
import os
import sys

from playwright.sync_api import sync_playwright

URL = os.environ.get("PROBE_URL", "https://www.nesine.com/iddaa?et=1&le=2")
MAX_LINES = int(os.environ.get("PROBE_LINES", "220"))
BLOCK_WORDS = ("captcha", "access denied", "erişim reddedildi", "are you a robot",
               "verify you are human", "cloudflare", "attention required", "forbidden")


def main():
    with sync_playwright() as p:
        kw = {}
        if os.environ.get("CHROMIUM_PATH"):          # sadece yerel test için
            kw["executable_path"] = os.environ["CHROMIUM_PATH"]
        browser = p.chromium.launch(**kw)
        ctx = browser.new_context(locale="tr-TR", timezone_id="Europe/Istanbul",
                                  viewport={"width": 1400, "height": 2400})
        page = ctx.new_page()
        try:
            resp = page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            status = resp.status if resp else None
            try:
                page.wait_for_load_state("networkidle", timeout=30000)
            except Exception:                         # sürekli istek atan sayfalar için normal
                pass
            page.wait_for_timeout(6000)               # oranların yüklenmesi için bekle
            title = page.title()
            text = page.inner_text("body")
        except Exception as e:                        # noqa: BLE001
            print(f"SAYFA AÇILAMADI: {e}")
            browser.close()
            sys.exit(1)
        browser.close()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    low = text.lower()
    blocked = [w for w in BLOCK_WORDS if w in low]
    print(f"HTTP durumu: {status}")
    print(f"Sayfa başlığı: {title}")
    print(f"Görünen metin: {len(lines)} satır")
    if status and status >= 400:
        print("SONUÇ: Site isteği reddetti (HTTP hata kodu). Devam edilmiyor.")
        sys.exit(2)
    if blocked:
        print(f"SONUÇ: Engel sayfası olabilir (bulunan ifadeler: {blocked}). Devam edilmiyor.")
    print("=" * 20, f"SAYFA METNİ (ilk {MAX_LINES} satır)", "=" * 20)
    print("\n".join(lines[:MAX_LINES]))
    print("=" * 20, "SON", "=" * 20)


if __name__ == "__main__":
    main()
