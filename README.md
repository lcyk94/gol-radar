# Gol Radar

Nesine bülteninden maçları ve **açık bütün bahis kalemlerini** okur, geçmiş sonuçlardan takım gücü hesaplar, olasılık ve günlük kupon önerisi üretir. Statik site olarak GitHub Pages'te yayınlanır.

## Veri kaynakları
| Veri | Kaynak |
|---|---|
| Maç listesi, tüm pazarlar, oranlar | Yalnızca Nesine (`nesine.py`, Playwright ile bülten JSON'u) |
| Geçmiş maç sonuçları (model eğitimi, dünkü kupon sonucu) | football-data.co.uk CSV (yalnızca geçmiş skorlar; oran/fikstür alınmaz) |
Yurtdışı oran, fikstür, ClubElo ve ESPN kullanılmaz (ESPN yalnızca `GOLRADAR_ESPN=1` ile elle açılır).

## Dosyalar
- `nesine.py` – bülten okuma, pazar adı öğrenme (`data/pazar_adlari.json`), oran geçmişi (`data/oran_gecmisi.jsonl`, değişimde zaman damgalı satır).
- `bot.py` – model (Dixon-Coles/Poisson), kupon motoru, site üretimi.
- `template.html` – arayüz. `tests/` – pytest (sentetik veri).

## Etiketler
`model` = takım geçmişinden; `blend` = %50 model + %50 piyasa; `cross` = lig gücü tahmini; `market` = Nesine oranından arındırılmış olasılık (botun kendi tahmini değildir).

## Bilinen sınırlar
- Canlı Nesine ile test edilmedi; sentetik veriyle test edildi. Adı bilinmeyen pazarlar adsız gösterilir, ad uydurulmaz.
- Sakat/cezalı modülü yok (doğrulanmış, kullanımı uygun ücretsiz kaynak yok).
- Nesine kullanım şartları kullanıcı tarafından kontrol edilmelidir. 403/captcha aşılmaz.
