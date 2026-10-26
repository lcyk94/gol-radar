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
- `pazarlar.py` – Nesine'nin ~40 bahis türü (ilk yarı, toplam gol, takım golü, handikap, MS&Alt/Üst, MS&KG, İY/MS vb.) için seçenek tanımları (Nesine'nin kendi bahis tablosundan), model olasılığı, sonuç değerlendirme ve oranlara karşı eşleme sınavı.
- `template.html` – arayüz. `tests/` – pytest (sentetik veri).

## Kupon seçimi
Aday: Nesine'nin maçta açtığı ve `pazarlar.py`'de tanımlı bütün bahis türleri. Seçimi **oran belirlemez**: her seçenek için modelin olasılığı, o ligin sıradan (ortalama gollü) bir maçındaki aynı seçeneğin olasılığıyla kıyaslanır (`edge = p - taban`); en az %55 olasılık ve 10 puan `edge` gerekir. Oran yalnızca kural süzgecidir (bacak ≥1,20, toplam ≥3, Alt yok). Aynı pazar ailesi tekrarlandıkça puan düşer (çeşitlilik). Her bacağa verilerden üretilen gerekçe yazılır. Bir bahis türü kuponlara ancak (1) eşlemesi her çalıştırmada modelin kendi maçlarında Nesine oranlarına karşı sınanıp geçerse ve (2) geçmişte yeşil dediği seçenekler modelin dediğinden en fazla 5 puan az tutarsa girer. İlk yarı payı football-data'daki HTHG/HTAG sütunlarından ölçülür; veri yoksa ilk yarı pazarları hesaplanmaz.

## Etiketler
`model` = takım geçmişinden; `blend` = %50 model + %50 piyasa; `cross` = lig gücü tahmini; `market` = Nesine oranından arındırılmış olasılık (botun kendi tahmini değildir).

## Bilinen sınırlar
- Canlı Nesine ile test edilmedi; sentetik veriyle test edildi. Adı bilinmeyen pazarlar adsız gösterilir, ad uydurulmaz.
- İlk yarı skorunun Nesine sonuç akışındaki türü (T=2) ve bitti durumu (S=5) log örneklerinden çıkarılan varsayımdır; ilk yarı bahisleri bu varsayıma göre sonuçlandırılır.
- Sonuç akışı biten maçı kısa süre tutar; `sonuc.yml` 20 dakikada bir okur. Okunmadan akıştan düşen maçın sonucu Nesine'den alınamaz (ana ligler ertesi gün football-data'dan gelir).
- Sakat/cezalı modülü yok (doğrulanmış, kullanımı uygun ücretsiz kaynak yok).
- Nesine kullanım şartları kullanıcı tarafından kontrol edilmelidir. 403/captcha aşılmaz.
