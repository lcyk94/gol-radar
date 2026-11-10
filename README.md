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

## Sayfadaki araçlar
- **Kuponum (sepet):** maç panelindeki "Nesine'de açık bütün bahisler" kartlarında bir seçeneğe tıklanınca kupona eklenir. Tarayıcıda yalnızca `[maç no, bahis türü, çizgi, seçenek sırası, elle yazılan oran]` saklanır (`gr-slip`); adlar ve oranlar her açılışta güncel veriden okunur. Toplam oran, modele göre tutma olasılığı (seçimler bağımsız varsayımıyla çarpım) ve MBS uyarısı gösterilir. Paylaşım bağlantısı `#k=maçno,tür,çizgi,sıra~...` biçimindedir.
- **Oran hareketi:** `data/oran_gecmisi.jsonl` içindeki ilk ve son kayıt karşılaştırılır (`bot.odds_moves`), yalnızca MS, Alt/Üst 2.5 ve KG için. İlk kayıt gerçek açılış oranı değil, botun maçı ilk gördüğü orandır. Satırda %3 ve üzeri değişimde ▲/▼ görünür.
- **Favoriler:** takım ve lig yanındaki ☆; seçimler tarayıcıda saklanır (`gr-fav`). "★ Favorilerim" ve "Avrupa kupaları" süzgeçleri vardır.
- **Skor ısı haritası ve form grafiği:** maç panelinde. Isı haritası modelin 6x6 skor matrisinden (`sm`, binde bir birimlerle) çizilir.
- **Bacak isabeti:** kupon bacağının altında, o bahis türünün geriye dönük sınavdaki isabeti (`performance.fam`).

## 14 günde milyoner köşesi
Sanal 100 TL'lik kasa her gün tamamen bota en güvendiği, toplam oranı en az 2 olan 1-3 maçlık kuponda oynanır (`bot.build_milyoner`: oran ≥ 2 koşuluyla tutma olasılığı en yüksek kupon; bacakların MBS koşulu kupondaki maç sayısını aşmaz). Kupon maç günü ilk kaydedildiği hâliyle sabit kalır, `kupon_gecmisi.json` içinde `mil` işaretiyle saklanır ve diğer kupon istatistiklerine karışmaz. Sonuçlar mevcut kupon denetimiyle gelir; `mil_summary` seriyi çıkarır: tutan kuponun oranıyla kasa çarpılır, tutmayan denemeyi bitirir (ertesi gün yeniden 100 TL), 14 gün üst üste tutarsa 100 × 2^14 = 1.638.400 TL hedefine ulaşılmış sayılır. Uygun kupon çıkmayan gün atlanır. Sayfa, kuponun tutma olasılığından 14 günlük serinin olasılığını açıkça gösterir; köşe gerçek parayla tüm kasayı oynamak için bir öneri değildir.

## Model ayarı (haftalık)

`.github/workflows/tune.yml` haftada bir `python bot.py tune` çalıştırır. Şut verisinden üretilen beklenen gol (xG vekili), yarı ömür, ceza ve olasılık kısması geçmiş maçlarda
walk-forward (her hafta yalnızca o güne kadarki veriyle) denenir. Ölçüt 1X2, Üst 2.5 ve KG log-loss ortalamasıdır. Kazanç 0,0005'ten azsa varsayılan model kalır.
Sonuç `data/model_ayar.json` dosyasına yazılır; site bunu okur. Yeni bir özellik ancak ölçümde işe yararsa kullanılır.
