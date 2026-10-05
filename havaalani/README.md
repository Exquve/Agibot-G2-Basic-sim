# G2 havaalanı bavul ayırma

Sabit duran bir AGIBOT G2, giriş bandından gelen bavulları dört çıkış bandına ayırır. Robot modeli internetten alınır; hol, bantlar ve ayırma kodu bu klasördedir.

Çıkışlar, ayırma masasının önünden bakar gibi:

| Bant | Renk | Yön |
| --- | --- | --- |
| A SOL | kırmızı | sola |
| B DÜZ SOL | turuncu | düz, robotun solundan devam |
| C DÜZ SAĞ | mavi | düz, robotun sağından devam |
| D SAĞ | yeşil | sağa |

G2 yerinden oynamaz. Karar ve itme hareketi kollarındadır; konum servoları resmi modelin eklemleridir. Bantlar bavulu taşır. Gösteri kasıtlı olarak tutma yerine yönlendirme yapar: modeldeki gripper küçük iş parçası içindir, bavul ise kolun önündeki masada kayar.

## Çalıştırma

```bash
cd havaalani
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python sort_demo.py
```

İlk açılışta G2 modeli Hugging Face’teki [GenieSimAssets](https://huggingface.co/datasets/agibot-world/GenieSimAssets) paketinden iner (CC BY-NC-SA 4.0). Komut tarayıcıda canlı görüntü açar: sürükleyerek kamerayı çevirin, tekerlekle yakınlaşın. Ayrı bir MuJoCo penceresi için aynı komuta `--native` ekleyin.

Görüntüsüz deneme ve kare kaydı:

```bash
.venv/bin/python sort_demo.py --seconds 22 --shots frames
```
