# AgroScan: діагностика хвороб рослин за фото листка

Навчальний ML-проєкт з дисципліни «Архітектура комп'ютерних систем» (ТНТУ).

Модель EfficientNet-B0, донавчена методом transfer learning, розпізнає 17 класів (хвороби та здоровий стан) для трьох культур: яблуня, картопля, томат. Демо на Streamlit показує три найімовірніші діагнози з впевненістю та коротку пораду.

Є дві версії моделі:
- **v1** навчена лише на лабораторних фото PlantVillage;
- **v2** (за замовчуванням у демо) додатково навчена на польових фото PlantDoc, з випадковими фонами та збалансованим семплером.

> ⚠️ На лабораторних фото обидві моделі точні (~99 %). На польових фото точність значно нижча: v2 ≈ 59 %, v1 ≈ 29 % на невеликому тесті PlantDoc (див. [Результати](#результати)). Це не засіб діагностики.

Усі метрики в цьому файлі взято з реальних запусків. Сирі логи та JSON лежать у [`results/`](results/).

## Обладнання та ПЗ

| | |
|---|---|
| Ноутбук | Lenovo IdeaPad Gaming 3 |
| GPU | NVIDIA GeForce RTX 2050, 4 ГБ VRAM (Ampere, compute 8.6), драйвер 581.80 |
| CPU | AMD Ryzen 5 5500H (4 ядра / 8 потоків) |
| ОС | Windows 11 |
| ПЗ | Python 3.11.9, PyTorch 2.14.1 + CUDA 12.6, cuDNN 9.10, torchvision 0.29.1, scikit-learn, matplotlib, Streamlit |

## Встановлення

```powershell
# venv варто тримати поза OneDrive: torch із CUDA займає ~5 ГБ
py -3.11 -m venv $HOME\.venvs\agroscan
& $HOME\.venvs\agroscan\Scripts\Activate.ps1
pip install -r requirements.txt
python check_env.py        # має показати "CUDA доступна: True" і назву GPU
```

Без NVIDIA GPU: приберіть з `requirements.txt` рядок `--extra-index-url` і суфікс `+cu126`. Демо працює на CPU (~25 мс на фото).

## Дані

Датасети **не зберігаються в репозиторії**. У репо лежить лише поділ на вибірки [`data/splits.csv`](data/splits.csv) і список класів [`data/classes.json`](data/classes.json).

| Датасет | Джерело | Ліцензія | Використання |
|---|---|---|---|
| PlantVillage (Mohanty et al., 2016) | [mohanty/PlantVillage](https://hf.co/datasets/mohanty/PlantVillage), Hugging Face | CC BY-SA 3.0 | навчання / val / test |
| PlantDoc (Singh et al., 2020) | [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset), GitHub | CC BY 4.0 | тест на польових фото |

```powershell
# 1. PlantVillage: data.zip (2.18 ГБ) і leaf-map.json → тека поза репо
$d="$HOME\datasets\plantvillage\download"; mkdir $d -Force
$u="https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main"
curl.exe -L -o "$d\leaf-map.json" "$u/leaf_grouping/leaf-map.json"
curl.exe -L -o "$d\data.zip" "$u/data.zip"

# 2. Витягнути яблуню / картоплю / томат і поділити 70/15/15 (seed 42)
python scripts/prepare_data.py --download $d --out $HOME\datasets\plantvillage\agroscan
python scripts/explore_data.py
```

> **Ліцензії зображень у `results/`.** Рисунки [`sample_grid.png`](results/sample_grid.png), [`plantdoc_examples.png`](results/plantdoc_examples.png), [`v2/bg_augmentation.png`](results/v2/bg_augmentation.png), [`v2/leak_candidates.png`](results/v2/leak_candidates.png) і [`v2/v1_vs_v2_examples.png`](results/v2/v1_vs_v2_examples.png) містять фото з датасетів і поширюються на умовах ліцензій цих датасетів:
> - **PlantVillage:** CC BY-SA 3.0, © Mohanty, Hughes, Salathé (2016);
> - **PlantDoc:** CC BY 4.0, © Singh et al. (2020).
>
> Посилання на першоджерела наведено в розділі [Посилання](#посилання).

Якщо фото лежать в іншій теці, ніж записано в `data/classes.json`, задайте змінну середовища `$env:AGROSCAN_IMAGES="<тека>\images"`.

**Підмножина:** 17 класів, 23 483 зображення.

**Поділ:** 16 516 / 3 504 / 3 463 (70.3 / 14.9 / 14.7 %).
- Ділили **за листками** (`leaf_id` з `leaf-map.json`) усередині кожного класу. Так усі знімки одного фізичного листка потрапляють лише в одну вибірку, і між train і test немає витоку даних.
- **Обмеження:** для ~29 % фото (частина класів томата) `leaf-map.json` не має запису. Кожне таке фото вважається окремим листком. Так само поводиться офіційний завантажувач датасету.
- **Дисбаланс класів:** приблизно 35× (від 5 357 фото в Tomato Yellow Leaf Curl Virus до 152 у Potato healthy).

Графіки: [`class_distribution.png`](results/class_distribution.png), [`sample_grid.png`](results/sample_grid.png).

## Навчання

```powershell
python scripts/train.py --arch efficientnet_b0     # ~22 хв на RTX 2050
python scripts/train.py --smoke                    # швидка перевірка: VRAM і швидкість
```

- **Етап 1 (3 епохи):** основа ImageNet заморожена, вчиться лише голова на 17 класів. AdamW, lr 1e-3.
- **Етап 2 (до 12 епох):** fine-tuning усієї мережі. lr 2e-4, OneCycle (cosine), рання зупинка за val macro-F1 (patience 4).
- **Налаштування:** 224×224, batch 64, AMP FP16 (пік VRAM 2.85 ГБ).
- **Функція втрат:** CrossEntropy з вагами класів (через дисбаланс) і label smoothing 0.1.
- **Аугментації:** RandomResizedCrop, віддзеркалення, поворот ±25°, ColorJitter.
- **Результат:** найкраща епоха 11, val macro-F1 0.9901. Чекпоінт: `models/agroscan_v1.pt` (15.7 МБ; `train.py` зберігає його як `agroscan_efficientnet_b0.pt`, потім перейменовано на v1). Лог: [`train_log_efficientnet_b0.csv`](results/train_log_efficientnet_b0.csv). Криві: [`training_curves.png`](results/training_curves.png).

### Модель v2: польові фото, випадкові фони, збалансований семплер

```powershell
python scripts/prepare_plantdoc.py --images $HOME\datasets\plantdoc\images   # поділ PlantDoc + пошук дублікатів
python scripts/extract_segmented.py --zip <data.zip> --out $HOME\datasets\plantvillage\agroscan\segmented
python scripts/train_v2.py                                                   # ~25 хв → models/agroscan_v2.pt
python scripts/leak_check.py                                                 # перевірка витоку test → навчання
python scripts/compare_v1_v2.py                                              # таблиця v1 vs v2
```

Що змінено у v2 порівняно з v1 (решта як у v1: EfficientNet-B0 з ImageNet, 224×224, batch 64, AMP, 2 + до 12 епох):

| | v1 | v2 |
|---|---|---|
| Дані | PlantVillage train (16 516) | + PlantDoc train (892 польових фото, 14 спільних класів) |
| Фон | як у датасеті | з імовірністю 0.5 сегментоване фото PlantVillage на випадковому фоні ([приклад](results/v2/bg_augmentation.png)) |
| Дисбаланс класів | ваги класів у loss | `WeightedRandomSampler`: ~25 % PlantDoc у кожній епосі, класи ∝ 1/√n; logit-adjusted CE (Menon et al., 2021) |
| Вибір епохи | val PlantVillage | val PlantDoc (224 фото) |
| Передбачення | логіти | логіти + log(prior навчального потоку): режим обрано на val ([`potato_healthy_check.json`](results/v2/potato_healthy_check.json)) |

**PlantDoc для v2:**
- офіційна тека `test` (114 фото) не використовується ні для навчання, ні для вибору епохи;
- теку `train` поділено 80/20 на train/val;
- 11 фото train, які є копіями фото test (MD5 / dHash), виключено з навчання;
- поділ записано в [`data/plantdoc_splits.csv`](data/plantdoc_splits.csv).

**Витік, знайдений після навчання.** Друга, незалежна перевірка ([`leak_check.py`](scripts/leak_check.py): ембединги EfficientNet-B0 з ImageNet, cos ≥ 0.78 або dHash ≤ 12, потім ручний перегляд кожної з 25 пар) знайшла ще **7 фото test**, чиї копії (кропи, перестиснення) потрапили в навчання v2: 5 у pd_train, 2 у pd_val. Список: [`data/plantdoc_test_leaks.json`](data/plantdoc_test_leaks.json). Ці 7 фото виключено з test, тож основні цифри v2 наведено на **107 фото**.

## Оцінка

```powershell
python scripts/evaluate.py          # PlantVillage test + криві + матриця помилок
python scripts/eval_plantdoc.py --zip <plantdoc.zip> --out $HOME\datasets\plantdoc\images
python scripts/bench_inference.py   # час інференсу CPU / GPU
```

Архів PlantDoc: `https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master` (~0.95 ГБ).

## Результати

### v1 vs v2 ([`comparison_v1_v2.md`](results/comparison_v1_v2.md))

Обидві моделі оцінено однаковим конвеєром на тих самих фото. Для PlantDoc у дужках наведено 95 % bootstrap-інтервал різниці v2 − v1 (2000 повторів).

| Набір | Метрика | v1 | **v2** | v2 − v1 |
|---|---|---|---|---|
| PlantVillage test (n = 3 463) | top-1 | 99.3 % | **98.8 %** | −0.5 п.п. |
| | top-3 | 100.0 % | **100.0 %** | 0 |
| | macro-F1 | 99.2 % | **98.9 %** | −0.3 п.п. |
| **PlantDoc test без витоку (n = 107)** | top-1 | 29.0 % | **58.9 %** | +29.9 п.п. [21.5; 39.3] |
| | top-3 | 53.3 % | **86.9 %** | +33.6 п.п. [24.3; 42.1] |
| | macro-F1 | 25.3 % | **57.7 %** | +32.4 п.п. [23.1; 41.1] |
| PlantDoc test повний (n = 114), довідково | top-1 | 29.8 % | 61.4 % | +31.6 п.п. |

**Приклади розбіжностей** ([`v1_vs_v2_examples.png`](results/v2/v1_vs_v2_examples.png), скрипт [`plot_v1_v2_examples.py`](scripts/plot_v1_v2_examples.py)). На чистому test є 34 фото, де права лише v2, і 2, де права лише v1. На рисунку 8 випадково вибраних фото першої групи (seed 42) і обидва фото другої. В обох помилках v2 правильно визначила хворобу (альтернаріоз), але сплутала культуру: картопля замість томата.

**Як вплинув витік.** На повному test (114) top-1 v2 становив 61.4 %, а без 7 фото з витоком — 58.9 % (−2.5 п.п.; macro-F1 −3.3 п.п.). Для v1, яка PlantDoc не бачила, цифри майже не змінилися: 29.8 % → 29.0 %.

**«Potato: healthy» у топ-3**, коли це не справжній клас:

| Набір | v1 | v2 |
|---|---|---|
| PlantVillage test | 99.0 % | 7.1 % |
| PlantDoc test | 79.4 % | 2.8 % |

Logit adjustment сам по собі цю проблему не прибрав (87 % на PlantVillage зі звичайними логітами), бо він навмисно підсилює рідкісні класи. Прибрало додавання log(prior) при передбаченні.

> ⚠️ **PlantDoc test малий: 107 фото, 13 класів, по 6–11 фото на клас.**
> - 95 % інтервали top-1 широкі: v1 [20.6; 38.3] %, v2 [49.5; 68.2] %. Тож «59 %» може бути і 50 %, і 68 %.
> - Різниця між моделями все одно надійна: на тих самих фото лише v2 вгадала 34, лише v1 — 2 (точний тест Мак-Немара p = 1.9·10⁻⁸).
> - Macro-F1 на такому наборі особливо шумний.
> - У PlantDoc є шум у мітках: серед знайдених копій «train ↔ test» є пари з різними мітками на тому самому фото, тож досяжна точність на test нижча за 100 %.

### PlantVillage, тестова вибірка, модель v1 (n = 3 463, модель її не бачила)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | Помилок |
|---|---|---|---|---|---|
| **0.9931** | 0.9912 | 0.9938 | **0.9924** | 0.9931 | 24 |

- Найчастіша помилка: Tomato Early blight → Late blight (7 разів).
- Найнижчий recall: Tomato Early blight (0.946).
- Деталі: [`test_classification_report.csv`](results/test_classification_report.csv), [`confusion_matrix_test.png`](results/confusion_matrix_test.png).

### PlantDoc, польові фото, модель v1 (лише 14 спільних класів)

Відповідність класів наведено в [`plantdoc_class_mapping.csv`](results/plantdoc_class_mapping.csv).
- PlantDoc має 28 класів. 14 з них мають відповідник серед наших 17.
- У PlantDoc немає трьох наших класів: Apple Black rot, Potato healthy, Tomato Target Spot.
- Модель обирає з усіх 17 класів, тож передбачення класу, якого немає в PlantDoc, рахується як помилка.

| Набір | n | Accuracy | Top-3 | Культура вгадана | F1 (macro) |
|---|---|---|---|---|---|
| Усі фото спільних класів | 1 241 | **0.258** | 0.494 | 0.674 | 0.219 |
| Лише тека `test` PlantDoc | 114 | 0.298 | 0.544 | 0.781 | 0.250 |

Точність падає з 99 % до 26 %. Це класичний зсув домену (domain shift): модель навчилась і на умовах зйомки (однорідне тло, один листок), а не лише на симптомах. На польових фото вона здебільшого обирає «Tomato: Early/Late blight» (див. [`confusion_matrix_plantdoc.png`](results/confusion_matrix_plantdoc.png), [`plantdoc_examples.png`](results/plantdoc_examples.png)).

Впевненість v1 на польових фото ненадійна: на PlantDoc навіть при top-1 ≥ 0.7 модель права лише у 42 % випадків ([`confidence_check.json`](results/confidence_check.json)). У v2 впевненість інформативніша: на PlantDoc val при top-1 ≥ 0.5 точність 69 %, нижче — 43 % ([`confidence_check_v2.json`](results/v2/confidence_check_v2.json)).

### Час інференсу, модель v1 (лише модель, медіана зі 100 запусків, [`inference_benchmark.json`](results/inference_benchmark.json))

Архітектура v2 та сама (EfficientNet-B0), тож час інференсу такий самий.

| Пристрій | Режим | Batch | Медіана |
|---|---|---|---|
| CPU Ryzen 5 5500H (4 потоки torch) | FP32 | 1 | 24.7 мс |
| CPU, повний шлях JPEG → топ-3 | FP32 | 1 | 26.6 мс |
| GPU RTX 2050 | FP32 | 1 | 10.6 мс |
| GPU RTX 2050 | AMP FP16 | 1 | 13.6 мс |
| GPU RTX 2050 | FP32 | 64 | 128 мс (500 img/s) |
| GPU RTX 2050 | AMP FP16 | 64 | 68 мс (936 img/s) |

### Апаратні спостереження ([`hardware_notes.md`](results/hardware_notes.md))

- **FP32 проти FP16 при навчанні (batch 64): ×15.1** (15.9 проти 239.6 img/s). Причина: FP32 потребує 5.5 ГБ, що більше за 4 ГБ VRAM. Драйвер у режимі WDDM переносить частину даних у системну RAM через PCIe. Якщо FP32 вміщується (batch 32), перевага FP16 лише ×1.9.
- **TF32 майже не впливає** (+5 %).
- **Depthwise-згортки однаково повільні в обох точностях** (×1.1). Виграш FP16 дають 1×1-згортки на Tensor Cores (×1.9) і поелементні операції, де FP16 удвічі зменшує обсяг даних, що передаються в пам'ять і з неї.
- **`channels_last` при AMP сповільнив EfficientNet-B0 з 248 до 16 img/s**, тому в навчанні його вимкнено. Причину окремо не досліджено.
- **DataLoader на Windows:** 72 img/s в одному процесі проти ~230 img/s із 6 воркерами. Старт воркерів займає ~30 с.
- **WinError 1455 при навчанні v2:** 18 воркерів DataLoader (кожен ~1.4 ГБ commit) вичерпали файл підкачки. Рішення: val-вибірки декодуються один раз у RAM, і лишається 6 воркерів (деталі — у `hardware_notes.md`).

## Демо

```powershell
streamlit run app.py --server.address localhost
```

Що показує застосунок:
- можна завантажити власне фото або вибрати приклад із тестової вибірки;
- перемикач моделі в боковій панелі: **v2** (за замовчуванням) або v1 для порівняння, з метриками й інтервалами обраної моделі;
- топ-3 діагнози українською з впевненістю, короткий опис і довідкова порада;
- попередження про низьку впевненість (поріг 0.5 перевірено для обох моделей: [v1](results/confidence_check.json), [v2](results/v2/confidence_check_v2.json));
- постійне попередження про польові фото.

## Структура

```
agroscan/            спільний код: data.py (датасет, трансформації), data_v2.py (PlantDoc, фони, семплер),
                     model.py, predict.py, advice.py (назви й поради), plots.py (стиль графіків)
scripts/             v1: prepare_data, explore_data, train, evaluate, eval_plantdoc, bench_inference,
                         bench_precision, profile_precision, confidence_check, smoke_predict
                     v2: prepare_plantdoc, extract_segmented, show_bg_aug, train_v2, leak_check,
                         record_leaks, compare_v1_v2, potato_healthy_check
app.py               Streamlit-демо (перемикач v1 / v2)
check_env.py         перевірка середовища та GPU
data/                splits.csv, classes.json, plantdoc_splits.csv, plantdoc_meta.json,
                     plantdoc_test_leaks.json (без зображень)
models/              agroscan_v1.pt, agroscan_v2.pt
results/             графіки, метрики, логи v1; results/v2/ — усе для v2; comparison_v1_v2.md
```

## Відомі обмеження

- Точність на польових фото навіть у v2 помірна: 58.9 % top-1 на 107 фото PlantDoc test, 95 % інтервал [49.5; 68.2] %. Польових даних для навчання було лише 892 фото.
- Витік test → train у v2 знайдено лише другою перевіркою (ембединги + ручний перегляд); не можна виключити ще не помічені кропи нижче порогу cos 0.78.
- ~29 % фото PlantVillage не мають ідентифікатора листка. Для них можливий невеликий витік «двійників» між вибірками.
- У v1 через ваги класів у loss рідкісний клас «Potato: healthy» часто стоїть на 2-му місці топ-3. У v2 це виправлено (7 % замість 99 % на PlantVillage test).
- У PlantDoc трапляються сумнівні мітки (наприклад, слайд із текстом або цілий плід замість листка). Їхню кількість не оцінювали.

## Посилання

- Mohanty S. P., Hughes D. P., Salathé M. Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7:1419, 2016. doi:10.3389/fpls.2016.01419
- Singh D., Jain N., Jain P., Kayal P., Kumawat S., Batra N. PlantDoc: A Dataset for Visual Plant Disease Detection. *CoDS-COMAD 2020*. doi:10.1145/3371158.3371196
- Tan M., Le Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
