# AgroScan: діагностика хвороб рослин за фото листка

Навчальний ML-проєкт з дисципліни «Архітектура комп'ютерних систем» (ТНТУ).

Модель EfficientNet-B0, донавчена методом transfer learning, розпізнає 17 класів (хвороби та здоровий стан) для трьох культур: яблуня, картопля, томат. Демо на Streamlit показує три найімовірніші діагнози з впевненістю та коротку пораду.

> ⚠️ Модель навчена на лабораторних фото PlantVillage (один листок на однорідному тлі). На польових фото її точність значно нижча (див. [Результати](#результати)). Це не засіб діагностики.

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

> **Ліцензії зображень у `results/`.** Рисунки [`sample_grid.png`](results/sample_grid.png) і [`plantdoc_examples.png`](results/plantdoc_examples.png) містять фото з датасетів і поширюються на умовах ліцензій цих датасетів:
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
- **Результат:** найкраща епоха 11, val macro-F1 0.9901. Чекпоінт: `models/agroscan_efficientnet_b0.pt` (15.7 МБ). Лог: [`train_log_efficientnet_b0.csv`](results/train_log_efficientnet_b0.csv). Криві: [`training_curves.png`](results/training_curves.png).

## Оцінка

```powershell
python scripts/evaluate.py          # PlantVillage test + криві + матриця помилок
python scripts/eval_plantdoc.py --zip <plantdoc.zip> --out $HOME\datasets\plantdoc\images
python scripts/bench_inference.py   # час інференсу CPU / GPU
```

Архів PlantDoc: `https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master` (~0.95 ГБ).

## Результати

### PlantVillage, тестова вибірка (n = 3 463, модель її не бачила)

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | Помилок |
|---|---|---|---|---|---|
| **0.9931** | 0.9912 | 0.9938 | **0.9924** | 0.9931 | 24 |

- Найчастіша помилка: Tomato Early blight → Late blight (7 разів).
- Найнижчий recall: Tomato Early blight (0.946).
- Деталі: [`test_classification_report.csv`](results/test_classification_report.csv), [`confusion_matrix_test.png`](results/confusion_matrix_test.png).

### PlantDoc, польові фото (лише 14 спільних класів)

Відповідність класів наведено в [`plantdoc_class_mapping.csv`](results/plantdoc_class_mapping.csv).
- PlantDoc має 28 класів. 14 з них мають відповідник серед наших 17.
- У PlantDoc немає трьох наших класів: Apple Black rot, Potato healthy, Tomato Target Spot.
- Модель обирає з усіх 17 класів, тож передбачення класу, якого немає в PlantDoc, рахується як помилка.

| Набір | n | Accuracy | Top-3 | Культура вгадана | F1 (macro) |
|---|---|---|---|---|---|
| Усі фото спільних класів | 1 241 | **0.258** | 0.494 | 0.674 | 0.219 |
| Лише тека `test` PlantDoc | 114 | 0.298 | 0.544 | 0.781 | 0.250 |

Точність падає з 99 % до 26 %. Це класичний зсув домену (domain shift): модель навчилась і на умовах зйомки (однорідне тло, один листок), а не лише на симптомах. На польових фото вона здебільшого обирає «Tomato: Early/Late blight» (див. [`confusion_matrix_plantdoc.png`](results/confusion_matrix_plantdoc.png), [`plantdoc_examples.png`](results/plantdoc_examples.png)).

Впевненість моделі на польових фото ненадійна: на PlantDoc навіть при top-1 ≥ 0.7 модель права лише у 42 % випадків ([`confidence_check.json`](results/confidence_check.json)).

### Час інференсу (лише модель, медіана зі 100 запусків, [`inference_benchmark.json`](results/inference_benchmark.json))

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

## Демо

```powershell
streamlit run app.py --server.address localhost
```

Що показує застосунок:
- можна завантажити власне фото або вибрати приклад із тестової вибірки;
- топ-3 діагнози українською з впевненістю, короткий опис і довідкова порада;
- попередження про низьку впевненість (поріг 0.5 обрано за [`confidence_check.json`](results/confidence_check.json));
- постійне попередження про польові фото.

## Структура

```
agroscan/            спільний код: data.py (датасет, трансформації), model.py, predict.py,
                     advice.py (назви й поради), plots.py (стиль графіків)
scripts/             prepare_data, explore_data, train, evaluate, eval_plantdoc,
                     bench_inference, bench_precision, profile_precision, confidence_check, smoke_predict
app.py               Streamlit-демо
check_env.py         перевірка середовища та GPU
data/                splits.csv, classes.json (без зображень)
models/              найкращий чекпоінт
results/             графіки, метрики, логи
```

## Відомі обмеження

- Низька точність на польових фото (26 % на PlantDoc). Для практичного використання потрібне донавчання на польових даних.
- ~29 % фото PlantVillage не мають ідентифікатора листка. Для них можливий невеликий витік «двійників» між вибірками.
- Через ваги класів у loss рідкісний клас «Potato: healthy» часто стоїть на 2-му місці топ-3 (на top-1 це не впливає).
- У PlantDoc трапляються сумнівні мітки (наприклад, слайд із текстом або цілий плід замість листка). Їхню кількість не оцінювали.

## Посилання

- Mohanty S. P., Hughes D. P., Salathé M. Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7:1419, 2016. doi:10.3389/fpls.2016.01419
- Singh D., Jain N., Jain P., Kayal P., Kumawat S., Batra N. PlantDoc: A Dataset for Visual Plant Disease Detection. *CoDS-COMAD 2020*. doi:10.1145/3371158.3371196
- Tan M., Le Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
