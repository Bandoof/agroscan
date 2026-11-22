"""AgroScan — демо: фото листка → топ-3 хвороби з впевненістю + коротка порада.

Запуск:  streamlit run app.py
"""
import csv
import json
import random
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from agroscan.advice import ADVICE, NAME_UA  # noqa: E402
from agroscan.data import load_meta, short_name  # noqa: E402
from agroscan.predict import Predictor  # noqa: E402

# Поріг «низької впевненості» 0.5 перевірено для кожної моделі окремо:
#   v1 — results/confidence_check.json (PlantVillage test: 99.8 % точність вище порогу, 92 % нижче);
#   v2 — results/v2/confidence_check_v2.json (PlantDoc val: 69 % вище порогу, 43 % нижче).
LOW_CONF = 0.5
MODELS = {"v2 (рекомендовано)": "v2", "v1 (лише PlantVillage)": "v1"}

st.set_page_config(page_title="AgroScan", page_icon="🌿", layout="centered")


@st.cache_resource
def get_predictor(version: str):
    p = Predictor(REPO / "models" / f"agroscan_{version}.pt", device="cpu")  # CPU: ~25–30 мс на фото
    for _ in range(3):  # прогрів, щоб показаний час інференсу не включав «холодний» перший запуск
        p.predict(Image.new("RGB", (256, 256)))
    return p


@st.cache_data
def load_json(name):
    p = REPO / "results" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def example_paths():
    """Кілька фото з тестової вибірки PlantVillage (якщо датасет є на цьому комп'ютері)."""
    meta_p, split_p = REPO / "data" / "classes.json", REPO / "data" / "splits.csv"
    if not (meta_p.exists() and split_p.exists()):
        return []
    _, root = load_meta()
    with open(split_p, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["label"], []).append(r["path"])
    rng = random.Random(0)
    paths = [root / rng.choice(v) for _, v in sorted(by_cls.items())]
    return [p for p in paths if p.exists()]


cmp = load_json("v2/comparison_v1_v2.json")

with st.sidebar:
    st.header("Модель")
    version = MODELS[st.radio("Версія", list(MODELS), index=0,
                              help="v1 залишено для порівняння: вона навчена лише на лабораторних фото.")]
    pred = get_predictor(version)
    if version == "v2":
        st.write(f"**{pred.arch}**, 17 класів. Навчена на PlantVillage + польових фото PlantDoc (train), "
                 "з випадковими фонами і збалансованим семплером.")
    else:
        st.write(f"**{pred.arch}**, 17 класів. Навчена лише на лабораторних фото PlantVillage.")
    if cmp:
        r = cmp["results"][version]
        pv, pd = r["plantvillage_test"], r["plantdoc_test_clean"]
        st.metric("Accuracy, PlantVillage test", f"{pv['top1']:.1%}")
        st.caption(f"macro-F1 {pv['f1_macro']:.3f}, n = {pv['n']} лабораторних фото")
        st.metric("Accuracy, польові фото PlantDoc test", f"{pd['top1']:.1%}")
        ci = cmp["plantdoc_test_clean_bootstrap_ci95"][f"top1_{version}"]
        st.caption(f"95 % інтервал {ci[0]:.0%}–{ci[1]:.0%}, top-3 {pd['top3']:.1%}, n = {pd['n']} "
                   "(малий набір, 13 класів)")
    st.divider()
    st.caption("Навчальний проєкт з дисципліни «Архітектура комп'ютерних систем», ТНТУ. "
               "Не є засобом діагностики.")

st.title("🌿 AgroScan")
st.write("Завантажте фото **одного листка** яблуні, картоплі або томата — модель покаже "
         "три найімовірніші діагнози.")

if cmp:
    field_acc = cmp["results"][version]["plantdoc_test_clean"]["top1"]
    st.warning(f"На лабораторних фото модель {version} точна (~99 %), але на польових фото — значно менше "
               f"(≈{field_acc:.0%} на PlantDoc test). Висока впевненість не гарантує правильної відповіді. "
               "Для рішень щодо обробки зверніться до агронома.", icon="⚠️")

source = st.radio("Джерело фото", ["Завантажити файл", "Приклад з тестової вибірки"], horizontal=True)
img, caption = None, None
if source == "Завантажити файл":
    up = st.file_uploader("Фото листка (JPG / PNG)", type=["jpg", "jpeg", "png"])
    if up is not None:
        try:
            img, caption = Image.open(up), up.name
        except UnidentifiedImageError:
            st.error("Не вдалося відкрити файл як зображення.")
else:
    ex = example_paths()
    if not ex:
        st.info("Датасет PlantVillage не знайдено на цьому комп'ютері — завантажте власне фото.")
    else:
        choice = st.selectbox("Приклад (справжній клас у назві)", ex, format_func=lambda p: short_name(p.parent.name))
        img, caption = Image.open(choice), f"Справжній клас: {short_name(choice.parent.name)}"

if img is not None:
    top3, ms = pred.predict(img, k=3)
    col_img, col_res = st.columns([1, 1.2])
    with col_img:
        st.image(img, caption=caption, width="stretch")
    with col_res:
        st.subheader("Результат")
        for rank, (cls, p) in enumerate(top3, 1):
            label = f"{rank}. {NAME_UA[cls]} — {p:.1%}"
            st.markdown(f"**{label}**" if rank == 1 else label)
            st.progress(min(max(p, 0.0), 1.0))
        st.caption(f"Модель {version} · час інференсу на CPU: {ms:.0f} мс")

    best_cls, best_p = top3[0]
    if best_p < LOW_CONF:
        st.error(f"Низька впевненість ({best_p:.0%} < {LOW_CONF:.0%}). Результат ненадійний: "
                 "сфотографуйте один листок крупним планом на однорідному тлі.", icon="❓")
    desc, tip = ADVICE[best_cls]
    st.subheader("Що це і що робити")
    st.write(desc)
    st.info(tip, icon="💡")
    st.caption("Порада довідкова. Препарати й дози — лише за етикеткою засобу, дозволеного до "
               "використання, та після консультації агронома.")
