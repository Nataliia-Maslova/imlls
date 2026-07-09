"""
add_ca_nl.py
════════════════════════════════════════════════════════════════
Додає переклади валенсіано (ca) та нідерландської (nl) до:
  - vocabulary_translated.xlsx   (всі 29 аркушів, колонки ca / nl)
  - imlls_database_with_titles.xlsx
      · phrases: ca, topic_ca, nl, topic_nl
      · lessons:  topic_ca, topic_nl

Checkpoint спільний для обох файлів — кожна унікальна фраза
перекладається лише раз.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ПІДГОТОВКА (один раз)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install deep-translator openpyxl

  Запускай з тієї ж папки де лежать обидва xlsx-файли.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ЗАПУСК
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python add_ca_nl.py            # перший запуск або продовження
  python add_ca_nl.py --reset    # почати з нуля

  Час: ~45-60 хвилин (~9000 запитів × 0.35с)
  При збої — просто перезапусти, продовжить з checkpoint.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  РЕЗУЛЬТАТ (оригінали не змінюються)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  vocabulary_ca_nl.xlsx
  imlls_database_ca_nl.xlsx
════════════════════════════════════════════════════════════════
"""

import json
import time
import argparse
from pathlib import Path
from datetime import timedelta

import openpyxl
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError, TooManyRequests

# ── Файли ────────────────────────────────────────────────────────
VOCAB_IN    = "vocabulary_translated.xlsx"
VOCAB_OUT   = "vocabulary_ca_nl.xlsx"

IMLLS_IN    = "imlls_database_with_titles.xlsx"
IMLLS_OUT   = "imlls_database_ca_nl.xlsx"

CHECKPOINT  = "checkpoint_ca_nl.json"

# ── Нові мови ────────────────────────────────────────────────────
NEW_LANGS = [
    ("ca", "ca"),   # Valencian / Catalan
    ("nl", "nl"),   # Dutch
]

SOURCE_LANG      = "en"
DELAY            = 0.35
DELAY_ON_ERROR   = 7.0
MAX_RETRIES      = 4
SAVE_EVERY       = 50   # нових перекладів між збереженнями checkpoint

# ── Checkpoint ───────────────────────────────────────────────────

def load_checkpoint():
    if Path(CHECKPOINT).exists():
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_checkpoint(cache):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# ── Переклад ─────────────────────────────────────────────────────

def translate_text(text, target_code):
    if not text or not str(text).strip():
        return ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = GoogleTranslator(source=SOURCE_LANG, target=target_code).translate(str(text))
            time.sleep(DELAY)
            return result or ""
        except TooManyRequests:
            wait = DELAY_ON_ERROR * attempt
            print(f"    ⚠ RateLimit — чекаю {wait:.0f}с (спроба {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
        except RequestError as e:
            print(f"    ⚠ RequestError: {e} — чекаю {DELAY_ON_ERROR}с")
            time.sleep(DELAY_ON_ERROR)
        except Exception as e:
            print(f"    ⚠ Помилка: {e} (спроба {attempt}/{MAX_RETRIES})")
            time.sleep(DELAY_ON_ERROR)
    return "[ERROR]"

# ── Збір унікальних фраз ──────────────────────────────────────────

def collect_unique_phrases():
    """Збирає всі унікальні фрази з обох файлів."""
    unique = set()

    # vocabulary_translated.xlsx
    if Path(VOCAB_IN).exists():
        wb = openpyxl.load_workbook(VOCAB_IN)
        for name in wb.sheetnames:
            ws = wb[name]
            headers = [cell.value for cell in ws[1]]
            if "en" not in headers:
                continue
            en_col = headers.index("en") + 1
            for row in ws.iter_rows(min_row=2, values_only=True):
                val = row[en_col - 1]
                if val:
                    unique.add(str(val).strip())

    # imlls_database_with_titles.xlsx — phrases (en + topic_en)
    if Path(IMLLS_IN).exists():
        wb = openpyxl.load_workbook(IMLLS_IN)

        ws = wb["phrases"]
        headers = [cell.value for cell in ws[1]]
        for col_name in ("en", "topic_en"):
            if col_name not in headers:
                continue
            col_idx = headers.index(col_name) + 1
            for row in ws.iter_rows(min_row=2, values_only=True):
                val = row[col_idx - 1]
                if val:
                    unique.add(str(val).strip())

        ws2 = wb["lessons"]
        headers2 = [cell.value for cell in ws2[1]]
        if "topic_en" in headers2:
            col_idx = headers2.index("topic_en") + 1
            for row in ws2.iter_rows(min_row=2, values_only=True):
                val = row[col_idx - 1]
                if val:
                    unique.add(str(val).strip())

    return unique

# ── Переклад усіх унікальних фраз ────────────────────────────────

def translate_all(cache, unique_phrases, total_needed):
    """Перекладає всі ще не перекладені фрази."""
    new_count   = 0
    error_count = 0
    start_time  = time.time()

    already_done = sum(
        1 for phrase in unique_phrases
        for _, label in NEW_LANGS
        if cache.get(phrase, {}).get(label) not in (None, "[ERROR]")
    )
    print(f"✅ Вже готово (з checkpoint): {already_done}/{total_needed}")
    if already_done == total_needed:
        print("   Всі переклади вже є!\n")
        return new_count, error_count

    print()

    for phrase in sorted(unique_phrases):
        phrase_cache = cache.setdefault(phrase, {})
        any_new = False

        for code, label in NEW_LANGS:
            if phrase_cache.get(label) not in (None, "[ERROR]"):
                continue

            if not any_new:
                done_total = already_done + new_count
                pct = done_total / total_needed * 100
                print(f"  [{pct:5.1f}%] {phrase[:70]!r}")
                any_new = True

            translated = translate_text(phrase, code)
            phrase_cache[label] = translated
            new_count += 1

            if translated == "[ERROR]":
                error_count += 1
                print(f"      [{label}] ❌ ERROR")
            else:
                print(f"      [{label}] {translated[:70]!r}")

            if new_count % SAVE_EVERY == 0:
                save_checkpoint(cache)
                elapsed = time.time() - start_time
                done_total = already_done + new_count
                remaining = total_needed - done_total
                rate = new_count / elapsed if elapsed > 0 else 1
                eta = timedelta(seconds=int(remaining / rate))
                print(f"\n  💾 Checkpoint | залишилось ≈{eta}\n")

    return new_count, error_count

# ── Запис у vocabulary_translated.xlsx ───────────────────────────

def write_vocabulary(cache):
    if not Path(VOCAB_IN).exists():
        print(f"⚠ {VOCAB_IN} не знайдено — пропускаю")
        return

    wb = openpyxl.load_workbook(VOCAB_IN)
    for name in wb.sheetnames:
        ws = wb[name]
        headers = [cell.value for cell in ws[1]]
        if "en" not in headers:
            continue
        en_col = headers.index("en") + 1

        # Додати нові колонки
        existing = set(headers)
        for _, label in NEW_LANGS:
            if label not in existing:
                ws.cell(row=1, column=ws.max_column + 1, value=label)
                headers.append(label)

        lang_col_map = {label: headers.index(label) + 1 for _, label in NEW_LANGS}

        for row_idx in range(2, ws.max_row + 1):
            val = ws.cell(row=row_idx, column=en_col).value
            if not val:
                continue
            phrase_cache = cache.get(str(val).strip(), {})
            for _, label in NEW_LANGS:
                ws.cell(row=row_idx, column=lang_col_map[label],
                        value=phrase_cache.get(label, ""))

        print(f"   ✓ {name}")

    wb.save(VOCAB_OUT)
    print(f"   → {VOCAB_OUT}")

# ── Запис у imlls_database_with_titles.xlsx ──────────────────────

def write_imlls(cache):
    if not Path(IMLLS_IN).exists():
        print(f"⚠ {IMLLS_IN} не знайдено — пропускаю")
        return

    wb = openpyxl.load_workbook(IMLLS_IN)

    # --- phrases ---
    ws = wb["phrases"]
    headers = [cell.value for cell in ws[1]]
    en_col    = headers.index("en") + 1
    topic_col = headers.index("topic_en") + 1

    # Вставити нові колонки одразу після ru / topic_ru
    # Порядок: ... ru, topic_ru, nl, topic_nl, ca, topic_ca
    existing = set(headers)
    for _, label in NEW_LANGS:
        if label not in existing:
            ws.cell(row=1, column=ws.max_column + 1, value=label)
            headers.append(label)
        topic_label = f"topic_{label}"
        if topic_label not in existing:
            ws.cell(row=1, column=ws.max_column + 1, value=topic_label)
            headers.append(topic_label)

    col_map = {h: i + 1 for i, h in enumerate(headers)}

    for row_idx in range(2, ws.max_row + 1):
        en_val    = ws.cell(row=row_idx, column=en_col).value
        topic_val = ws.cell(row=row_idx, column=topic_col).value

        for _, label in NEW_LANGS:
            # phrase
            if en_val:
                ws.cell(row=row_idx, column=col_map[label],
                        value=cache.get(str(en_val).strip(), {}).get(label, ""))
            # topic
            if topic_val:
                ws.cell(row=row_idx, column=col_map[f"topic_{label}"],
                        value=cache.get(str(topic_val).strip(), {}).get(label, ""))

    print("   ✓ phrases")

    # --- lessons ---
    ws2 = wb["lessons"]
    headers2 = [cell.value for cell in ws2[1]]
    topic_col2 = headers2.index("topic_en") + 1

    existing2 = set(headers2)
    for _, label in NEW_LANGS:
        topic_label = f"topic_{label}"
        if topic_label not in existing2:
            ws2.cell(row=1, column=ws2.max_column + 1, value=topic_label)
            headers2.append(topic_label)

    col_map2 = {h: i + 1 for i, h in enumerate(headers2)}

    for row_idx in range(2, ws2.max_row + 1):
        topic_val = ws2.cell(row=row_idx, column=topic_col2).value
        if not topic_val:
            continue
        for _, label in NEW_LANGS:
            ws2.cell(row=row_idx, column=col_map2[f"topic_{label}"],
                     value=cache.get(str(topic_val).strip(), {}).get(label, ""))

    print("   ✓ lessons")

    wb.save(IMLLS_OUT)
    print(f"   → {IMLLS_OUT}")

# ── Main ─────────────────────────────────────────────────────────

def main(reset):
    print("🔍 Збираю унікальні фрази з обох файлів...")
    unique = collect_unique_phrases()
    total_needed = len(unique) * len(NEW_LANGS)

    print(f"   Унікальних фраз: {len(unique)}")
    print(f"   Нових перекладів: {total_needed}")
    eta = timedelta(seconds=int(total_needed * DELAY * 1.1))
    print(f"   Приблизний час: {eta}\n")

    if reset:
        Path(CHECKPOINT).unlink(missing_ok=True)
        print("🔄 Checkpoint скинуто\n")

    cache = load_checkpoint()

    # ── Переклад ─────────────────────────────────────────────────
    print("🌐 Перекладаю...")
    new_count, error_count = translate_all(cache, unique, total_needed)
    save_checkpoint(cache)

    print(f"\n✅ Переклад завершено. Нових: {new_count}, Помилок: {error_count}")

    # ── Запис у файли ────────────────────────────────────────────
    print(f"\n📝 Записую vocabulary_translated.xlsx...")
    write_vocabulary(cache)

    print(f"\n📝 Записую imlls_database_with_titles.xlsx...")
    write_imlls(cache)

    print("\n🎉 Готово!")
    if error_count:
        print(f"⚠ {error_count} помилок — перезапусти скрипт щоб повторити їх.")

# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Ігнорувати checkpoint і почати з нуля")
    args = parser.parse_args()
    main(reset=args.reset)
