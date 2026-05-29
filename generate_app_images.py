"""
generate_app_images.py
Generates banner/topic images for the IMLLS app using the gpt-image-1 API.

Images generated:
  • 29 vocabulary topic images  → static/app_images/vocab_<slug>.png
  • 1  reading module banner    → static/app_images/reading_banner.png
  • 1  my-phrases module banner → static/app_images/my_phrases_banner.png

Usage:
    pip install openai requests
    python generate_app_images.py --api-key sk-proj-YOUR_KEY_HERE

Skips already-generated files, so safe to re-run after interruption.

Integration helpers (add to your Streamlit app):
    APP_IMG_DIR = ROOT / "static" / "app_images"

    def vocab_image(slug: str) -> Path:
        return APP_IMG_DIR / f"vocab_{slug}.png"

    # slugs match VOCAB_SLUGS dict below
"""

import argparse
import base64
import os
import time
import requests
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Installing openai...")
    os.system("pip install openai requests --break-system-packages")
    from openai import OpenAI


OUTPUT_DIR = Path(__file__).parent / "static" / "app_images"

# ---------------------------------------------------------------------------
# Style prefix applied to every prompt
# ---------------------------------------------------------------------------
_STYLE = (
    "Warm watercolour illustration, soft cream background, "
    "cartoon children aged 6-10, friendly and cheerful mood, "
    "pastel colours, hand-drawn feel. "
    "Absolutely no text, no words, no letters, no numbers, no labels, "
    "no signs, no writing of any kind anywhere in the image."
)

def _p(scene: str) -> str:
    """Combine style prefix with scene description."""
    return f"{_STYLE} Scene: {scene}"


# ---------------------------------------------------------------------------
# All images to generate
# ---------------------------------------------------------------------------
# Each entry: (filename_without_extension, short_label, scene_description)
IMAGES = [
    # ── Vocabulary topics ──────────────────────────────────────────────────
    (
        "vocab_greetings",
        "Greetings Basics & Courtesy",
        "Two cheerful children waving hello to each other, one bowing slightly, "
        "both smiling warmly, colourful backpacks on their backs.",
    ),
    (
        "vocab_questions",
        "Questions Directions & Emergencies",
        "A child with a puzzled expression holding a large round question-mark "
        "balloon, standing at a crossroads with friendly arrows pointing in "
        "different directions.",
    ),
    (
        "vocab_daily_life",
        "Daily Life Routine & Feelings",
        "A child going through a cheerful morning routine — stretching, "
        "eating breakfast, brushing teeth — shown as a circular daily cycle "
        "with small cozy home scenes.",
    ),
    (
        "vocab_basic",
        "Basic",
        "A curious child surrounded by everyday objects — a cup, a ball, "
        "a chair, a flower — each glowing softly as if being discovered "
        "for the first time.",
    ),
    (
        "vocab_verbs",
        "Verbs",
        "A lively child doing many actions at once — jumping, running, "
        "drawing, eating, sleeping — shown in small fun vignettes around "
        "the central figure.",
    ),
    (
        "vocab_food",
        "Food",
        "A child sitting at a round table covered with colourful fruits, "
        "vegetables, and dishes — apples, carrots, soup bowl, bread — "
        "all looking delicious and freshly painted.",
    ),
    (
        "vocab_city",
        "City",
        "A miniature cartoon city with a child walking down a sunny street "
        "past small shops, a park bench, a fountain, and cheerful buildings "
        "in pastel colours.",
    ),
    (
        "vocab_restaurant_food_shopping",
        "Restaurant Food & Shopping",
        "A child at a cozy café table with a plate of food, and next to them "
        "a small market stall with fruit and bread — warm indoor light, "
        "friendly atmosphere.",
    ),
    (
        "vocab_travel_lodging_weather",
        "Travel Lodging & Weather",
        "A child with a tiny suitcase standing outside a cheerful small hotel, "
        "with a bright sun, a cloud with raindrops, and a rainbow all visible "
        "in the sky above.",
    ),
    (
        "vocab_emotions",
        "Emotions",
        "Six small circular portraits of the same cartoon child showing "
        "different feelings — happy, sad, surprised, angry, scared, excited — "
        "arranged like a bouquet of feelings.",
    ),
    (
        "vocab_house",
        "House and Home",
        "A cozy cross-section of a cartoon house showing a child in the living "
        "room, a cat in the kitchen, and a bed in the bedroom — warm lamp-light "
        "throughout.",
    ),
    (
        "vocab_weather",
        "Weather",
        "A child outdoors experiencing all four weather types at once in "
        "different quadrants of the sky — sunshine, rain, snow, and wind — "
        "dressed in a mix of seasonal clothes.",
    ),
    (
        "vocab_shopping",
        "Shopping",
        "A child with a small basket at a colourful market stall choosing "
        "between fruit, toys, and books, with a friendly shopkeeper nearby.",
    ),
    (
        "vocab_daily_routine",
        "Daily Routine",
        "A timeline strip showing a child's day: waking up with sunrise, "
        "eating breakfast, going to school, playing, and sleeping under the "
        "moon — all in small vignette panels.",
    ),
    (
        "vocab_doctor",
        "At the Doctor",
        "A child sitting on an examination table looking calm, a kind doctor "
        "with a stethoscope smiling reassuringly, colourful posters of the "
        "human body (no text) on the wall.",
    ),
    (
        "vocab_food_drinks",
        "Food and Drinks",
        "A cheerful picnic blanket covered with a variety of foods and drinks — "
        "sandwiches, juice, cake, milk, and fresh vegetables — two children "
        "reaching for their favourites.",
    ),
    (
        "vocab_work",
        "Work",
        "Several cartoon adults doing different jobs — a chef cooking, a builder "
        "with a hard hat, a teacher pointing at a board — while a small child "
        "watches with admiration.",
    ),
    (
        "vocab_school",
        "School",
        "A bright classroom with a child at a desk drawing, a pencil holder, "
        "a globe, colourful notebooks, and a friendly teacher in the background.",
    ),
    (
        "vocab_travel",
        "Travel",
        "A child sitting in a small aeroplane window looking out at clouds and "
        "a tiny city below, suitcase in the overhead compartment, excited smile.",
    ),
    (
        "vocab_hobbies",
        "Hobbies",
        "A child surrounded by hobby items — a paintbrush and canvas, a football, "
        "a book, a guitar, and a camera — all floating around them joyfully.",
    ),
    (
        "vocab_clothes",
        "Clothes",
        "An open wardrobe with colourful clothing items hanging neatly — "
        "a dress, a coat, boots, a hat, a scarf — a child choosing an outfit "
        "with a big smile.",
    ),
    (
        "vocab_transport",
        "Transport",
        "A bright overhead-view city street with a bus, a bicycle, a car, "
        "a tram, and a child on a scooter — all moving along cheerfully "
        "in pastel traffic.",
    ),
    (
        "vocab_restaurant",
        "Restaurant",
        "Inside a small cozy restaurant — a child at a table being served "
        "a steaming bowl of soup by a smiling waiter, menu (blank) on the table, "
        "warm candle light.",
    ),
    (
        "vocab_friends",
        "Friends and Relationships",
        "Three children of different backgrounds laughing together, arms around "
        "each other's shoulders, confetti in the air, sunny park setting.",
    ),
    (
        "vocab_family",
        "Family",
        "A warm family portrait: grandparents, parents, and two children all "
        "sitting together on a sofa, a pet cat at their feet, cozy home interior.",
    ),
    (
        "vocab_technology",
        "Technology",
        "A child at a desk with a laptop (blank screen), a tablet, headphones, "
        "and a small robot toy — all surrounded by soft glowing lines suggesting "
        "connectivity.",
    ),
    (
        "vocab_holidays",
        "Holidays",
        "A festive scene with a decorated tree, wrapped gifts, lanterns, "
        "and two children dancing around a celebration — multicultural holiday "
        "symbols blended together.",
    ),
    (
        "vocab_sports",
        "Sports",
        "Four children each playing a different sport — kicking a football, "
        "swimming, shooting a basketball, doing gymnastics — in a sunny outdoor "
        "setting.",
    ),
    (
        "vocab_city_directions",
        "City and Directions",
        "A bird's-eye cartoon map of a small town with a child holding a large "
        "illustrated map, arrows showing paths to a park, a school, and a shop.",
    ),

    # ── Module banners ─────────────────────────────────────────────────────
    (
        "reading_banner",
        "Reading module banner",
        "A child lying on a soft rug reading a large open book, warm lamp-light "
        "spilling onto illustrated pages, cozy library shelves in the background, "
        "calm and inviting atmosphere.",
    ),
    (
        "my_phrases_banner",
        "My Phrases module banner",
        "A child writing in a beautiful notebook with a golden pen, surrounded "
        "by floating speech bubbles of different shapes and pastel colours, "
        "suggesting personal creative expression.",
    ),
]


# ---------------------------------------------------------------------------
# Integration reference — print at end so dev can copy-paste
# ---------------------------------------------------------------------------
VOCAB_SLUGS = {label: fname for fname, label, _ in IMAGES if fname.startswith("vocab_")}


def generate_images(api_key: str, start_from: int = 0, end_at: int | None = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key)

    subset = IMAGES[start_from: end_at]
    total = len(subset)
    generated = 0
    skipped = 0
    failed = []

    print(f"\nGenerating {total} images → {OUTPUT_DIR}\n")

    for idx, (slug, label, scene) in enumerate(subset, 1):
        filename = f"{slug}.png"
        output_path = OUTPUT_DIR / filename

        if output_path.exists():
            print(f"[{idx}/{total}] {slug}  ⏭  already exists, skipping")
            skipped += 1
            continue

        print(f"[{idx}/{total}] {slug}  Generating: {label}...")

        try:
            response = client.images.generate(
                model="gpt-image-1",
                prompt=_p(scene),
                size="1024x1024",
                quality="medium",
                n=1,
            )

            img_obj = response.data[0]
            if hasattr(img_obj, "b64_json") and img_obj.b64_json:
                img_data = base64.b64decode(img_obj.b64_json)
            else:
                img_data = requests.get(img_obj.url, timeout=30).content

            with open(output_path, "wb") as f:
                f.write(img_data)

            generated += 1
            print(f"          ✓ Saved {filename}")

            # gpt-image-1 rate limit: ~5 req/min on standard tier
            if idx < total:
                time.sleep(13)

        except Exception as e:
            err = str(e)
            print(f"          ✗ FAILED: {err[:100]}")
            failed.append({"slug": slug, "error": err})
            time.sleep(5)

    # Summary
    print(f"\n{'='*55}")
    print(f"Done!  Generated: {generated}  |  Skipped: {skipped}  |  Failed: {len(failed)}")

    if failed:
        print("\nFailed images:")
        for item in failed:
            print(f"  {item['slug']}: {item['error'][:70]}")
        failed_path = Path(__file__).parent / "failed_app_images.json"
        import json
        with open(failed_path, "w") as fp:
            json.dump(failed, fp, indent=2)
        print(f"\nFailed list saved to: {failed_path}")

    print(f"\nAll images saved to:  {OUTPUT_DIR}")
    print(
        "\nIntegration tip — in your Streamlit app:\n"
        "  APP_IMG_DIR = ROOT / 'static' / 'app_images'\n"
        "  # Vocabulary topic image:\n"
        "  img = APP_IMG_DIR / 'vocab_greetings.png'\n"
        "  # Reading banner:\n"
        "  img = APP_IMG_DIR / 'reading_banner.png'\n"
        "  # My Phrases banner:\n"
        "  img = APP_IMG_DIR / 'my_phrases_banner.png'\n"
    )
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate IMLLS app images with gpt-image-1")
    parser.add_argument("--api-key", required=True, help="OpenAI API key (sk-proj-...)")
    parser.add_argument(
        "--start-from", type=int, default=0,
        help="0-based index to start from (default: 0 = first image)",
    )
    parser.add_argument(
        "--end-at", type=int, default=None,
        help="0-based index to stop before (default: all images)",
    )
    args = parser.parse_args()

    generate_images(
        api_key=args.api_key,
        start_from=args.start_from,
        end_at=args.end_at,
    )
