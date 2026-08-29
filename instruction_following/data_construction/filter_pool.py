"""Base-prompt filter pipeline (DESIGN.md §2.2, v2 implementation).

Six stages, cheapest first, each keeping a kill ledger so every rule can say
who it killed (stats.json -> paper appendix):

  S0 normalize + exact dedupe
  S1 degenerate content (empty / symbol soup / URL dump / code block / non-EN)
  S2 length window 5-60 words   [lower bound is evidence-backed, see below]
  S3 instruction-conflict (six families, strict tier + loose bare-noun tier)
  S4 refusal-risk (small conservative list; Step 0 refusal rate is the backstop)
  S5 per-task eligibility flags (no pool splitting — one pool, five bool flags)

The 5-word lower bound replaces DESIGN.md's original 10: analysis/len_correlation.py
measured real replies in dolly / oasst / wildchat and found the short-reply risk
sits in the 1-4 word bin, while 5-9 word prompts draw replies statistically
indistinguishable from 10+ (oasst median 117 vs 171 words, <=20-word replies
8.9% vs 3.9%). 10 -> 5 lifts oasst survival 61% -> 82%.

Run:
  .venv/bin/python instruction_following/data_construction/filter_pool.py
"""
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from kernel.config import DATA_CONSTRUCTION, RAW  # noqa: E402


MIN_WORDS, MAX_WORDS = 5, 60

# ---------------------------------------------------------------- S1 patterns
CODE_BLOCK = re.compile(r"```|<\?php|def \w+\(|#include\b|SELECT .* FROM ", re.I)
URL = re.compile(r"https?://|www\.\w+\.")
# a high share of non-ASCII characters means it is probably not English
NON_ASCII_RATIO = 0.10

# ---------------------------------------------------------------- S3 patterns
# Strict tier: action patterns that DEMAND a format. Six families; a hit is a
# genuine conflict with the instruction we will inject.
STRICT = {
    # A. output structure
    "structure": re.compile(
        r"\b(give|make|write|provide|create|generate|show)\s+(me\s+)?(a\s+|an\s+)?"
        r"(list|table|outline|summary|bullet)"
        r"|\bin\s+(a\s+)?(table|list|json|xml|markdown|csv)\b"
        r"|\bas\s+(a\s+)?(list|table|bullet|json)\b"
        r"|\bbullet\s*(point|list)|\bnumbered\s+list\b|\bwith\s+headings\b"
        r"|\bstep[-\s]by[-\s]step\b|\bin\s+steps\b", re.I),
    # B. length / level of detail
    "length": re.compile(
        r"\bin\s+(a\s+)?(one|two|three|four|five|\d+)\s+"
        r"(word|words|sentence|sentences|paragraph|paragraphs|line|lines|page|pages)\b"
        r"|\b(one|two|three|1|2|3)[-\s](sentence|paragraph|word)\b"
        r"|\b(be\s+)?(brief|concise|succinct)\b|\bin\s+short\b|\bkeep\s+it\s+short\b"
        r"|\bin\s+detail\b|\bdetailed\s+(explanation|answer|description)\b"
        r"|\bshort\s+(answer|response|reply)\b|\blong\s+(answer|response)\b"
        r"|\b(no\s+more|less|fewer)\s+than\s+\d+\s+(word|sentence|paragraph)", re.I),
    # C. sentence-initial imperatives - the biggest leak: these demand structure
    #    without any "give me" phrasing
    "imperative": re.compile(
        r"^\s*(list|name|enumerate|outline|summarize|summarise|translate|rewrite|"
        r"rephrase|convert|reword|abbreviate)\b", re.I),
    # D. language
    "language": re.compile(
        r"\btranslat(e|ion)\b|\bin\s+(french|spanish|german|chinese|japanese|korean|"
        r"russian|italian|portuguese|arabic|hindi|dutch|polish|swedish|turkish)\b"
        r"|\banswer\s+in\s+\w+ese\b", re.I),
    # E. casing / punctuation - collides directly with Tasks 1 and 5
    "casing": re.compile(
        r"\ball\s+caps\b|\bupper\s*case\b|\blower\s*case\b|\bcapital\s+letters?\b"
        r"|\bno\s+punctuation\b|\bwithout\s+(commas?|punctuation)\b", re.I),
    # F. genre / literary form
    "form": re.compile(
        r"\b(as|in\s+the\s+form\s+of)\s+(a\s+|an\s+)?"
        r"(poem|haiku|sonnet|song|rap|limerick|essay|letter|email|script|dialogue|story)\b"
        r"|\bwrite\s+(me\s+)?(a\s+|an\s+)?(poem|haiku|sonnet|song|rap|limerick)\b", re.I),
}
# Loose tier: bare nouns. With surplus prompts we accept over-killing, but it is
# ledgered separately so the false-kill rate and topic bias can be quantified.
LOOSE = re.compile(r"\b(list|table|bullet|json|markdown|paragraphs?)\b", re.I)

# ---------------------------------------------------------------- S4 patterns
REFUSAL_RISK = re.compile(
    r"\b(kill|suicide|bomb|explosive|weapon|gun|drug|cocaine|heroin|meth|"
    r"hack|exploit|malware|virus|steal|fraud|launder|porn|nsfw|sex|nude|"
    r"racist|nazi|slur|jailbreak|dan mode|ignore (all )?previous)\b", re.I)

# ---------------------------------------------------------------- S5 patterns
# Task 1 (all lowercase): code-inducing prompts and proper-noun-dense ones
T1_CODE = re.compile(r"\b(code|program|script|function|algorithm|syntax|python|"
                     r"java|sql|html|css|regex|api)\b", re.I)
# Task 4 (one sentence): prompts that plainly ask for a long answer
T4_LONG = re.compile(r"\b(essay|story|article|blog|report|novel|chapter|"
                     r"biography|speech|review)\b"
                     r"|\bwrite\s+(me\s+)?(a|an|the)\s+\w+\s+(about|on)\b", re.I)


def proper_noun_density(text: str) -> float:
    """Share of capitalised words after the first - a cheap proxy for proper-noun
    density, which is what makes an all-lowercase answer unnatural."""
    words = text.split()
    if len(words) < 2:
        return 0.0
    caps = sum(1 for w in words[1:] if w[:1].isupper() and w[1:2].islower())
    return caps / len(words)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def load_oasst() -> list:
    rows = []
    for src in ["oasst1", "oasst2"]:
        for line in open(RAW / src / "pool.jsonl", encoding="utf-8"):
            rows.append(json.loads(line))
    return rows


def run():
    rows = load_oasst()
    ledger = Counter()
    seen, kept = set(), []

    for r in rows:
        text = normalize(r["text"])
        ledger["S0_input"] += 1

        key = text.lower()
        if key in seen:
            ledger["S0_dup"] += 1
            continue
        seen.add(key)

        # S1 degenerate content
        if not text:
            ledger["S1_empty"] += 1
            continue
        if CODE_BLOCK.search(text):
            ledger["S1_code"] += 1
            continue
        if len(URL.findall(text)) >= 2:
            ledger["S1_url"] += 1
            continue
        non_ascii = sum(1 for c in text if ord(c) > 127) / len(text)
        if non_ascii > NON_ASCII_RATIO:
            ledger["S1_non_english"] += 1
            continue
        if sum(c.isalpha() for c in text) / len(text) < 0.5:
            ledger["S1_symbol_soup"] += 1
            continue

        # S2 length window
        n_words = len(text.split())
        if n_words < MIN_WORDS:
            ledger["S2_too_short"] += 1
            continue
        if n_words > MAX_WORDS:
            ledger["S2_too_long"] += 1
            continue

        # S3 instruction conflict (strict tier)
        hit = next((fam for fam, pat in STRICT.items() if pat.search(text)), None)
        if hit:
            ledger[f"S3_strict_{hit}"] += 1
            continue
        # S3 loose tier: bare nouns, ledgered separately
        if LOOSE.search(text):
            ledger["S3_loose_barenoun"] += 1
            continue

        # S4 refusal risk
        if REFUSAL_RISK.search(text):
            ledger["S4_refusal_risk"] += 1
            continue

        # S5 per-task eligibility flags (one pool, five flags - no splitting)
        t1_ok = not T1_CODE.search(text) and proper_noun_density(text) <= 0.15
        t4_ok = not T4_LONG.search(text)
        kept.append({
            "prompt_id": hashlib.sha1(text.encode()).hexdigest()[:12],
            "text": text,
            "n_words": n_words,
            "source": r["source"],
            "source_message_id": r["meta"].get("message_id"),
            "task_flags": {"task1": t1_ok, "task2": True, "task3": True,
                           "task4": t4_ok, "task5": True},
        })
        ledger["kept"] += 1

    DATA_CONSTRUCTION.mkdir(parents=True, exist_ok=True)
    pool_path = DATA_CONSTRUCTION / "oasst_pool_filtered.jsonl"
    with open(pool_path, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    stats = {"filter_version": code_hash, "source": "oasst1+2",
             "ledger": dict(ledger),
             "task_eligible": {f"task{i}": sum(r["task_flags"][f"task{i}"] for r in kept)
                               for i in range(1, 6)}}
    (DATA_CONSTRUCTION / "filter_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"filter version {code_hash}\n")
    for k, v in ledger.most_common():
        print(f"  {k:26s} {v:6d}")
    print(f"\nkept {ledger['kept']} / {ledger['S0_input']} = "
          f"{ledger['kept']/ledger['S0_input']:.1%}")
    print("per-task availability:", stats["task_eligible"])
    print(f"\n-> {pool_path}")


if __name__ == "__main__":
    run()
