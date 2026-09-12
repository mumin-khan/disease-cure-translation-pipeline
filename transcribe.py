import os
import re
import sys
import json
import difflib
import subprocess
import urllib.request
from faster_whisper import WhisperModel

# --- Config ---
CHUNK_SECONDS = 30        # Whisper is designed around ~30s windows; shorter chunks hurt accuracy
WHISPER_MODEL = "small"   # tiny/base/small/medium/large-v3; small is a good speed/accuracy balance on CPU/M-series
LANGUAGE = None           # e.g. "ur", "hi", "en" — None lets Whisper auto-detect per chunk

OLLAMA_URL = "http://localhost:11434/api/chat"
TRANSLATE_MODEL = "gemma4:12b"
# gemma4:12b has a 32768-token context window shared by prompt + input + output.
# Urdu/Arabic-script text runs ~2-3 chars/token, so keep well under the limit per call
# while still giving the model enough surrounding context for coherent translation.
TRANSLATE_CHARS_PER_CALL = 6000
TRANSLATE_PROMPT = (
    "You are translating an Urdu/Arabic-script transcript of an Islamic lecture into English. "
    "This is a literal, faithful translation task, not a paraphrase or summary task.\n"
    "Rules:\n"
    "1. Translate sentence by sentence, in the same order as the source. Do not skip, merge, "
    "compress, summarize, or invent content. Every sentence in the source must have a "
    "corresponding sentence in the output.\n"
    "2. Do not add any content, imagery, examples, explanation, book titles, names, or citations "
    "that are not explicitly present in the source text. If you are not certain of a name or "
    "title, transliterate what is said rather than substituting a real book/person you recall "
    "from outside knowledge.\n"
    "3. Keep names and honorifics consistent throughout: always render as \"Imam Ibn al-Qayyim "
    "(may Allah have mercy on him)\", \"Rasulullah (peace and blessings of Allah be upon him)\", "
    "\"Dua\" for supplication, etc. Never alter, abbreviate, or misspell a proper name once "
    "introduced.\n"
    "4. If a word or phrase is unclear or garbled in the source, translate your best literal "
    "interpretation of the meaning — never leave it in Urdu/Arabic script or transliterated Roman "
    "Urdu, and never replace it with unrelated invented content. The output must be entirely in "
    "English.\n"
    "5. Do not output any preamble, headers, numbered lists, or meta-commentary like \"Here is the "
    "translation\". Output ONLY the translated English prose, nothing else.\n\n"
    "Text to translate:\n\n"
)


def get_duration(wav_path):
    """Return audio duration in seconds using ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def make_chunk(wav_path, start, length, out_path):
    """Extract one chunk with ffmpeg, resampled to 16kHz mono."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-ss", str(start), "-t", str(length),
         "-i", wav_path,
         "-ar", "16000", "-ac", "1",
         out_path],
        check=True,
    )


def main():
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        wav_path = input("Enter the path to the .wav file: ").strip()
    wav_path = wav_path.replace('"', "").replace("'", "")

    if not os.path.exists(wav_path):
        print(f"Error: File '{wav_path}' does not exist.")
        return

    base = os.path.splitext(wav_path)[0]
    out_txt = base + ".transcript.txt"
    en_txt = base + ".english.txt"
    chunk_tmp = base + ".chunk.wav"

    duration = get_duration(wav_path)
    total_chunks = int(duration // CHUNK_SECONDS) + (1 if duration % CHUNK_SECONDS else 0)
    print(f"Audio length: {duration:.1f}s  ->  {total_chunks} chunk(s) of {CHUNK_SECONDS}s")

    print(f"Loading Whisper model '{WHISPER_MODEL}'...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    # Resume support: one line per chunk in the output file
    done = 0
    if os.path.exists(out_txt):
        with open(out_txt, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
        if done:
            print(f"Resuming: {done} chunk(s) already in {out_txt}")

    with open(out_txt, "a", encoding="utf-8") as out:
        for i in range(done, total_chunks):
            start = i * CHUNK_SECONDS
            length = min(CHUNK_SECONDS, duration - start)
            make_chunk(wav_path, start, length, chunk_tmp)

            print(f"[{i + 1}/{total_chunks}] {start:.0f}-{start + length:.0f}s ...",
                  end=" ", flush=True)

            segments, _ = model.transcribe(chunk_tmp, language=LANGUAGE)
            text = " ".join(seg.text.strip() for seg in segments).strip()

            out.write(text.replace("\n", " ") + "\n")
            out.flush()
            print(text if text else "(empty)")

    if os.path.exists(chunk_tmp):
        os.remove(chunk_tmp)
    print(f"\nDone. Transcript saved to: {out_txt}")

    translate_to_english(out_txt, en_txt)


TRANSLATE_MAX_TOKENS = 4000
# Guards against runaway/looping generation (observed with qwen3.5:9b): without a cap,
# a model stuck in a repetition loop keeps decoding until it fills the context window,
# taking minutes and producing no valid output instead of failing fast.


def translate_text(text):
    """Send text to TRANSLATE_MODEL and return the English translation."""
    payload = {
        "model": TRANSLATE_MODEL,
        "stream": False,
        "keep_alive": "15m",
        "think": False,  # skip reasoning trace on thinking models (e.g. qwen3.5) — otherwise
                          # the whole token budget can be consumed by <think> content, leaving
                          # nothing for the actual translation
        "options": {"num_predict": TRANSLATE_MAX_TOKENS},
        "messages": [{"role": "user", "content": TRANSLATE_PROMPT + text}],
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=2400) as r:
        resp = json.loads(r.read())
    return resp["message"]["content"].strip()


def group_lines(lines, max_chars):
    """Group transcript lines into batches under max_chars each, without splitting
    mid-sentence. Each line is one 30s chunk's transcript, so grouping several
    together gives gemma real surrounding context while staying under its
    context window."""
    groups = []
    current = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            groups.append(current)
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len
    if current:
        groups.append(current)
    return groups


def translate_to_english(out_txt, en_txt):
    """Translate the saved transcript to English in a handful of large batches via
    gemma4:12b, each batch staying well under the model's context window so nothing
    gets truncated, while still giving far more context than per-30s-chunk translation."""
    with open(out_txt, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    groups = group_lines(lines, TRANSLATE_CHARS_PER_CALL)
    total = len(groups)
    print(f"\nTranslating to English -> {total} batch(es) "
          f"(~{TRANSLATE_CHARS_PER_CALL} chars each)")

    done = 0
    if os.path.exists(en_txt):
        with open(en_txt, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
        if done:
            print(f"Resuming translation: {done} batch(es) already in {en_txt}")

    with open(en_txt, "a", encoding="utf-8") as out:
        for i in range(done, total):
            batch_text = " ".join(groups[i])
            print(f"[{i + 1}/{total}] translating ({len(batch_text)} chars) ...",
                  end=" ", flush=True)

            try:
                english = translate_text(batch_text)
            except Exception as e:
                print(f"\nError on batch {i + 1}: {e}")
                print("Progress saved. Re-run to resume from here.")
                break

            out.write(english.replace("\n", " ") + "\n")
            out.flush()
            print("done.")

    print(f"\nEnglish translation saved to: {en_txt}")


ARABIC_CHAR_RANGE = re.compile(r"[؀-ۿ]")

INTERLEAVE_RUNS_PER_BATCH = 3
# Even with a strict verbatim instruction, gemma4:12b occasionally substitutes a word
# (observed: "ailment" -> "allure", "al-Qayyim" -> "al-Qayim") in a small fraction of runs.
# Running several times and keeping the run with the fewest word-level diffs against the
# source catches this instead of trusting a single pass blindly.

INTERLEAVE_PROMPT_TEMPLATE = (
    "You are given (1) an English text and (2) a numbered list of Arabic paragraphs. Your task "
    "is a strict structural edit, NOT a translation or rewrite.\n\n"
    "IMPORTANT: most of the Arabic paragraphs probably do NOT belong in this particular English "
    "text — they are candidates pulled from a larger book, and each one belongs in only ONE place "
    "in the whole book, which may not be here. Only insert a paragraph if this specific English "
    "text is actually discussing that paragraph's exact content. It is normal and expected for "
    "ZERO paragraphs to belong here. Do NOT force a paragraph in just because none fit — err on "
    "the side of leaving a paragraph out if you are not confident it belongs in THIS text.\n\n"
    "TASK:\n"
    "1. Reproduce the ENTIRE English text below, split into one sentence per line, in the EXACT "
    "SAME WORDS AND ORDER as given. Copy each word exactly as spelled in the source — do not "
    "substitute synonyms, do not paraphrase, do not summarize, do not correct spelling, do not "
    "drop a single sentence or word.\n"
    "2. For each Arabic paragraph, decide independently whether it belongs in this specific "
    "English text. If yes, find the ONE point where the topic matches and insert that Arabic "
    "paragraph as its own block (wrapped in blank lines) at exactly that point, prefixed with its "
    "number in brackets like \"[3]\" on its own line before it, then continue with the English "
    "sentences immediately after, unchanged. If a paragraph does not belong, do not insert it at "
    "all — do not mention it, do not force it in anywhere.\n"
    "3. Do not translate the Arabic. Do not add any commentary, headers, or notes of your own "
    "beyond the \"[N]\" marker specified above.\n"
    "4. Output ONLY the resulting sequence: English sentences (one per line) with zero or more "
    "Arabic paragraphs inserted at their matching points, each preceded by its \"[N]\" marker. "
    "Nothing else — no preamble, no explanation.\n\n"
    "ARABIC PARAGRAPHS (numbered; most likely do NOT belong here):\n{arabic_block}\n\n"
    "ENGLISH TEXT (reproduce every sentence, EXACTLY as written, one per line, inserting only the "
    "Arabic paragraph(s) that truly belong here, each marked with its \"[N]\"):\n{english_text}\n"
)


def load_arabic_paragraphs(path):
    """Arabic paragraphs file: one or more paragraphs separated by blank lines, in the
    order they appear in the source book."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    return paragraphs


def format_arabic_block(paragraphs, available_numbers):
    """Render only the still-unplaced paragraphs as a numbered block for the prompt,
    keeping each paragraph's original 1-based index as its number."""
    return "\n\n".join(
        f"[{n}]\n{paragraphs[n - 1]}" for n in available_numbers
    )


PLACED_MARKER_RE = re.compile(r"^\[(\d+)\]\s*$", re.MULTILINE)


def find_placed_paragraph_numbers(candidate_text):
    """Return the set of paragraph numbers the model claims to have inserted, based on the
    '[N]' marker lines it was instructed to place immediately before each Arabic paragraph."""
    return {int(n) for n in PLACED_MARKER_RE.findall(candidate_text)}


def strip_to_english_words(text):
    """Return the word list of only the non-Arabic lines in a model's interleaved output,
    for diffing against the original all-English source batch."""
    words = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ARABIC_CHAR_RANGE.search(line):
            continue
        words.extend(line.split())
    return words


def diff_score(original_text, candidate_text):
    """Count word-level diff ops (insert/delete/replace) between the original English batch
    and the English portion of a candidate interleaved output. Zero means an exact match."""
    orig_words = original_text.split()
    cand_words = strip_to_english_words(candidate_text)
    sm = difflib.SequenceMatcher(None, orig_words, cand_words)
    return sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag != "equal"
    )


def call_ollama_interleave(prompt):
    payload = {
        "model": TRANSLATE_MODEL,
        "stream": False,
        "keep_alive": "15m",
        "think": False,
        "options": {"num_predict": TRANSLATE_MAX_TOKENS},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=2400) as r:
        resp = json.loads(r.read())
    return resp["message"]["content"].strip()


def interleave_batch(english_batch, paragraphs, available_numbers):
    """Run the interleave prompt INTERLEAVE_RUNS_PER_BATCH times, offering only the
    paragraphs in `available_numbers` (i.e. not yet placed in an earlier batch) as
    candidates. Returns (best_text, best_score, placed_numbers) — placed_numbers is the
    set of paragraph numbers the winning run actually inserted, which the caller must
    remove from the pool before processing the next batch so a paragraph can never be
    placed in more than one batch."""
    if not available_numbers:
        return english_batch, 0, set()

    arabic_block = format_arabic_block(paragraphs, available_numbers)
    prompt = INTERLEAVE_PROMPT_TEMPLATE.format(
        arabic_block=arabic_block, english_text=english_batch
    )
    best_text, best_score, best_placed = None, None, set()
    for run in range(1, INTERLEAVE_RUNS_PER_BATCH + 1):
        candidate = call_ollama_interleave(prompt)
        score = diff_score(english_batch, candidate)
        placed = find_placed_paragraph_numbers(candidate)
        bogus = placed - set(available_numbers)
        if bogus:
            # Model invented a marker number that wasn't offered — treat as a failed run,
            # never silently accept content we can't attribute to a real paragraph.
            print(f"    run {run}/{INTERLEAVE_RUNS_PER_BATCH}: diff_score={score}, "
                  f"placed={sorted(placed)} -- REJECTED, invented marker(s) {sorted(bogus)}",
                  flush=True)
            continue
        print(f"    run {run}/{INTERLEAVE_RUNS_PER_BATCH}: diff_score={score}, "
              f"placed={sorted(placed) if placed else 'none'}", flush=True)
        if best_score is None or score < best_score:
            best_text, best_score, best_placed = candidate, score, placed
        if best_score == 0:
            break  # exact match found, no need to burn more runs
    return best_text, best_score, best_placed


BATCH_SEPARATOR = "\n\n<<<BATCH_BOUNDARY>>>\n\n"
# Marks where one English batch's interleaved output ends and the next begins, so the file
# stays human-readable (real newlines, one sentence per line) while remaining resumable —
# resume count is the number of separators seen so far, not the number of lines.


def build_arabic_interleaved(en_txt, arabic_txt, out_txt):
    """For each English batch in en_txt, decide per-Arabic-paragraph whether it belongs in
    that batch and interleave it via gemma4:12b (verified against the source, see
    interleave_batch). Each paragraph can be placed in at most one batch total: once placed,
    it is removed from the pool offered to every subsequent batch, so it cannot be duplicated
    across batches even if its topic seems related in more than one place. Batches with no
    matching Arabic pass through unchanged. Resumable: reconstructs already-placed paragraph
    numbers by re-scanning out_txt's existing '[N]' markers, so a restart can't re-offer (and
    re-duplicate) a paragraph already placed in an earlier run."""
    with open(en_txt, "r", encoding="utf-8") as f:
        english_batches = [line.rstrip("\n") for line in f if line.strip()]

    arabic_paragraphs = load_arabic_paragraphs(arabic_txt)
    print(f"Loaded {len(arabic_paragraphs)} Arabic paragraph(s) from {arabic_txt}")
    available_numbers = set(range(1, len(arabic_paragraphs) + 1))

    done = 0
    if os.path.exists(out_txt):
        with open(out_txt, "r", encoding="utf-8") as f:
            existing = f.read()
        if existing.strip():
            done = existing.count(BATCH_SEPARATOR) + 1
            already_placed = find_placed_paragraph_numbers(existing)
            available_numbers -= already_placed
            print(f"Resuming: {done} batch(es) already in {out_txt}; "
                  f"paragraph(s) {sorted(already_placed) or 'none'} already placed")

    with open(out_txt, "a", encoding="utf-8") as out:
        for i in range(done, len(english_batches)):
            batch = english_batches[i]
            print(f"[{i + 1}/{len(english_batches)}] {len(available_numbers)} "
                  f"paragraph(s) still unplaced, checking for matches...")

            best_text, best_score, placed = interleave_batch(
                batch, arabic_paragraphs, available_numbers
            )
            if best_score != 0:
                print(f"    WARNING: best run for batch {i + 1} still has "
                      f"{best_score} word-level diff(s) from source. Kept anyway — review manually.")
            available_numbers -= placed
            if placed:
                print(f"    placed paragraph(s) {sorted(placed)} in this batch; "
                      f"{len(available_numbers)} remain unplaced.")

            if i > 0:
                out.write(BATCH_SEPARATOR)
            out.write(best_text)
            out.flush()
            print("    done.")

    if available_numbers:
        print(f"\nWARNING: paragraph(s) {sorted(available_numbers)} were never placed in any "
              f"batch — likely because the lecture hasn't reached that topic yet.")
    print(f"\nArabic-interleaved file saved to: {out_txt}")


if __name__ == "__main__":
    main()
