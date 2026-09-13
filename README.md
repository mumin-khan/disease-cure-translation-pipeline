# Local Lecture Translation Pipeline

A local Python workflow for transcribing Urdu/Arabic Islamic lectures, translating them
into English, and inserting matching passages from an Arabic source text.

It was built for a lecture series based on Ibn al-Qayyim's *Al-Da' wal-Dawa'*, but the
workflow can be adapted to similar long-form transcription and translation projects.

## How it works

```text
Audio → Whisper transcription → Gemma translation → Arabic placement → Review
```

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcribes 30-second audio
  chunks locally.
- [Ollama](https://ollama.com/) runs `gemma4:12b` for translation and Arabic placement.
- FFmpeg and FFprobe handle audio conversion, chunking, and duration detection.
- Python's `difflib` checks whether the placement step unexpectedly changed the English
  text.

## Requirements

- Python 3
- FFmpeg, including FFprobe
- [Ollama](https://ollama.com/) running locally
- `gemma4:12b`
- `faster-whisper`

Install the Python dependency:

```bash
python3 -m pip install faster-whisper
```

Pull the local model:

```bash
ollama pull gemma4:12b
```

Make sure Ollama is running before starting the pipeline:

```bash
ollama serve
```

## Quick start

### 1. Convert the lecture to WAV

For an MP3 or M4A input, run:

```bash
python3 convert.py
```

The script asks for the source path and creates a WAV file beside it. If the lecture is
already a WAV file, skip this step.

### 2. Transcribe and translate

```bash
python3 transcribe.py lecture.wav
```

This creates:

```text
lecture.transcript.txt   # one Whisper chunk per line
lecture.english.txt      # one translated batch per line
```

Both stages support resuming. If an output file already contains completed chunks or
batches, the script continues from the next one.

### 3. Prepare the Arabic passages

Create `lecture.arabic_paragraphs.txt`. Put each logical source passage in its own block
and separate blocks with a blank line:

```text
First Arabic passage...

Second Arabic passage...

Third Arabic passage...
```

Keep one complete hadith, quotation, or section in each block. The order should match
the source book.

### 4. Insert the Arabic passages

Arabic placement is a separate step; `transcribe.py lecture.wav` does not call it
automatically. Run:

```bash
python3 - <<'PY'
from transcribe import build_arabic_interleaved

build_arabic_interleaved(
    "lecture.english.txt",
    "lecture.arabic_paragraphs.txt",
    "lecture.arabic_english.txt",
)
PY
```

The result is written to `lecture.arabic_english.txt`.

## Placement safeguards

For each English batch, the placement stage:

1. Offers only Arabic passages that have not already been placed.
2. Allows the model to decide that none of the passages belong in the batch.
3. Generates up to three candidates.
4. Scores unexpected word-level changes to the English text.
5. Keeps the candidate with the lowest diff score.
6. Rejects marker numbers that were not present in the input.
7. Removes placed passages from the pool so they cannot be inserted again later.

A nonzero best score produces a warning for manual review. Passages that remain
unplaced after the final batch are also reported.

## Default configuration

| Setting | Value |
|---|---|
| Whisper model | `small` |
| Whisper device | CPU with int8 computation |
| Audio chunk size | 30 seconds |
| Translation model | `gemma4:12b` |
| Translation batch size | About 6,000 characters |
| Maximum generated tokens | 4,000 |
| Placement attempts | Up to 3 per batch |
| Ollama endpoint | `http://localhost:11434/api/chat` |

These values are defined near the top of `transcribe.py` and can be changed for other
hardware or models. The translation prompt is specialized for faithful Urdu/Arabic to
English translation and should be reviewed before using the code for another domain.

## Limitations

The automated checks protect the structure of the Arabic-placement output. They do not:

- prove that the English translation is accurate;
- detect every hallucination, omission, or repetition in the translation;
- prove that an Arabic passage was placed in the correct context; or
- replace review by someone who understands the source material.

Treat the generated files as drafts. Verify the translation and Arabic placement against
the original sources before publishing them.

## Privacy

The repository's `.gitignore` uses an allowlist. Only `transcribe.py`, `convert.py`,
`README.md`, and `.gitignore` are tracked. Audio, transcripts, translations, generated
outputs, blogs, and session logs remain excluded.
