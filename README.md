# Local Lecture Translation Pipeline

A local workflow for transcribing Urdu/Arabic Islamic lectures, translating them into
English, and inserting matching Arabic source passages into the translated text.

The pipeline uses:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcription
- [Ollama](https://ollama.com/) for running `gemma4:12b` locally
- FFmpeg and FFprobe for audio conversion, chunking, and duration detection
- Python's `difflib` for checking whether Arabic placement changed the English text

## Requirements

- Python 3
- FFmpeg and FFprobe
- Ollama with `gemma4:12b` installed
- `faster-whisper`

Install the Python dependency:

```bash
python3 -m pip install faster-whisper
```

Download the model through Ollama:

```bash
ollama pull gemma4:12b
```

## Usage

Convert an MP3 or M4A lecture to WAV:

```bash
python3 convert.py
```

Transcribe and translate the WAV file:

```bash
python3 transcribe.py lecture.wav
```

This produces:

- `lecture.transcript.txt`
- `lecture.english.txt`

To insert Arabic passages, prepare a text file containing one logical Arabic passage per
block, separated by blank lines. Then call the placement function:

```bash
python3 -c 'from transcribe import build_arabic_interleaved; build_arabic_interleaved("lecture.english.txt", "lecture.arabic_paragraphs.txt", "lecture.arabic_english.txt")'
```

The placement stage runs up to three candidates per English batch, keeps the candidate
with the lowest word-level diff, rejects invented paragraph markers, and removes placed
Arabic passages from the remaining pool to prevent duplicate insertion.

## Important limitation

The automated checks protect the structure of the placement output. They do not prove
that the translation is factually correct. Review translations and Arabic placements
against the original sources before publishing them.

## Privacy

The `.gitignore` intentionally excludes all files except the two Python scripts and this
README. Audio, transcripts, translations, generated outputs, blogs, and session logs are
kept out of the public repository.
