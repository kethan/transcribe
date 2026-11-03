# Verification performed before implementation

Date: **2026-11-03**. This document records the actual external contracts that
were inspected, the compatibility conclusions, and how the three systems are
synchronised. It is the "before writing code" part of the implementation
approach. Section 1a was added after the user directed the pipeline to use the
Bodhan Hugging Face model instead of waiting for the hosted cloud API; it is
verified against the official model card and repository metadata before the
local backend was implemented.

## 1. Bodhan transcription: local Hugging Face model (default) and optional cloud API

The delivery now **defaults to the local Bodhan Hugging Face model**, so a run does
not wait for a hosted-API key. The hosted Bodhan API is retained as an explicitly
selected fallback (`--asr-backend cloud`). Both backends write the same cached
`{"text": ...}`-compatible chunk response, so diarization and alignment are
unchanged.

### 1a. Local HF model (default)

Official references inspected:

- [`bodhan-ai/indic-transcribe-core` model card](https://huggingface.co/bodhan-ai/indic-transcribe-core)
- [Model file list / custom Python code](https://huggingface.co/bodhan-ai/indic-transcribe-core/tree/main)
- [HF model API metadata](https://huggingface.co/api/models/bodhan-ai/indic-transcribe-core)

Verified facts:

| Item | Value |
| --- | --- |
| Model | `bodhan-ai/indic-transcribe-core` (gated; accept license, then `HF_TOKEN`) |
| Architecture | 1.2B Canary-1B-v2 FastConformer encoder + Transformer decoder, custom `indic_canary` code |
| Input | 16 kHz mono WAV/FLAC/MP3; resampled automatically; **single speaker** |
| Output | Native-script text string; optional detected-language tag |
| Languages | 25 Indian languages + English; code-mixing and automatic LID supported |
| Chunk limit | trained on ≤ 30 s; a single `asr(...)` pass refuses > 45 s; official long-form helper cuts at natural pauses to ≤ 30 s |
| Local API | `snapshot_download(...)`, add snapshot dir to `sys.path`, `from indic_transcribe import IndicTranscribe`, `IndicTranscribe.from_pretrained(model_dir, device=...)`, `asr.transcribe(path, lang="gu", return_lid=True)` |
| Dependencies | `torch`, `torchaudio`, `transformers`, `sentencepiece`, `soundfile`; **no NeMo needed for ASR** (NeMo is still used for diarization) |
| License | Indic Open Model License v1.0; commercial self-hosting allowed with attribution/conditions |

The model is **not** a standard `transformers.pipeline()` checkpoint: it ships
`indic_transcribe.py`, `modeling_indic_canary.py`, and dual SentencePiece tokenizers
that must be loaded from a real local directory. The implementation therefore uses
`snapshot_download(..., allow_patterns=...)` (or `--hf-asr-model-dir`) and the
model's own wrapper rather than inventing an `AutoProcessor` call. The allow-list
downloads only the HF inference files and skips the duplicate `nemo/*.nemo`
checkpoint in the same repo. It calls `transcribe()` once per
silence-aware ≤ 28 s chunk, keeps each chunk's absolute original-timeline offsets,
and reuses the model repo's `long_form` repetition guard when available. A
repetition loop splits the chunk at a quiet boundary up to two levels, then collapses
repeats with an explicit warning. No evenly distributed timestamps are created.

**Important:** the local model's documented output is still a **text string**, not
word timestamps. CTC forced alignment remains necessary for word start/end times.
Its single-speaker / native-script limitations also remain: overlapping words may
be merged into one ASR stream, and English words inside Indic speech are rendered
in the native script.

### 1b. Hosted Bodhan API (optional)

Official references inspected:

- [Bodhan API Reference — Speech to Text](https://console.bodhan.ai/api-docs/)
- [Bodhan quickstart / OpenAI-compatible `/v1` routes](https://console.bodhan.ai/)
- [Official Bodhan-AI Cookbook speech-to-text notebook](https://github.com/Bodhan-AI/Bodhan-AI-Cookbook/blob/main/getting-started/speech-to-text/speech_to_text.ipynb)

Verified request contract:

| Item | Verified value |
| --- | --- |
| Endpoint | `POST https://api.bodhan.ai/v1/audio/transcriptions` |
| Auth | `Authorization: Bearer $BODHAN_API_KEY` |
| Content type | `multipart/form-data` |
| Required fields | `file`, `model=indic-transcribe` |
| Optional field | `language` (ISO code such as `gu`, `hi`, `en`) |
| Audio limit | up to **30 seconds per request**; longer audio is rejected |
| Documented 200 response | `{"text": "..."}` — a single required string |
| 415 | audio duration could not be measured / container not metered; send WAV (FLAC/OGG/MP3 also accepted) |
| Errors | JSON `{"error": {"message", "code", "request_id"}}`; `x-ratelimit-*` headers are present on other routes |
| Billing | per second; docs recommend trimming silence |

**Important finding:** the published 200 schema contains only `text`. It does
**not** document `segments`, `words`, word timestamps, `response_format`, or
`timestamp_granularities`. Therefore the optional cloud backend:

1. sends only the documented `file`, `model`, and `language` fields;
2. splits long recordings into silence-aware ≤ 28 s WAV chunks (conservative
   margin below the 30 s limit), with exact original-timeline offsets;
3. retries 429/5xx/network failures with exponential backoff and honours
   `Retry-After`;
4. recursively splits a chunk if the gateway returns 413/duration-limit;
5. parses the documented `text` response;
6. *defensively* recognises extra `words` / `segments[].words` timestamps if a
   Bodhan deployment returns them anyway, but never asks for undocumented
   fields and never invents missing times.

The official cookbook uses the same `resp.json()["text"]` shape and shows fixed
25-second chunking for long audio. The local model card also states
"Single speaker" and 16 kHz mono input; this is an explicit limitation for
overlapping multi-speaker meetings: Bodhan may merge or miss overlapping words.
We do not replace its text with Whisper or another ASR. The default local
backend avoids `BODHAN_API_KEY` entirely, but the gated HF repo still requires
one-time license acceptance plus `HF_TOKEN` (or a supplied local snapshot).

## 2. NVIDIA NeMo diarization

"Neomotron" is interpreted as **NVIDIA NeMo**, not a separate endpoint. The
chosen pretrained model is
[`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization),
loaded through NeMo's `SortformerEncLabelModel`.

Verified from the model card, the [NVIDIA blog](https://huggingface.co/blog/nvidia/nemotron-diarization),
[NeMo Speech docs](https://docs.nvidia.com/nemo/speech/nightly/), the
`nemo-toolkit==3.0.0` wheel source, and the pinned `NVIDIA-NeMo/Speech` main
commit `e5d79d5a49cbec49d79440b0070aac0c32a8dbb8`:

| Item | Verified value |
| --- | --- |
| Runtime | NeMo Speech built from the pinned GitHub main commit; **not** PyPI `nemo-toolkit[asr]==3.0.0` |
| Why not PyPI 3.0.0 | Nemotron-3-Diarization uses a 31-layer RoPE encoder, but PyPI 3.0.0's `TransformerEncoder` accepts only `abs_pos`, `rel_pos`, `no_pos`. Instantiation fails with `self_attention_model='rope' is not supported`. The pinned GitHub commit adds `rope` to `_SUPPORTED_SELF_ATTENTION_MODELS`. |
| Python / torch | documentation requires Python 3.12+ and PyTorch 2.7+; GPU/CUDA recommended |
| Model class | `nemo.collections.asr.models.SortformerEncLabelModel` |
| Load | `SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization")`; HF token may be required |
| Audio | 16 kHz, single-channel `.wav`/`.flac`/`.opus`/`.mp3`; chunked inference has no hard duration limit |
| Speakers | up to 8; IDs are arrival-ordered generic channels, **not** voice identity |
| Overlap | per-speaker activity intervals can overlap; all reported DER evaluations include overlapping speech |
| Frame | 10 ms output resolution, 80 ms encoder frames |
| Offline config (80 ms frames) | `spkcache_len=264`, `fifo_len=40`, `chunk_len=340`, `chunk_right_context=40`, `spkcache_update_period=300`; call `_check_streaming_parameters()` |
| `diarize()` return | a list of lists of strings such as `"0.500 3.120 speaker_0"`; the model card also describes `(begin_seconds, end_seconds, speaker_index)` |

`verify_environment()` now inspects the installed `TransformerEncoder` and
fails early when `rope` is missing, before downloading/running the ASR chunks.

The implementation parses both shapes, clips intervals to the recording
duration, keeps overlapping intervals from different speakers, and maps the
first-arrival speaker channels to stable `speaker_0`, `speaker_1`, ... IDs. It
does **not** silently substitute `pyannote`, `whisperx.diarize`, or another
library. The older NeMo clustering/MSDD pipeline
([`diar_msdd_telephonic`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_msdd_telephonic))
is not used because its RTTM-style clustering output is not the overlapping
per-speaker activity representation needed here.

## 3. Word-level alignment and the Gujarati Wav2Vec2 model

Because Bodhan's documented response has no word timestamps, word times must
come from forced alignment of Bodhan's own transcript against the audio. No
timestamps are ever distributed evenly across a sentence.

### 3.1 Evaluation of `gchhablani/wav2vec2-large-xlsr-gu`

Inspected files:
[model card](https://huggingface.co/gchhablani/wav2vec2-large-xlsr-gu),
[`config.json`](https://huggingface.co/gchhablani/wav2vec2-large-xlsr-gu/raw/main/config.json),
[`vocab.json`](https://huggingface.co/gchhablani/wav2vec2-large-xlsr-gu/raw/main/vocab.json),
[`tokenizer_config.json`](https://huggingface.co/gchhablani/wav2vec2-large-xlsr-gu/raw/main/tokenizer_config.json),
[`preprocessor_config.json`](https://huggingface.co/gchhablani/wav2vec2-large-xlsr-gu/raw/main/preprocessor_config.json).

| Item | Verified value |
| --- | --- |
| Architecture | `Wav2Vec2ForCTC`, 16 kHz input |
| Output | 80 CTC columns; `config.pad_token_id=79`, `vocab_size=80` |
| Blank | `[PAD]` (id 79); `[UNK]` is id 78 |
| Tokenizer | `Wav2Vec2CTCTokenizer`, `word_delimiter_token="|"`, `do_lower_case=false` |
| Vocabulary | Gujarati letters/marks, Gujarati digits, ZWJ/ZWNJ, `|`, `[UNK]`, `[PAD]` |
| Missing | Latin letters, Devanagari letters, ASCII digits, punctuation marks |
| Training | OpenSLR SLR78 Gujarati read speech; self-reported WER 23.55%; evaluation strips punctuation and lowercases |
| Preprocessor | `do_normalize=true`, `sampling_rate=16000`, `return_attention_mask=true` |

**Conclusion:** the model is a valid character-level CTC aligner for
**Gujarati-only, punctuation-stripped** text. It is incompatible with the
required mixed Gujarati/Hindi/English transcript: Devanagari, Latin script,
ASCII digits and punctuation are out-of-vocabulary. Mapping them to `[UNK]` or
dropping them would not be accurate forced alignment. The pipeline exposes it
as `--alignment-model gujarati-wav2vec2` and `--alignment-model auto`; in
`gujarati-wav2vec2` mode an OOV character is a hard, reported error rather
than a silent substitution.

### 3.2 Why WhisperX is not the answer

Inspected [WhisperX `alignment.py`](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)
and the [torchaudio CTC forced-alignment tutorial](https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html).

WhisperX's alignment loader accepts `Wav2Vec2ForCTC` checkpoints, but it builds
a **character dictionary** from `Wav2Vec2Processor`, lowercases characters,
replaces spaces with `|`, and has no default Gujarati alignment model (`gu` is
absent from `DEFAULT_ALIGN_MODELS_HF`). For unknown characters it appends a
wildcard column whose score is the maximum non-blank emission score. That is a
fallback for unmatched symbols, not a reliable timestamp for Hindi/English
words. Therefore "it is a Wav2Vec2 model" is not sufficient for WhisperX
compatibility with a code-mixed Bodhan transcript. WhisperX is not used here.

### 3.3 Technically valid alternative implemented by default

Default: [`MahmoudAshraf/mms-300m-1130-forced-aligner`](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)
plus [`uroman`](https://pypi.org/project/uroman/) romanization.

Verified from its `config.json`, `tokenizer_config.json` and `vocab.json`:

- `Wav2Vec2ForCTC`, 16 kHz, `vocab_size=31`, `pad_token_id=0`;
- `<blank>` is id 0; the remaining columns are `a-z` and apostrophe;
- `word_delimiter_token=null`; the checkpoint is a conversion of the MMS
  `ctc_alignment_mling_uroman` forced-alignment model;
- the [ctc-forced-aligner source](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
  romanizes native-script text with uroman, normalizes to `[a-z' ]`, appends a
  synthetic `<star>` column, and maps character spans back to the **original**
  words.

uroman was installed and tested locally:

```text
આત્મા જ્ઞાતા છે.       -> aatmaa jnyaataa che.
आत्मा ज्ञाता है।       -> aatmaa jnyaataa hai.
सम्यक्त्व ના પચ્ચીસ દોષ  -> samyaktva naa pacciis doss ...
```

This gives one CTC alphabet that covers Gujarati, Devanagari and Latin script
after romanization. The original Bodhan word strings are preserved; only the
*acoustic target sequence* is romanized. A self-contained NumPy Viterbi CTC
forced aligner (with `<star>` edge tokens and midpoint blank splitting) maps
aligned character frames back to original words. This avoids the unrelated
PyPI package named `ctc-forced-aligner` (the PyPI project is a different
Deskpai package) and avoids a C++/torchaudio dependency that may conflict with
NeMo 3.0 / torchaudio 2.9 deprecations.

**License note:** the MMS forced-alignment checkpoint is CC-BY-NC 4.0. For
commercial use, replace it with a suitably licensed multilingual CTC model
whose tokenizer covers the romanized character set, and pass its id with
`--mms-model`.

Tokens with no alignable characters (for example a standalone ASCII `11`) are
merged into an adjacent word with an explicit warning; no zero-duration or
interpolated timestamp is emitted. If CTC alignment cannot produce a valid
interval for every word, the run fails and writes a failure report.

## 4. Timeline synchronisation

All three systems are anchored to one cached **16 kHz mono PCM16 WAV** derived
from the original recording without modifying it:

```text
original WAV/MP3
   └─ ffmpeg/copy -> work/audio/<hash>/audio_16k_mono.wav   (t = 0 baseline)
        ├─ Bodhan HF local (default) / cloud (optional):
        │    silence-aware chunks [start_sample, end_sample)
        │    text per chunk, with absolute start/end seconds preserved
        ├─ NeMo: whole canonical WAV -> overlapping speaker intervals in seconds
        └─ CTC: each chunk/segment waveform -> character frames
             frame times + chunk start = absolute word start/end
                  └─ greatest-overlap speaker assignment
                       └─ grouped segments + validation + UTF-8 JSON
```

Word timestamps are CTC frame spans (approximately 20 ms resolution), never
estimated from sentence duration. NeMo intervals may overlap; the assignment
sums each speaker's merged interval overlap with each word, ranks by
`(-overlap, earliest interval start, speaker_id)`, and returns `null` when the
winner covers less than `--speaker-min-coverage` or the runner-up is within
`--speaker-ambiguity-margin`. Segment grouping breaks on speaker change,
Bodhan chunk/segment boundary, meaningful pause, sentence-ending punctuation,
or `--max-segment-seconds`; it never creates one segment per word.

## 5. Explicit limitations

- Neither the local Bodhan HF model nor the hosted API documents a
  word-timestamped response. The optional `words`/`segments` parser is
  defensive only; CTC forced alignment is the normal timestamp source.
- The local Bodhan HF repo is **gated**: accept the Indic Open Model License on
  its model page and set `HF_TOKEN`, or provide `--hf-asr-model-dir` with an
  already-downloaded snapshot. No Bodhan cloud API key is needed by default.
- Both Bodhan ASR paths describe single-speaker input. Overlapping speech may
  be merged into one ASR stream; NeMo can detect overlapping speaker activity
  but cannot separate words that Bodhan did not transcribe separately.
- The local model writes native-script text only and renders English words
  inside Indic speech in the native script. Automatic LID is uneven for
  Hindi-belt languages; pass `--language` when known.
- Nemotron-3-Diarization supports up to 8 speakers; more speakers may be
  missed or confused. Speaker IDs are consistent within one recording, not
  identities across recordings.
- The default MMS aligner is CC-BY-NC 4.0; check licensing for commercial use.
- The bundled `audio/*.mp3` and `audio/*.json` in this repository are **not**
  Bodhan API/HF model responses. The JSON contains `spacing`, `audio_event`,
  `logprob`, `characters`, and `channel_index` fields (an ElevenLabs Scribe 2
  schema) and Hindi text. `tools/scribe2_reference.py` converts it to the
  strict schema only as a later verification reference; it is never used as a
  source of Bodhan text, timestamps, or speaker IDs for a pipeline run.
