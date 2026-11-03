# Gujarati meeting transcription + NeMo diarization + CTC alignment

A complete, reproducible Python pipeline that turns a Gujarati/Hindi/English
meeting recording into the exact requested `segments[].words[]` JSON:

- **Bodhan `indic-transcribe-core` from Hugging Face** supplies the transcript
  locally by default — no Bodhan cloud API key and no waiting for a hosted API.
- The hosted **Bodhan AI API** remains available only as an explicit
  `--asr-backend cloud` fallback.
- **NVIDIA NeMo `SortformerEncLabelModel` / Nemotron-3-Diarization** supplies
  overlapping speaker intervals.
- **CTC forced alignment** supplies real word start/end times when Bodhan's
  response does not include them.
- Every word gets a deterministic `speaker_id` or `null`.
- The final JSON is validated before export; a separate processing report is
  written.

> Read [`docs/VERIFICATION.md`](docs/VERIFICATION.md) first. It records the
> official Bodhan cloud request/response schema, the local Bodhan HF model
> contract, the NeMo 3.0 model/API compatibility check, the
> `gchhablani/wav2vec2-large-xlsr-gu` vocabulary evaluation, why WhisperX is
> not a valid mixed-script aligner here, the MMS + uroman alternative, and the
> shared-timeline synchronisation design.

## Kaggle: easiest path

Use **[`kaggle/transcribe.ipynb`](kaggle/transcribe.ipynb)**. It has two cells:
a GPU diagnostic and a main runner. The main cell clones this branch, installs
the dependencies while preserving Kaggle's CUDA `torch`/`torchaudio`, checks
your `HF_TOKEN`, prints `pipeline.py --check`, streams the real run to
`pipeline.log`, runs the bundled MP3, and writes all outputs to
`/kaggle/working/transcribe_outputs/`. See [`kaggle/README.md`](kaggle/README.md)
for the four Kaggle settings (GPU, Internet, Secret, import notebook). You do
not need to understand the other files to run it.

## What was verified before coding (summary)

1. **Local Bodhan HF model (default):** `bodhan-ai/indic-transcribe-core` is a
   gated 1.2B Canary/FastConformer ASR model with custom `indic_canary` Python
   code. It expects 16 kHz mono audio, accepts ≤ 30 s chunks, outputs a
   native-script **text string**, supports 25 languages plus code-mixing and
   language ID, and ships its own `IndicTranscribe` wrapper — it is not a normal
   `transformers.pipeline()` checkpoint. [Model card](https://huggingface.co/bodhan-ai/indic-transcribe-core) · [Files](https://huggingface.co/bodhan-ai/indic-transcribe-core/tree/main)
2. **Bodhan cloud API (optional):** `POST https://api.bodhan.ai/v1/audio/transcriptions`,
   multipart `file`, `model=indic-transcribe`, optional `language`, bearer auth,
   ≤ 30 s per request, documented 200 response `{"text": "..."}`. No documented
   `segments`, `words`, word timestamps, or `response_format` fields. [API docs](https://console.bodhan.ai/api-docs/) · [official notebook](https://github.com/Bodhan-AI/Bodhan-AI-Cookbook/blob/main/getting-started/speech-to-text/speech_to_text.ipynb)
3. **NeMo:** `SortformerEncLabelModel` loads
   `nvidia/Nemotron-3-Diarization`; it expects 16 kHz mono audio, supports up to
   8 speakers and overlapping activity, and uses the documented 30.4 s offline
   configuration. **PyPI `nemo-toolkit[asr]==3.0.0` cannot instantiate this
   checkpoint** because the checkpoint requires a RoPE encoder and that release
   supports only `abs_pos`, `rel_pos`, `no_pos`. The pinned `NVIDIA-NeMo/Speech`
   main commit in `requirements.txt` adds `rope` support. [Model card](https://huggingface.co/nvidia/Nemotron-3-Diarization) · [NVIDIA blog](https://huggingface.co/blog/nvidia/nemotron-diarization)
4. **Alignment:** Bodhan has no word timestamps, so forced alignment is
   required. `gchhablani/wav2vec2-large-xlsr-gu` has an 80-column Gujarati
   character vocabulary with blank `[PAD]` id 79 and no Latin, Devanagari,
   ASCII digits or punctuation. It is valid for pure Gujarati text only. The
   default aligner is `MahmoudAshraf/mms-300m-1130-forced-aligner` (31-column
   Latin CTC vocabulary, `<blank>` id 0) with `uroman` romanization; original
   Bodhan words are preserved and only the acoustic target characters are
   romanized. [Gujarati model](https://huggingface.co/gchhablani/wav2vec2-large-xlsr-gu) · [MMS aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) · [uroman](https://pypi.org/project/uroman/)
5. **Synchronisation:** one cached 16 kHz mono WAV is the time baseline. Bodhan
   chunks carry absolute sample offsets; NeMo runs on that same WAV; CTC frame
   spans are offset by their chunk start; speaker assignment uses greatest
   temporal overlap. See the diagram in `docs/VERIFICATION.md`.

## Repository layout

```text
pipeline.py            top-level launcher (python pipeline.py ...)
gujarati_pipeline/
  __main__.py        CLI and one-command entry point
  config.py          env/CLI config, cache digests, validation
  audio.py           ffmpeg/copy to canonical 16 kHz mono WAV, silence-aware chunks
  bodhan.py          optional documented cloud API client + response parser
  transcription.py   local HF ASR (default), optional cloud backend, shared cache
  diarization.py     NVIDIA NeMo Sortformer load/infer/parse/cache
  ctc.py             self-contained NumPy CTC Viterbi + word spans
  alignment.py       MMS+uroman / Gujarati CTC aligners, auto routing, cache
  assembly.py        speaker overlap assignment and segment grouping
  validation.py      final JSON schema/text/timestamp/speaker validation
  pipeline.py        7-stage orchestration, progress, reports, failure handling
tests/test_pipeline.py  offline unit tests (no API/GPU/model weights needed)
tools/scribe2_reference.py  strict-schema conversion + ElevenLabs Scribe 2 verification
docs/VERIFICATION.md   detailed API/model research and limitations
requirements.txt       pinned Python dependencies
.env.example           environment variable names (no secrets)
```

## Installation (Kaggle GPU, Python 3.12+)

NeMo Speech 3.0 requires **Python 3.12 or later** and **PyTorch 2.7+**. Use a
Python 3.12+ GPU Kaggle image / container. Kaggle normally ships a matching
CUDA-enabled `torch`; verify it rather than replacing it with a CPU wheel.
`torchaudio` is deliberately unpinned so `pip` cannot force a newer `torch`
that breaks that pair. The Kaggle notebook pins both packages with a generated
constraints file. Do **not** install `nemo-toolkit[asr]==3.0.0` from PyPI: it
lacks the RoPE encoder needed by Nemotron-3-Diarization. Use the pinned GitHub
main commit in `requirements.txt`.

```bash
# System audio tools
apt-get -qq update && apt-get -qq install -y ffmpeg libsndfile1

# In a Kaggle notebook use %pip instead of pip.
pip install -q --upgrade pip
pip install -q "numpy>=1.26,<3" "requests>=2.32,<3" "uroman==1.3.1.1"
pip install -q "huggingface_hub>=0.34,<2" "sentencepiece>=0.2,<0.3" "soundfile>=0.12,<0.14"
pip install -q "torchaudio" "transformers==4.57.6"
pip install -q "Cython" "packaging"
pip install -q "nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@e5d79d5a49cbec49d79440b0070aac0c32a8dbb8"

# Verify the preinstalled torch build:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Or from the repository root:

```bash
pip install -r requirements.txt
```

Do **not** install the unrelated PyPI package named `ctc-forced-aligner`: it
is a different Deskpai project. The CTC Viterbi aligner in this repository is
self-contained and uses only NumPy/PyTorch/Transformers/uroman.

### One-time gated-model setup

`bodhan-ai/indic-transcribe-core` is gated. The model page says to accept its
terms and then run `hf auth login`; the same `HF_TOKEN` is also used for the
Nemotron-3-Diarization checkpoint. A Bodhan cloud API key is **not** involved.

1. Log in to Hugging Face, then open
   <https://huggingface.co/bodhan-ai/indic-transcribe-core>.
2. Click **Agree and access repository**. Fill the gated form fields
   (`Company / Organization`, `Country`, `Intended Use Case`, and the license
   checkbox). The repo uses automatic approval, so access is normally immediate;
   if a manual review is pending, wait for the accepted state.
3. Create a **read** token at <https://huggingface.co/settings/tokens>. It must
   belong to the same account that accepted the gate. Do not paste it into chat
   or source files.
4. Provide it to the runtime:

```bash
# Local shell
export HF_TOKEN="hf_..."
hf auth login --token "$HF_TOKEN"     # optional; caches the token too

# Or Kaggle: Add-ons -> Secrets -> add secret named HF_TOKEN, then in a cell:
# import os
# from kaggle_secrets import UserSecretsClient
# os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

5. Verify access **before** starting the long MP3 run. This downloads only a
   config file and the small custom-code file, not the weights:

```bash
python -m tools.check_hf_access
```

The pipeline itself uses `snapshot_download(..., allow_patterns=...)` to fetch
only the HF inference files (`model.safetensors`, tokenizer files and custom
Python code), skipping the duplicate `nemo/*.nemo` checkpoint. The first run
still downloads roughly 4.9 GB. To avoid a runtime download entirely, download
the snapshot once and pass `--hf-asr-model-dir /path/to/snapshot`.

## Configuration

Set credentials in the environment (Kaggle *Secrets* or shell `export`). No
credential is stored in code, cache metadata, or reports.

| Variable | Required | Default / meaning |
| --- | --- | --- |
| `HF_TOKEN` | yes for default local backend, unless `--hf-asr-model-dir` is used | gated Bodhan HF repo + NeMo checkpoint access |
| `ASR_BACKEND` | no | `hf-local` (default); `cloud` switches to hosted Bodhan API |
| `HF_ASR_MODEL` | no | `bodhan-ai/indic-transcribe-core` |
| `HF_ASR_MODEL_DIR` | no | already-downloaded HF snapshot path |
| `ASR_COMPUTE_DTYPE` | no | `auto` → bfloat16 on CUDA, float32 on CPU |
| `BODHAN_LANGUAGE` | no | `gu`; use `hi`, `en`, or `""` for automatic LID |
| `BODHAN_API_KEY` | only with `ASR_BACKEND=cloud` | hosted Bodhan bearer token |
| `BODHAN_BASE_URL` | no | `https://api.bodhan.ai` |
| `BODHAN_MODEL` | no | `indic-transcribe` |
| `NEMO_DIAR_MODEL` | no | `nvidia/Nemotron-3-Diarization` or local `.nemo` path |
| `ALIGNMENT_MODEL` | no | `mms-uroman`, `gujarati-wav2vec2`, or `auto` |
| `ALIGNMENT_LANGUAGE` | no | ISO 639-3 hint for uroman, default `guj` |
| `MMS_ALIGN_MODEL` | no | default MMS CTC checkpoint |
| `GUJARATI_ALIGN_MODEL` | no | `gchhablani/wav2vec2-large-xlsr-gu` |
| `PIPELINE_DEVICE` | no | `auto`, `cuda`, or `cpu` |

Optional verification before a long run:

```bash
python pipeline.py --check --audio /path/to/meeting.wav
```

`--check` inspects the selected ASR backend, NeMo/torch/Python, the
canonical-audio requirement, the CTC tokenizers/configs, uroman, ffmpeg, and
credential presence. It does not download model weights or call an API.

## Single execution command

```bash
export HF_TOKEN="..."            # never commit this; accept the gated license first

python pipeline.py \
  --audio "/kaggle/input/your-dataset/meeting.wav" \
  --language gu \
  --output-dir outputs
```

`python -m gujarati_pipeline` is an equivalent entry point. The command writes
`outputs/transcript.json`, `outputs/processing_report.json`, and
`outputs/processing_report.md`. `--output path/to/file.json` overrides the final
JSON path directly. No `BODHAN_API_KEY` is used.

For the repository's bundled MP3 (ffmpeg decodes it; the original file is not
modified):

```bash
python pipeline.py \
  --audio "audio/20261031 CHD 3 GTH 11 12 13 14 15 (51) CHN.mp3" \
  --language hi \
  --output-dir outputs
```

The bundled `audio/*.json` is **not** a Bodhan response and is not used as
transcript, timestamp, or speaker truth. See the limitation note in
`docs/VERIFICATION.md`.

To explicitly use the hosted API instead:

```bash
export BODHAN_API_KEY="..."
python pipeline.py --audio meeting.wav --asr-backend cloud --language gu --output-dir outputs
```

## Test on the bundled MP3 end-to-end

Prerequisites: a Python **3.12+ GPU** environment, `torch>=2.7`, the pinned
RoPE-capable `NVIDIA-NeMo/Speech` build from `requirements.txt`, the other
`requirements.txt` packages, and gated HF access from the steps above. The
bundled MP3 is ~75.6 minutes, so the first
run downloads the Bodhan ASR checkpoint, the Nemotron diarization checkpoint,
and the MMS alignment checkpoint; allow substantial time and GPU memory.

```bash
export HF_TOKEN="hf_..."   # or load it from Kaggle Secrets

python -m tools.check_hf_access

python pipeline.py \
  --audio "audio/20261031 CHD 3 GTH 11 12 13 14 15 (51) CHN.mp3" \
  --language hi \
  --output-dir outputs
```

`--language hi` is appropriate for this Hindi-heavy sample; use `gu` for a
Gujarati-dominant meeting or `--language ""` for the model's automatic LID.
If you already downloaded the Bodhan snapshot, add
`--hf-asr-model-dir /path/to/snapshot`.

Then compare the real pipeline result with the attached ElevenLabs Scribe 2
reference:

```bash
python -m tools.scribe2_reference verify \
  --candidate outputs/transcript.json \
  --report-md outputs/scribe2_verification_report.md
```

## Outputs

| Path | Content |
| --- | --- |
| `outputs/transcript.json` | final UTF-8 JSON, exactly `{"segments": [...]}` |
| `outputs/processing_report.json` | detailed per-stage report and warnings |
| `outputs/processing_report.md` | brief report with required metrics |

`work/` holds organised intermediates:

```text
work/audio/<hash>/                 canonical 16 kHz mono WAV + metadata
work/transcription/audio/          trimmed chunk WAVs
work/transcription/responses/      raw ASR text per chunk (cache)
work/diarization/<hash>/           parsed NeMo intervals + speaker mapping
work/alignment/<hash>/             aligned word timestamps per transcript unit
```

Caches are keyed by the original audio SHA-256 plus stage configuration
(backend/model/language/device/chunking, NeMo model/config, alignment
model/language). Re-run without `--force` to reuse valid results. `--force all`
recomputes every stage; `--force transcription,alignment` recomputes only those
stages. `--transcription-cache-only` requires cached ASR text for every chunk.

## Pipeline stages

1. **Audio:** detects GPU/torch, copies an already-canonical 16 kHz mono PCM16
   WAV or uses ffmpeg to decode/downmix/resample. Original untouched. Prints
   duration and conversion method.
2. **Chunking:** energy VAD finds low-energy boundaries near 25 s, caps chunks
   at 28 s (Bodhan API and local model training limit 30 s), trims
   leading/trailing silence, and keeps absolute original offsets. Hard cuts are
   reported.
3. **Bodhan HF ASR:** `snapshot_download`s the gated model (or uses
   `--hf-asr-model-dir`), imports its own `IndicTranscribe` wrapper, and
   transcribes each chunk with the configured `gu`/`hi`/`en` language hint (or
   automatic LID). Repetition loops are split at quiet boundaries, then
   collapsed with a warning. Cloud backend uploads with retries/backoff only
   when explicitly selected.
4. **NeMo:** sets the offline Sortformer parameters, runs
   `SortformerEncLabelModel.diarize()` on the canonical WAV, parses
   `"start end speaker_N"` strings, merges duplicate speaker intervals for
   overlap scoring, and assigns stable `speaker_N` IDs by first arrival.
5. **Alignment:** if Bodhan did not supply word times, runs CTC forced
   alignment on each transcript unit. CTC character frames are mapped back to
   the **original** word strings; `<star>` absorbs untranscribed edges; no
   evenly distributed timestamps.
6. **Assembly:** for each word, sums overlap with each speaker's merged
   intervals. Winner must cover at least `--speaker-min-coverage` (default 0.5)
   and beat the runner-up by `--speaker-ambiguity-margin` (default 0.1), else
   `speaker_id=null`. Groups words on speaker change, pause, ASR
   chunk/segment boundary, sentence-ending punctuation, or max segment length.
7. **Validation/export:** checks exact schema keys, finite chronological
   timestamps within duration, `start < end`, `type: "word"`, segment/word text
   consistency, Unicode preservation (no U+FFFD), no dropped characters,
   consistent speaker IDs, and `null` attribution. Writes JSON + reports.

Useful overrides:

```bash
--hf-asr-model-dir /kaggle/models/indic-transcribe-core
--asr-compute-dtype bfloat16
--alignment-model auto                 # try Gujarati CTC first, fall back to MMS+uroman
--alignment-model gujarati-wav2vec2     # pure Gujarati only; OOV is a hard failure
--pause-threshold 0.7                   # meaningful pause
--speaker-min-coverage 0.5              # confidence threshold
--speaker-ambiguity-margin 0.1          # tie/ambiguity threshold
--bodhan-workers 2                     # cloud backend only; keep within rate limits
--force all                            # ignore valid caches
--transcription-cache-only              # never run ASR; require cached responses
```

## Offline tests

The deterministic core is unit-tested without Bodhan cloud credentials, HF
weights, NeMo, Transformers, or a GPU:

```bash
python -m unittest discover -s tests -v
```

18 tests cover: PCM16 WAV round-trip, silence-aware chunk offsets, documented
and optional Bodhan response parsing, local HF backend selection/caching with a
fake ASR wrapper (no cloud key required), CTC Viterbi collapse/word spans,
uroman normalization and explicit token merging, Gujarati OOV detection,
deterministic speaker overlap/tie handling, segment grouping, final JSON
validation (including dropped-word detection), and ElevenLabs Scribe 2 strict
conversion/verification.

## Attached MP3 / ElevenLabs Scribe 2 reference

The repository contains an MP3 and a companion JSON with ElevenLabs Scribe 2
word/segment metadata. That JSON is **not** a Bodhan, NeMo or pipeline result.
I added `tools/scribe2_reference.py` to convert it into the exact strict schema
and to verify a future pipeline JSON against it.

Measured with `mutagen`: the MP3 is **4538.6265625 s**, 48 kHz mono. The sample
JSON has 265 segments and 12,827 objects: 6,520 `word`, 6,281 `spacing`, and 26
`audio_event`. The strict conversion at
`outputs/scribe2_reference_strict.json` has:

- 265 segments and 6,546 word objects;
- 6 stable speaker IDs (`speaker_0` … `speaker_5`);
- 0 unattributed words;
- max word end 4530.594 s, within the audio duration;
- exact top-level `segments` and exact segment/word keys;
- canonical source text preserved.

Conversion decisions are reported in
`outputs/scribe2_conversion_report.md`: spacing objects are dropped, all 26
`audio_event` tokens are mapped to `type: "word"` because the strict schema has
no other type, and 25 zero-duration Scribe 2 word tokens are repaired using
the next real interval boundary from the same source response (no timestamp is
invented). This is a **schema conversion of the attached reference**, not an
independent ASR/diarization run.

Create it again:

```bash
pip install -r tools/requirements.txt   # optional: mutagen for MP3 duration
python -m tools.scribe2_reference convert
```

After a real pipeline run produces `outputs/transcript.json`, verify it:

```bash
python -m tools.scribe2_reference verify \
  --candidate outputs/transcript.json \
  --report-md outputs/scribe2_verification_report.md
```

The verifier checks the candidate's strict schema and compares canonical text,
word sequence, matched-word timestamp deltas, and speaker agreement against the
converted Scribe 2 reference. It writes JSON and Markdown reports. A self-check
using the converted reference as the candidate is already in
`outputs/scribe2_verification_selfcheck.md`; it matches perfectly because it is
the same file, so it demonstrates the verifier works, not transcription
accuracy.

## Accuracy / honesty limitations

- **No fabricated timestamps:** if CTC alignment cannot produce a valid
  interval for a word, the run fails and writes a failure report. Tokens with
  no CTC-alignable characters are merged into a neighbour with an explicit
  warning; they are never silently dropped.
- **Bodhan has no documented word timestamps in either backend:** the default
  path uses forced alignment. If a future Bodhan deployment returns reliable
  word times, the parser uses them without sending undocumented request
  parameters.
- **The local HF model is gated:** license acceptance plus `HF_TOKEN`, or a
  supplied `--hf-asr-model-dir`, is still required. There is no Bodhan cloud
  API key requirement by default.
- **Overlapping speech:** NeMo can output overlapping speaker intervals and the
  grouping retains separate speaker segments, but Bodhan's model card
  describes single-speaker input. Words from simultaneous speakers that Bodhan
  merges into one ASR stream cannot be separated by this pipeline.
- **Native-script output:** the local model renders English words inside Indic
  speech in the native script; automatic LID is uneven for Hindi-belt
  languages. Pass `--language` when known.
- **Speaker IDs** are stable within one recording only. They are not voice
  recognition / identity across recordings.
- **Model licensing:** the local Bodhan model is under the Indic Open Model
  License v1.0; the default MMS forced-alignment checkpoint is CC-BY-NC 4.0.
  Commercial use requires reviewing both licenses.
- **Sandbox:** the full GPU pipeline was not executed here because this
  workspace has no HF gated-model access, CUDA GPU, NeMo install, or downloaded
  weights. The offline unit tests and `--check` were run; `--check` correctly
  reports the missing heavy dependencies in this sandbox.

## Sources

- [Bodhan Indic-Transcribe-core model card](https://huggingface.co/bodhan-ai/indic-transcribe-core)
- [Bodhan Indic-Transcribe-core files / custom code](https://huggingface.co/bodhan-ai/indic-transcribe-core/tree/main)
- [Bodhan API Reference](https://console.bodhan.ai/api-docs/)
- [Bodhan quickstart](https://console.bodhan.ai/)
- [Bodhan AI Cookbook — speech to text](https://github.com/Bodhan-AI/Bodhan-AI-Cookbook)
- [NVIDIA Nemotron-3-Diarization model card](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- [NVIDIA Nemotron 3 Diarization blog](https://huggingface.co/blog/nvidia/nemotron-diarization)
- [NeMo Speech 3.0 on PyPI](https://pypi.org/project/nemo-toolkit/3.0.0/) — lacks RoPE support
- [NVIDIA-NeMo/Speech main](https://github.com/NVIDIA-NeMo/Speech) — pinned RoPE-capable source build
- [gchhablani/wav2vec2-large-xlsr-gu](https://huggingface.co/gchhablani/wav2vec2-large-xlsr-gu)
- [MahmoudAshraf/mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)
- [ctc-forced-aligner source](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [WhisperX alignment source](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)
- [Torchaudio CTC forced alignment tutorial](https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html)
- [Hugging Face gated models](https://huggingface.co/docs/hub/models-gated)
- [Hugging Face Hub CLI / `hf auth login`](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
- [`snapshot_download` allow/ignore patterns](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download)
