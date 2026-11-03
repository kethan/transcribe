# Kaggle quickstart — one notebook

This folder exists to answer one question: **“which file do I run on Kaggle?”**

Use **`kaggle/transcribe.ipynb`**. It has one code cell. You do not need to
understand `gujarati_pipeline/`, `tools/`, `docs/`, or `tests/` to run it.

## Before you run

1. Accept the gate at
   <https://huggingface.co/bodhan-ai/indic-transcribe-core> and create a read
   `HF_TOKEN`.
2. In Kaggle: **Add-ons → Secrets → Add a new secret**, name it exactly
   `HF_TOKEN`, paste the token, and enable it for the notebook.
3. Create a notebook and import/upload `kaggle/transcribe.ipynb`.
4. In notebook **Settings / Session options**:
   - **Internet: On** (needed for `git clone`, `pip install`, and HF downloads).
   - **Accelerator: GPU** — T4 or P100. A 75-minute MP3 on CPU is impractical.
   - Use the latest Python GPU image. The pinned NeMo 3.0 stack expects
     Python 3.12+ / PyTorch 2.7+; check the printed `torch` version after install.
5. Run the single cell.

## What the cell does

It clones `arena/793a5a3b-transcribe` into `/kaggle/working/transcribe`,
installs `requirements.txt` and `tools/requirements.txt`, checks your HF gate,
uses the bundled `audio/*.mp3` and `audio/*.json`, runs `pipeline.py`, and
writes everything to `/kaggle/working/transcribe_outputs/`.

The first execution can take a long time: it downloads the ~4.9 GB Bodhan ASR
checkpoint, the Nemotron diarization checkpoint, and the MMS alignment
checkpoint. It also verifies `outputs/transcript.json` against the converted
ElevenLabs Scribe 2 reference.

## Only two values to edit

At the top of the cell:

```python
ASR_LANGUAGE = "hi"          # "hi" for the bundled Hindi-heavy MP3
ALIGNMENT_LANGUAGE = "hin"  # ISO 639-3 for uroman; use "guj" for Gujarati
AUDIO_PATH = ""             # blank = bundled MP3; or set /kaggle/input/.../file.mp3
```

If you set `AUDIO_PATH` to a different MP3, the Scribe 2 verification is skipped
automatically because the attached reference belongs to the bundled audio.

## Output files

After the cell finishes, open `/kaggle/working/transcribe_outputs/`:

- `transcript.json` — real strict-schema pipeline output
- `processing_report.json` / `.md`
- `scribe2_reference_strict.json`
- `scribe2_conversion_report.json` / `.md`
- `scribe2_verification_report.json` / `.md`

Kaggle keeps `/kaggle/working` for the session; download the output directory
before stopping the notebook.

## If it fails

- `401` / `403` / “gated repo”: accept the license on the model page and make
  sure the Kaggle Secret belongs to the same HF account.
- `torch.cuda.is_available() == False`: enable GPU in Settings and restart.
- `ffmpeg MISSING`: the MP3 cannot be converted to canonical WAV. Use a WAV
  input or install/enable ffmpeg in the Kaggle image.
- `pip` replaces Kaggle's CUDA-enabled PyTorch with a CPU build: stop and install
  the missing packages individually rather than forcing a new `torch` wheel.
- Out of disk: Kaggle gives 20 GB under `/kaggle/working`; the three checkpoints
  plus the MP3 and intermediate WAVs can be tight. Use `HF_HOME` on scratch space
  or a shorter clip for a first smoke test.
