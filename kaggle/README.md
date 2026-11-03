# Kaggle quickstart — one notebook

This folder exists to answer one question: **“which file do I run on Kaggle?”**

Use **`kaggle/transcribe.ipynb`**. It has two code cells: a GPU diagnostic and
the main runner. You do not need to understand `gujarati_pipeline/`, `tools/`,
`docs/`, or `tests/` to run it.

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
5. **Run the GPU diagnostic cell first.** Require `nvidia-smi` success and
   `torch.cuda.is_available() == True`. If not, fix the accelerator and
   **restart the session** before continuing.
6. Run the main cell.

## What the main cell does

It clones `arena/793a5a3b-transcribe` into `/kaggle/working/transcribe`,
installs `requirements.txt` and `tools/requirements.txt` while pinning
Kaggle's preinstalled CUDA `torch`/`torchaudio`, checks your HF gate, uses the
bundled `audio/*.mp3` and `audio/*.json`, runs `pipeline.py` with full
speaker-turn segments, and writes everything to
`/kaggle/working/transcribe_outputs/`. It refuses to continue if
`torch.cuda.is_available()` is false after installation.

The first execution can take a long time: it downloads the ~4.9 GB Bodhan ASR
checkpoint, the Nemotron diarization checkpoint, and the MMS alignment
checkpoint. It also verifies `transcript.json` against the converted ElevenLabs
Scribe 2 reference. No transcript text is hardcoded; the final words come from
the gated Bodhan model's responses, and ElevenLabs Scribe 2 is reference only.

## Only two values to edit

At the top of the main cell:

```python
ASR_LANGUAGE = "hi"          # "hi" for the bundled Hindi-heavy MP3
ALIGNMENT_LANGUAGE = "hin"  # ISO 639-3 for uroman; use "guj" for Gujarati
AUDIO_PATH = ""             # blank = bundled MP3; or set /kaggle/input/.../file.mp3
```

If you set `AUDIO_PATH` to a different MP3, the Scribe 2 verification is skipped
automatically because the attached reference belongs to the bundled audio.

## Output files

After the main cell finishes, open `/kaggle/working/transcribe_outputs/`:

- `pipeline.log` — full stdout/stderr of the child pipeline process
- `transcript.json` — real strict-schema pipeline output, grouped as full speaker turns
- `bodhan_transcript.txt` — verbatim actual Bodhan ASR text, for download
- `bodhan_transcript.json` — same text with chunk spans and provenance
- `processing_report.json` / `.md`
- `scribe2_reference_strict.json`
- `scribe2_conversion_report.json` / `.md`
- `scribe2_speaker_turns.json` / `scribe2_speaker_turns_report.md`
- `scribe2_verification_report.json` / `.md`

Kaggle keeps `/kaggle/working` for the session; download the output directory
before stopping the notebook.

The notebook also re-exports the Bodhan text from
`/kaggle/working/transcribe/work/transcription/responses/` with
`python -m tools.bodhan_reference`, so a preserved ASR cache can recover the
downloadable TXT/JSON reference without rerunning the model. The separate
`scribe2_speaker_turns.json` keeps ElevenLabs' extra word fields; the strict
`transcript.json` does not.

## If it fails

- `self_attention_model='rope' is not supported` / `Model instantiation failed`:
  you have PyPI `nemo-toolkit[asr]==3.0.0`. Nemotron-3-Diarization needs the
  RoPE-capable NeMo Speech main build. `requirements.txt` now pins GitHub
  commit `e5d79d5a49cbec49d79440b0070aac0c32a8dbb8`. Restart the session,
  reinstall from the updated `requirements.txt`, and re-run. The pipeline's
  `--check` now detects this before the ASR stage.
- `CalledProcessError ... returned non-zero exit status 1`: this traceback is
  only the notebook wrapper. The real child-process message is the
  `Pipeline failed: ...` or `Configuration error: ...` line immediately above
  it. Use the updated notebook: it prints `pipeline.py --check` first and
  streams every line of the real run to `/kaggle/working/transcribe_outputs/pipeline.log`.
  Paste the last ~50 lines of that log when asking for help.
- `401` / `403` / “gated repo”: accept the license on the model page and make
  sure the Kaggle Secret belongs to the same HF account.
- `torch.cuda.is_available() == False`: enable GPU in Settings, Internet On,
  and restart. Kaggle restarts the session when the accelerator changes.
- `Could not find cuda drivers ... cudart_stub.cc`: this specific message often
  comes from **TensorFlow**, not PyTorch. Run the diagnostic cell and trust
  `torch.cuda.is_available()`. The notebook sets `USE_TF=0` and
  `TF_CPP_MIN_LOG_LEVEL=3` to keep that warning out of the way.
- `nvidia-smi` works but `torch.version.cuda is None`: a CPU-only PyTorch wheel
  was installed. **Restart the notebook** to restore Kaggle's CUDA image, then
  re-run; the main cell now writes `/kaggle/working/torch_constraints.txt` and
  uses `pip -c` so it cannot swap the CUDA `torch`/`torchaudio` pair.
- `ffmpeg MISSING`: the MP3 cannot be converted to canonical WAV. Use a WAV
  input or install/enable ffmpeg in the Kaggle image.
- Out of disk: Kaggle gives 20 GB under `/kaggle/working`; the three checkpoints
  plus the MP3 and intermediate WAVs can be tight. Use `HF_HOME` on scratch space
  or a shorter clip for a first smoke test.
