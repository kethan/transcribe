"""Gujarati meeting transcription, diarization and forced-alignment pipeline.

The local Bodhan Hugging Face ASR model is the default transcription backend;
the hosted Bodhan cloud API is an explicit option. The package is deliberately
importable without GPU/NeMo/Transformers/requests/torch so that deterministic
parts (chunking, CTC Viterbi alignment, speaker overlap, segment grouping, ASR
cache handling and JSON validation) can be unit-tested offline. Heavy
dependencies are imported lazily inside the stage that needs them.
"""

__version__ = "1.1.0"
