from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from gujarati_pipeline.transcription import HF_ASR_INFERENCE_FILES

DEFAULT_ASR_MODEL = "bodhan-ai/indic-transcribe-core"
DEFAULT_DIAR_MODEL = "nvidia/Nemotron-3-Diarization"


def token_from_env() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def check_model(hf_api: Any, repo_id: str, token: str, small_files: list[str]) -> dict[str, Any]:
    info = hf_api.model_info(repo_id=repo_id, token=token)
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "gated": info.gated,
        "private": bool(getattr(info, "private", False)),
        "sha": getattr(info, "sha", None),
        "small_files": {},
    }
    for filename in small_files:
        path = hf_api.hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        result["small_files"][filename] = str(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.check_hf_access",
        description="Verify HF_TOKEN, gated-model approval and small-file download access.",
    )
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--diar-model", default=DEFAULT_DIAR_MODEL)
    parser.add_argument(
        "--skip-diar",
        action="store_true",
        help="Only check the Bodhan ASR repo (useful before the NeMo dependency is installed).",
    )
    args = parser.parse_args(argv)

    token = token_from_env()
    if not token:
        print(
            "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set.\n"
            "1. Accept the gated model license in your browser.\n"
            "2. Create a read token at https://huggingface.co/settings/tokens.\n"
            "3. export HF_TOKEN='hf_...' or add it as a Kaggle Secret.\n"
            "Never paste the token into chat or source files.",
            file=sys.stderr,
        )
        return 2
    try:
        import huggingface_hub
    except Exception as exc:  # pragma: no cover - installation issue
        print(f"huggingface_hub is not installed: {exc}", file=sys.stderr)
        print("Install it with: pip install -U 'huggingface_hub>=0.34,<2'", file=sys.stderr)
        return 2

    try:
        who = huggingface_hub.whoami(token=token)
        print(f"Authenticated as: {who.get('name') or who.get('user') or 'unknown'}")
        asr = check_model(
            huggingface_hub,
            args.asr_model,
            token,
            ["config.json", "indic_transcribe.py"],
        )
        print(
            f"ASR repo {asr['repo_id']}: gated={asr['gated']}, "
            f"config + custom code download OK"
        )
        print(f"Pipeline downloads only: {', '.join(HF_ASR_INFERENCE_FILES)}")
        results: dict[str, Any] = {"asr": asr}
        if not args.skip_diar:
            diar = check_model(huggingface_hub, args.diar_model, token, ["config.json"])
            print(f"Diarization repo {diar['repo_id']}: gated={diar['gated']}, config download OK")
            results["diarization"] = diar
        print(
            "\nACCESS OK. Next: run `python pipeline.py --audio ... --output-dir outputs` "
            "on a Python 3.12+ GPU environment with NeMo 3.0 / torch>=2.7 installed."
        )
        return 0
    except Exception as exc:
        print(f"Access check failed: {exc}", file=sys.stderr)
        print(
            "Most common causes: you did not click 'Agree and access repository' on the "
            "model page, HF_TOKEN belongs to a different account, the token lacks read "
            "scope, or manual approval is still pending.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
