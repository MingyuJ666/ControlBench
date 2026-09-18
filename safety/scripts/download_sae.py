"""Download exactly three pinned Gemma Scope instruction dictionaries."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safety_steering.sae import SPEC, dictionary_record
from safety_steering.data import write_new


def main():
    from huggingface_hub import hf_hub_download

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.out.resolve()
    for layer, filename in SPEC["layers"].items():
        record = root / f"layer-{layer}.json"
        if record.exists():
            dictionary_record(root, layer)
            continue
        path = Path(
            hf_hub_download(
                SPEC["repo"], filename, revision=SPEC["revision"], local_dir=root
            )
        )
        write_new(
            record,
            dict(
                repo=SPEC["repo"],
                revision=SPEC["revision"],
                format=SPEC["format"],
                block_index=layer,
                filename=filename,
                path=str(path.relative_to(root)),
            ),
        )


if __name__ == "__main__":
    main()
