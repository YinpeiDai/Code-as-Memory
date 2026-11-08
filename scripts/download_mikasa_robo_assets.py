"""Fetch the ManiSkill YCB asset archive pinned by the installed mani_skill release.

ShellGame* MIKASA-Robo tasks load ``ycb:`` cups. mani_skill 3.0.0b15 records
SHA-256 ``174001ba...`` for ``mani_skill2_ycb.zip``, but the Hugging Face ``main``
file changed in March 2025, so the stock downloader rejects it. This script
downloads the same archive from the last Hugging Face revision whose LFS object
matches the recorded checksum and lets ManiSkill verify and extract it.

Run in the MIKASA-Robo interpreter:

    MS_ASSET_DIR=/abs/path/maniskill_data \
        envs/mikasa_robo/.venv/bin/python scripts/download_mikasa_robo_assets.py
"""

import dataclasses
import os
import sys

REVISION = "4c134e2e33b11b1fa751c1d7b6afb2e89231b875"  # haosulab/ManiSkill2, 2024-09-12
URL = (
    "https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/"
    f"{REVISION}/data/mani_skill2_ycb.zip"
)


def main():
    if not os.environ.get("MS_ASSET_DIR"):
        raise SystemExit("Set MS_ASSET_DIR to the destination asset directory")
    from mani_skill.utils import assets
    from mani_skill.utils.download_asset import download

    source = assets.DATA_SOURCES["ycb"]
    pinned = dataclasses.replace(source, url=URL)
    print(f"ManiSkill ycb checksum {source.checksum}; source {URL}")
    path = download(pinned, verbose=False, non_interactive=True)
    print(f"Extracted to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
