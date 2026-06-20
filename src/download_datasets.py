"""
Omli — Real Dataset Downloader
================================
Downloads 3 real datasets automatically:
  1. ESC-50        → noise clips (rain, crowd, traffic, etc.)
  2. LibriSpeech   → adult speech (dev-clean, ~337 MB)
  3. Zenodo 200495 → real child speech WAV files (~50 MB)

Run:
  python src/download_datasets.py
"""

import os
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
RAW      = BASE_DIR / "data" / "raw"

# ── progress bar for downloads ──
class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url, dest_path):
    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"  Already exists: {dest_path.name} — skipping")
        return
    print(f"  Downloading {dest_path.name}...")
    with DownloadProgress(unit="B", unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
    print(f"  Done.")


# ─────────────────────────────────────────────
# 1. ESC-50 — noise clips
# ─────────────────────────────────────────────
def download_esc50():
    print("\n[1/3] ESC-50 (noise clips)")
    dest_dir = RAW / "esc50"
    dest_dir.mkdir(exist_ok=True)
    zip_path = RAW / "esc50.zip"

    download(
        "https://github.com/karolpiczak/ESC-50/archive/master.zip",
        zip_path
    )

    audio_dir = dest_dir / "ESC-50-master" / "audio"
    if not audio_dir.exists():
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        print("  Extracted.")

    wavs = list(audio_dir.glob("*.wav"))
    print(f"  Found {len(wavs)} noise clips in {audio_dir}")
    return audio_dir


# ─────────────────────────────────────────────
# 2. LibriSpeech dev-clean — adult speech
# ─────────────────────────────────────────────
def download_librispeech():
    print("\n[2/3] LibriSpeech dev-clean (adult speech, ~337 MB)")
    dest_dir = RAW / "librispeech"
    dest_dir.mkdir(exist_ok=True)
    tar_path = RAW / "dev-clean.tar.gz"

    download(
        "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        tar_path
    )

    libri_dir = dest_dir / "LibriSpeech" / "dev-clean"
    if not libri_dir.exists():
        print("  Extracting (this takes a minute)...")
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall(dest_dir)
        print("  Extracted.")

    flacs = list(libri_dir.rglob("*.flac"))
    print(f"  Found {len(flacs)} adult audio files in {libri_dir}")
    return libri_dir


# ─────────────────────────────────────────────
# 3. Zenodo 200495 — real child speech
# ─────────────────────────────────────────────
def download_child_speech():
    print("\n[3/3] Zenodo child speech (11 children aged ~5 years)")
    dest_dir = RAW / "child_speech"
    dest_dir.mkdir(exist_ok=True)
    zip_path = RAW / "child_speech.zip"

    download(
        "https://zenodo.org/record/200495/files/children_recordings.zip",
        zip_path
    )

    if not any(dest_dir.glob("**/*.wav")):
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        print("  Extracted.")

    wavs = list(dest_dir.rglob("*.wav"))
    print(f"  Found {len(wavs)} child audio files")
    return dest_dir


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Omli — Downloading Real Datasets")
    print("=" * 50)
    RAW.mkdir(parents=True, exist_ok=True)

    esc50_dir   = download_esc50()
    libri_dir   = download_librispeech()
    child_dir   = download_child_speech()

    print("\n" + "=" * 50)
    print("  All datasets downloaded.")
    print(f"  ESC-50 noise : {esc50_dir}")
    print(f"  LibriSpeech  : {libri_dir}")
    print(f"  Child speech : {child_dir}")
    print("\nNext: python src/day1_real_data.py")