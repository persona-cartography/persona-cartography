"""Visualisation and plotting tools for LoRA analysis and eval results."""

from pathlib import Path

from dotenv import load_dotenv

# The monorepo dataset is currently a private HF repo; load HF_TOKEN from .env on
# import so that every plotting script importing this package authenticates its
# HfFileSystem reads automatically (a no-op once the repo is made public).
load_dotenv()

PAPER_FIGURES_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures"
