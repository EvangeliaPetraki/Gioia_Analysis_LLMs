from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.env_config import PROJECT_ROOT, get_path

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.code_gioia_pdf_and_excel import process_folder as run_gioia_first_order
from analysis.gioia_second_order import process_folder as run_gioia_second_order
from analysis.policy_analysis_simplified_chutes import process_folder as run_policy_analysis


DEFAULT_INPUT_FOLDER = get_path("ANALYSIS_INPUT_FOLDER", PROJECT_ROOT / "analysis" / "input")
DEFAULT_OUTPUT_ROOT = get_path("ANALYSIS_OUTPUT_ROOT", PROJECT_ROOT / "analysis" / "output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the document analysis pipeline in the correct order."
    )
    parser.add_argument(
        "--input-folder",
        default=str(DEFAULT_INPUT_FOLDER),
        help="Folder containing the source PDF documents.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder where analysis outputs will be created.",
    )
    return parser


def run_pipeline(input_folder: Path, output_root: Path) -> None:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    policy_output = output_root / "policy_analysis"
    gioia_first_output = output_root / "gioia_first_order"
    gioia_second_output = output_root / "gioia_second_order"

    print(f"Input folder: {input_folder}")
    print(f"Output root: {output_root}")

    print("\n[1/3] Running policy analysis on source PDFs...")
    run_policy_analysis(str(input_folder), str(policy_output))

    print("\n[2/3] Running Gioia first-order analysis on source PDFs...")
    run_gioia_first_order(str(input_folder), str(gioia_first_output))

    print("\n[3/3] Running Gioia second-order analysis on first-order PDFs...")
    run_gioia_second_order(str(gioia_first_output), str(gioia_second_output))

    print("\nAnalysis pipeline completed successfully.")


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(Path(args.input_folder), Path(args.output_root))


if __name__ == "__main__":
    main()
