from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cva_portfolio import run_cva_analysis  # noqa: E402


if __name__ == "__main__":
    results = run_cva_analysis(
        market_data_path=PROJECT_ROOT / "data" / "Market_data.xlsx",
        output_dir=PROJECT_ROOT / "results",
    )

    print("\nCVA results [PLN]:")
    print(results["cva_results"].to_string(index=False))
    print(f"\nVariation Margin CVA reduction: {results['reduction_vm']:.2%}")
    print("\nOutputs saved to:", PROJECT_ROOT / "results")
