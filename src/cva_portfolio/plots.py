from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


def save_ee_plot(ee_profiles: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5.5))
    plt.plot(ee_profiles["TimeYears"], ee_profiles["EE_FXForward_PLN"], label="FX Forward")
    plt.plot(ee_profiles["TimeYears"], ee_profiles["EE_IRS_PLN"], label="Receiver IRS")
    plt.plot(ee_profiles["TimeYears"], ee_profiles["EE_Portfolio_PLN"], label="Portfolio without VM", linewidth=2)
    plt.plot(ee_profiles["TimeYears"], ee_profiles["EE_Portfolio_VM_PLN"], label="Portfolio with VM", linewidth=2, linestyle="--")
    plt.title("Expected Exposure profiles")
    plt.xlabel("Time [years]")
    plt.ylabel("Expected Exposure [PLN]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_cva_plot(cva_results: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(cva_results["Position"], cva_results["CVA_PLN"], edgecolor="black", linewidth=0.8)
    plt.title("CVA comparison")
    plt.ylabel("CVA [PLN]")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    max_value = cva_results["CVA_PLN"].max()
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max_value*0.025, f"{height:,.0f} PLN", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, max_value * 1.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
