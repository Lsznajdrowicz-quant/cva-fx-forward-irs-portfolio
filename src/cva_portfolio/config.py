from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelConfig:
    """Configuration for the CVA portfolio simulation."""

    valuation_date: str = "2025-12-31"
    historical_start_date: str = "2022-12-31"
    n_sim: int = 10_000
    n_steps: int = 36
    dt: float = 1 / 12
    seed: int = 42

    notional_fxf: float = 1_000_000.0
    strike_fxf: float = 4.4439

    notional_irs: float = 10_000_000.0
    k_fixed: float = 0.0202
    tau_fixed: float = 1.0
    tau_float: float = 0.25

    recovery_rate: float = 0.40
    cds_zero_rate: float = 0.0200
    premium_freq_months: int = 3

    r_eur_flat: float = 0.0200
    r_pln_flat: float = 0.0375
    sigma_fx: float = 0.0415297
    alpha: float = 0.05

    fixed_pay_idx: list[int] = field(default_factory=lambda: [12, 24, 36])
    float_pay_idx: list[int] = field(default_factory=lambda: list(range(3, 37, 3)))

    @property
    def months(self):
        return np.arange(0, self.n_steps + 1)

    @property
    def time_grid(self):
        return self.months / 12

    @property
    def lgd(self):
        return 1 - self.recovery_rate

    @property
    def float_reset_idx(self):
        return [0] + self.float_pay_idx[:-1]
