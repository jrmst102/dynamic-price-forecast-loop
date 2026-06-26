import sys

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/calibrated_markets.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "data/calibrated_worlds_config.csv"
NOISE = 0.05  # assumed exogenous price-noise level; tune if you change the world
df = pd.read_csv(src)
cfg = pd.DataFrame(
    {
        "category": df["category"],
        "elasticity": df["beta_iv"].round(2),
        "shock_std": df["sigma_xi"].round(2),
        "shock_ar1": df["rho"].round(2),
        "first_stage_r2": df["first_stage_r2"].round(2),
        "cost_shifter_std": (
            NOISE * np.sqrt(df["first_stage_r2"] / (1 - df["first_stage_r2"]))
        ).round(3),
        "lambda_anchor": df["lambda_implied"].round(3),
        "direction": np.where(df["lambda_implied"] > 0, "loop", "promo"),
    }
).sort_values("lambda_anchor", ascending=False)
cfg.to_csv(out, index=False)
print(f"wrote {out} ({len(cfg)} rows)")
