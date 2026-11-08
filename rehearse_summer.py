"""Rehearse the hidden test on past summers.

The hidden test trains on sales up to May 31, 2026 and predicts homes sold June to
September 2026: one to four months ahead, in the season when prices dip. run.py validate
only predicts one month ahead, December to May. This script trains on sales up to
May 31 of 2024 and of 2025 and predicts June to September of the same year, so its
errors are the best guide to the hidden-test error. From the repo folder:

    python rehearse_summer.py
"""
import numpy as np
import pandas as pd

import data
import models

listings = data.load("listings")
rows, pooled = [], {"off_market": [], "list_aware": []}
for year in [2024, 2025]:
    train = listings[listings["date_sold"] < f"{year}-06-01"]
    for track in ["off_market", "list_aware"]:
        model = models.make_model(track).fit(train)
        for month in [6, 7, 8, 9]:
            homes = listings[(listings["date_sold"].dt.year == year) & (listings["date_sold"].dt.month == month)]
            predicted = model.predict(data.hide_answers(homes, track))
            signed = 100 * (predicted - homes["sold_price"]) / homes["sold_price"]
            pooled[track].append(signed.to_numpy())
            rows.append({"year": year, "track": track, "month": month, "homes": len(homes),
                         "MdAPE": np.median(signed.abs()), "MAPE": np.mean(signed.abs()),
                         "median_signed_error": np.median(signed)})
        print(f"{year} {track}: done")

table = pd.DataFrame(rows)
print(table.round(2).to_string(index=False))
print("\nAverage over both summers:")
print(table.groupby("track")[["MdAPE", "MAPE", "median_signed_error"]].mean().round(2))
print("\nAll summer homes together (pooled):")
for track, parts in pooled.items():
    signed = np.concatenate(parts)
    print(f"{track}: MdAPE {np.median(np.abs(signed)):.2f}%  MAPE {np.mean(np.abs(signed)):.2f}%  "
          f"median signed error {np.median(signed):+.2f}%  ({len(signed):,} homes)")
