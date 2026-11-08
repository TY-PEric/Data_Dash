# SSU Data Dash 2026: Toronto home prices

Our submission for the SSU Data Dash 2026 case. Team members are listed in the report.

Built on the official starter ([erwinzhang7/SSU-Data-Dash](https://github.com/erwinzhang7/SSU-Data-Dash));
`run.py`, `data.py`, `check_submission.py`, `tests/` and `requirements.txt` are the starter's, unchanged.
The starter's own README is kept as [`STARTER_README.md`](STARTER_README.md). The starter is MIT-licensed by Erzheng Zhang;
its licence is kept as [`STARTER_LICENSE`](STARTER_LICENSE).

## Results

Walk-forward validation, December 2025 to May 2026, from `python run.py validate`: each month is
predicted by models trained only on the sales before it. Brackets: bootstrap 95% interval.

| Track | Homes | MdAPE | MAPE | Median signed error |
|---|---|---|---|---|
| Off-market | 10,069 | **7.57%** (7.39 to 7.75) | 10.09% | +1.83% |
| List-aware | 10,069 | **4.15%** (4.06 to 4.23) | 5.26% | +1.14% |

Starter baselines on the same months: 20.8% (off-market) and 7.3% (list-aware).

**Expected hidden-test error.** The hidden test predicts June to September 2026, one to four months
past the training data, in the summer, when prices dip. The validation months contain no summer and
look only one month ahead, so we also rehearse the test on past summers with `rehearse_summer.py`:
train on sales up to May 31, predict June to September of the same year, for 2024 and 2025.

| Track | Summer rehearsal MdAPE (2024 + 2025, all homes pooled) | Median signed error |
|---|---|---|
| Off-market | 7.42% | +1.18% |
| List-aware | 4.21% | +0.05% |

Full results: [`reports/validation_report.md`](reports/validation_report.md),
[`figures/error_by_price_range.png`](figures/error_by_price_range.png).

## Reproduce

```
pip install -r requirements.txt
python run.py validate      # about 12 minutes; writes outputs/validation_predictions.csv
python run.py report        # reports/ and figures/
python run.py predict       # submissions/submission_off_market.csv and submission_list_aware.csv
python run.py check         # checks the two submission files
python rehearse_summer.py   # summer rehearsal, a few minutes
pytest                      # the starter's 18 tests
```

The data downloads from Hugging Face the first time (see `data.py`). Everything is deterministic
(fixed random seeds), so a rerun gives the same predictions.

## The model

`make_model()` in [`models.py`](models.py) returns the two models in [`claude_model.py`](claude_model.py).
Idea: take time out of every past price, learn what the home is worth on its own, put time back.

### Off-market (`ClaudeModel`)

1. **Size.** For range sizes such as "900-1099", `SizeEstimator` estimates the true size from the
   home's other facts (rooms, parking, garage, lot, location, style), never from prices. It learns
   from exact-size homes given a pretend range of the kind sellers use. Price per square foot uses
   the estimated size.
2. **Price index** (`PriceIndex`). Monthly log price per square foot for each home type (co-ops follow
   condos), adjusted for which neighbourhoods sold that month. Split into a 12-month season and a
   deseasonalised level. Every past sale is converted to "index-free" prices; predictions add back the
   latest level (average of the last 3 months, held flat) plus the season of the expected sale month.
3. **Comparable sales.** Index-free price per square foot of the nearest 5, 10 and 30 sales of the same
   type and their distances; the nearest 20 of any type; neighbourhood x type and neighbourhood levels
   (leaving the home itself out).
4. **Similar homes and land.** The 10 nearest homes in a space of location, size and (for houses) lot
   size; for houses, the neighbours' land price (price per lot area).
5. **Lot.** Frontage, depth, log lot area, lot vs floor area, lot vs the neighbours' lots.
6. **Price pockets** (`features.LocationLevels`). Inside each neighbourhood, a small tree on latitude and
   longitude groups nearby sales with similar prices; the level of a home's pocket and neighbourhood
   over the year before it was listed.
7. **Time on market.** How long the home had already been listed without selling, compared with the
   typical time on market of similar homes nearby (20 merged areas x type x 4 size bands, falling back
   to municipality x type) in the year before its listing.
8. **Remarks** (`remarks_features.py`). About 44 yes/no questions from the listing description
   (condition, systems, kitchen and bath, floors, space, outside, condo building, nearby places, how it
   is sold) and a 0-3 condition score.
9. **Trees.** scikit-learn `HistGradientBoostingRegressor`: 1,000 trees, learning rate 0.03, 63 leaves,
   at least 40 homes per leaf, 60% of the columns per split, L2 = 1.0. Target: log price minus the
   index of the sale month.

### List-aware (`ListAwareBlend`)

The geometric mean of two designs that miss in opposite directions:

- **index version:** the off-market approach, with the list price in index-free dollars as a feature;
- **ratio version:** learns log(sold price / list price) directly and multiplies the list price back.

Both use the comparables, time on market and remarks features, plus: the list price vs the
comparables, the expected sale month, and market heat (average sold / list over the 90 days before
listing, city-wide, for condos and for detached houses). Trees: 300, absolute-error loss.

### Not used by `make_model()`

`time_effects.py`, `TimeAdjustedModel` and `LocationFeatureModel` in `models.py` are an earlier
alternative design (season and year effects by neighbourhood tier), kept for comparison and listed
as comments in `make_model()`.

## Honesty notes

- Every feature is built from the training sales only. `data.hide_answers()` removes sold price, sale
  date, days on market and the external AVM from the homes being predicted; off-market never sees the
  list price.
- Known issue in the comparable-sales features: during training, a sale is meant to be left out of its
  own comparables by dropping its nearest neighbour. Condos in one building share one location, so the
  nearest neighbour is often another unit: a condo's own price is inside its 10 nearest comparables for
  about 31% of condo training rows (30 nearest: about 65%). This does not affect the validation scores
  (validation homes are never in the training data), but it teaches the trees to trust comparables
  more than they deserve on new homes.
- The list-aware `sale_month` feature is exact in validation (every validation home sold in its month)
  but estimated for the hidden test (a home listed before June is taken to sell in June), so
  validation flatters it slightly.
- The model priced homes slightly high on average (median signed error +1.8% off-market, +1.1%
  list-aware on the validation months).
