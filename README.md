# SSU Data Dash 2026: Toronto home prices

Our submission for the SSU Data Dash 2026 case.

Built on the official starter ([erwinzhang7/SSU-Data-Dash](https://github.com/erwinzhang7/SSU-Data-Dash));
`run.py`, `data.py`, `check_submission.py`, `tests/` and `requirements.txt` are the starter's, unchanged.

| File | What it is |
|---|---|
| `models.py` | `make_model()`: picks the model for each track |
| `valuation_model.py` | the two models: off-market `IndexAdjustedGBM`, list-aware `ListPriceEnsemble` |
| `features.py` | starter features and micro-market price levels (`MicroMarketLevels`) |
| `remarks_features.py` | text features: yes/no answers from the listing descriptions |
| `time_effects.py` | an alternative time-adjustment design; imported by `models.py`, not used by `make_model()` |
| `run.py`, `data.py`, `check_submission.py` | the starter's pipeline: validate, report, predict, check |
| `rehearse_summer.py` | the summer rehearsal |
| `submissions/` | the two submission files |
| `reports/`, `figures/` | the validation report and error figure |
| `tests/`, `pytest.ini` | the starter's tests |

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

You need Python 3.10 or newer and an internet connection. Nothing has to be downloaded by hand:
`pip` installs the libraries and the code fetches the data itself.

**1. Get the code.** Either clone it:

```
git clone https://github.com/TY-PEric/Data_Dash.git
cd Data_Dash
```

or click **Code → Download ZIP** at the top of this page, unzip it, and open a terminal in the
unzipped folder.

**2. Install the libraries** (pinned versions, from `requirements.txt`):

```
pip install -r requirements.txt
```

**3. Make the submission files.** This trains on all 150,000 past sales and predicts the test homes:

```
python run.py predict       # about 3 minutes
python run.py check         # should print "OK" for both files
```

It writes `submissions/submission_off_market.csv` and `submissions/submission_list_aware.csv`,
the same files as in this repository: every random step has a fixed seed, so a rerun gives the
same predictions.

**4. Optional: the validation numbers in Results.**

```
python run.py validate      # about 12 minutes; month-by-month validation, December 2025 to May 2026
python run.py report        # writes reports/validation_report.md and figures/error_by_price_range.png
python rehearse_summer.py   # about 3 minutes; the summer rehearsal (its "pooled" lines)
pytest                      # the starter's 18 tests
```

## The model

`make_model()` in [`models.py`](models.py) returns the two models in [`valuation_model.py`](valuation_model.py).
Idea: take time out of every past price, learn what the home is worth on its own, put time back.

### Off-market: index-adjusted gradient boosting (`IndexAdjustedGBM`)

1. **Floor-area imputation.** For range sizes such as "900-1099", `FloorAreaImputer` estimates the true size from the
   home's other facts (rooms, parking, garage, lot, location, style), never from prices. It learns
   from exact-size homes given a pretend range of the kind sellers use. Price per square foot uses
   the estimated size.
2. **Mix-adjusted price index** (`MixAdjustedPriceIndex`). Monthly log price per square foot for
   each home type (co-ops follow condos), adjusted for which neighbourhoods sold that month. Split into a 12-month season and a
   deseasonalised level. Every past sale is converted to "index-free" prices; predictions add back the
   latest level (average of the last 3 months, held flat) plus the season of the expected sale month:
   the listing month, or the first month after the training data if the home was listed before it.
3. **Comparable sales.** Index-free price per square foot of the nearest 5, 10 and 30 sales of the same
   type and their distances; the nearest 20 of any type; neighbourhood x type and neighbourhood levels
   (leaving the home itself out).
4. **Similar homes and land.** The 10 nearest homes in a space of location, size and (for houses) lot
   size; for houses, the neighbours' land price (price per lot area).
5. **Lot.** Frontage, depth, log lot area, lot vs floor area, lot vs the neighbours' lots.
6. **Micro-market levels** (`features.MicroMarketLevels`). Inside each neighbourhood, a small tree on latitude and
   longitude groups nearby sales with similar prices into micro-markets; the level of a home's
   micro-market and neighbourhood
   over the year before it was listed.
7. **Time on market.** How long the home had already been listed without selling, compared with the
   typical time on market of similar homes nearby (20 merged areas x type x 4 size bands, falling back
   to municipality x type) in the year before its listing.
8. **Text features** (`remarks_features.py`). 45 yes/no questions from the listing description
   (condition, systems, kitchen and bath, floors, space, outside, condo building, nearby places, how it
   is sold) and a 0-3 condition score.
9. **Trees.** scikit-learn `HistGradientBoostingRegressor`: 1,000 trees, learning rate 0.03, 63 leaves,
   at least 40 homes per leaf, 60% of the columns per split, L2 = 1.0. Target: log price minus the
   index of the sale month.

### List-aware: two-model ensemble (`ListPriceEnsemble`)

The geometric mean of two designs that miss in opposite directions:

- **index version:** the same price index as off-market (the trees learn index-free prices, the
  projected index is added back), with the list price in index-free dollars as a feature;
- **ratio version:** learns log(sold price / list price) directly and multiplies the list price back.

Both use the starter features, the comparables, time on market and remarks features (not the
micro-market levels, lot or imputed floor areas), plus: the list price vs the comparables, the expected sale month,
and market heat (average log sold / list over the 90 days before listing, city-wide, for condos and
for detached houses). Trees: 300, learning rate about 0.067, absolute-error loss.

### Not used by `make_model()`

`time_effects.py`, `TimeAdjustedModel` and `LocationFeatureModel` in `models.py` are an earlier
alternative design (season and year effects by neighbourhood tier), kept for comparison and listed
as comments in `make_model()`.
