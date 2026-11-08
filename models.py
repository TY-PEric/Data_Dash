"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import features


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    # Combined version: Claude's price index + comparable sales, MJ's pockets, timeless prices, no months feature,
    # market heat and sale month for list-aware, MJ's yes/no answers from the descriptions (remarks_features.py),
    # off-market: lot size, lot vs neighbours, MJ's similar comparables (size + lot) and the neighbours' land price,
    # and for range sizes ("900-1099"): the range itself, MJ's estimated size, and comparables priced with estimated sizes.
    # Both tracks: time on market (how long a home had already sat unsold, compared with what's normal for similar
    # homes in its area: 20 merged areas x type x 4 size bands, falling back to municipality x type).
    # MJ's Option B is kept below as a commented-out line.
    import claude_model
    if track == "list_aware":
        return claude_model.ListAwareBlend(extra_comps=True, n_iter=300, loss="absolute_error", remarks=True,
                                           extras=("dom",))   # combined + remarks + time on market
        # return TimeAdjustedModel(features.list_aware_features, list_aware=True, sale_month_feature=True)   # MJ's Option B
        # return LocationFeatureModel(features.list_aware_features, min_pocket_sales=70)
        # return ListPriceBaseline()
        # return FeatureModel(features.list_aware_features)
    return claude_model.ClaudeModel(extra_comps=True, n_iter=300, pockets=True, remarks=True,
                                    extras=("lot", "size", "dom"), similar_weight=2.0,
                                    # tuned trees: 1000 slower-learning, bigger trees, each seeing 60% of the columns
                                    tree=dict(max_iter=1000, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=40,
                                              max_features=0.6, l2_regularization=1.0))
    # comps + MJ pockets + remarks + lot + range sizes + time on market + tuned trees
    # return TimeAdjustedModel(features.off_market_features)   # MJ's Option B
    # return LocationFeatureModel(features.off_market_features, min_pocket_sales=70)
    # return NearestSalesBaseline(k=10)
    # return FeatureModel(features.off_market_features)


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))


class LocationFeatureModel:
    """FeatureModel plus MJ's neighbourhood and pocket price levels
    (features.LocationLevels). The levels are rebuilt from the training
    sales every time the model is trained, so they only know the past.
    use="community" adds only the neighbourhood level, to compare the two.
    """

    def __init__(self, make_features, use="pocket", min_pocket_sales=150):
        self.make_features = make_features
        self.use = use
        self.min_pocket_sales = min_pocket_sales
        self.model = HistGradientBoostingRegressor(random_state=0)

    def all_features(self, homes):
        X = self.make_features(homes).join(self.levels.transform(homes))
        if self.use == "community":
            X = X.drop(columns=["pocket_level", "pocket_recent_sales", "pocket_price_guess"])
        return X

    def fit(self, train):
        self.levels = features.LocationLevels(self.min_pocket_sales).fit(train)
        self.model.fit(self.all_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.all_features(homes)))


# ---------------------------------------------------------------------------
# MJ's time plan (Option B): take time out, let the tree price the home, put time back once.
# ---------------------------------------------------------------------------
import pandas as pd
import time_effects as te

_TIME_CACHE = {}   # the same training sales give the same time numbers for both tracks


def time_numbers(train, k=100):
    """Season and year effects for every group, built from these training sales only."""
    cutoff = (train["date_sold"].max().to_period("M") + 1).start_time      # first month we can't see
    key = (cutoff, len(train), k)
    if key not in _TIME_CACHE:
        _TIME_CACHE.clear()
        _TIME_CACHE[key] = (cutoff, te.build(train, cutoff, k=k))
    return _TIME_CACHE[key]


class TimeAdjustedModel:
    """Option B.

    fit:     1. season and year effects per tier x type group (yearly difference method + shrinkage)
             2. every past sale -> timeless price = price / season effect / year effect (its own sale month)
             3. pocket levels built from timeless prices; no "months" feature
             4. the tree learns timeless prices (off-market) or sold / list (list-aware)
    predict: timeless price x season effect x year guess.
             year guess = average of the "latest month" and "last 3 months" year effects (MJ's choice).
             The home's sale month is taken as its listing month, or the first month after the training
             data if it was listed before that (it can't have sold before then).
    """

    KEYS = ["house_type_name", "tier"]

    def __init__(self, make_features, list_aware=False, k=100, min_pocket_sales=70, sale_month_feature=False, coop_like_condo=False):
        self.make_features, self.list_aware, self.k = make_features, list_aware, k
        self.sale_month_feature = sale_month_feature   # list-aware: tell the tree the calendar month of the sale
        self.coop_like_condo = coop_like_condo         # co-ops use condo apartments' city-wide time numbers
        self.min_pocket_sales = min_pocket_sales
        self.model = HistGradientBoostingRegressor(random_state=0)

    # ----- groups and time numbers -----
    def tiers_of(self, homes):
        tier = homes["community"].map(self.shared_tiers)
        for t, m in self.own_tiers.items():
            mine = homes["house_type_name"] == t
            tier[mine] = homes.loc[mine, "community"].map(m)
        return tier

    def borrow(self, q, got):
        """co-ops: drop their own (noisy) numbers and look up condo apartments' city-wide ones instead."""
        if self.coop_like_condo:
            coop = (q["house_type_name"] == "Co-Op Apt").to_numpy()
            got = got.copy(); got[coop] = np.nan
            q = q.copy(); q.loc[coop, "house_type_name"] = "Condo Apt"
        return q, got

    def season_of(self, homes, months):
        """log season effect for each home's group in calendar month `months` (1-12)."""
        q = pd.DataFrame({"house_type_name": homes["house_type_name"].to_numpy(), "tier": self.tiers_of(homes).to_numpy(), "cal": months})
        got = q.merge(self.season, on=self.KEYS + ["cal"], how="left")["season_s"]
        q, got = self.borrow(q, got)
        broad = q.merge(self.broad_season, on=["house_type_name", "cal"], how="left")["season"]
        return got.fillna(broad).fillna(0).to_numpy()

    def year_of_past(self, sales):
        """log year effect of each past sale's group in its own sale month."""
        q = pd.DataFrame({"house_type_name": sales["house_type_name"].to_numpy(), "tier": self.tiers_of(sales).to_numpy(),
                          "m": sales["date_sold"].dt.to_period("M").to_numpy()})
        got = q.merge(self.year, on=self.KEYS + ["m"], how="left")["year_s"]
        q, got = self.borrow(q, got)
        broad = q.merge(self.broad_year, on=["house_type_name", "m"], how="left")["year"]
        return got.fillna(broad).fillna(0).to_numpy()

    def year_guess_of(self, homes):
        q = pd.DataFrame({"house_type_name": homes["house_type_name"].to_numpy(), "tier": self.tiers_of(homes).to_numpy()})
        got = q.merge(self.guess, on=self.KEYS, how="left")["guess"]
        q, got = self.borrow(q, got)
        return got.fillna(q.merge(self.broad_guess, on="house_type_name", how="left")["guess"]).fillna(0).to_numpy()

    def sale_month_of(self, homes):
        listed = homes["date_listed"].dt.to_period("M")
        first = self.cutoff.to_period("M")
        return listed.where(listed >= first, first).dt.month.to_numpy()

    # ----- features -----
    def all_features(self, homes, time_of_sale, sale_month=None):
        X = self.make_features(homes).drop(columns=["months"], errors="ignore")
        X = X.join(self.levels.transform(homes))
        if self.list_aware and self.sale_month_feature:
            X["sale_month"] = sale_month
        if self.list_aware:
            # how far the asking price sits above what the pocket level and timing suggest
            X["list_vs_expected"] = np.log(homes["list_price"].to_numpy()) - (X["pocket_price_guess"].to_numpy() + time_of_sale)
            # market heat: average sold/list of sales in the 90 days before this home was listed
            listed = homes["date_listed"].to_numpy("datetime64[D]").astype(np.int64)
            for name, (days, total) in self.heat.items():
                a = np.searchsorted(days, listed - 90); b = np.searchsorted(days, listed)
                with np.errstate(invalid="ignore", divide="ignore"):
                    X["heat_" + name] = np.where(b > a, (total[b] - total[a]) / (b - a), np.nan)
        return X

    def remember_heat(self, train):
        """running totals of log(sold/list) by sale date, for the market-heat feature (past sales only)."""
        t = train.sort_values("date_sold")
        r = np.log(t["sold_price"] / t["list_price"]).to_numpy()
        self.heat = {"city": (t["date_sold"].to_numpy("datetime64[D]").astype(np.int64), np.concatenate([[0], np.cumsum(r)]))}

    def fit(self, train):
        self.cutoff, R = time_numbers(train, self.k)
        self.shared_tiers = R["shared_tiers"]
        past = R["sales"]
        self.own_tiers = {t: past[past.house_type_name == t].groupby("community").tier.first() for t in te.OWN_TIERS}
        self.season, self.broad_season = R["season"], R["broad_season"]
        self.year, self.broad_year = R["year"], R["broad_year"]
        last = self.cutoff.to_period("M") - 1
        def guess(table, keys, col):
            latest = table[table.m == last].groupby(keys)[col].mean()
            last3 = table[table.m.isin([last - 2, last - 1, last])].groupby(keys)[col].mean()
            return ((latest + last3) / 2).fillna(last3).rename("guess").reset_index()
        self.guess = guess(self.year, self.KEYS, "year_s")
        self.broad_guess = guess(self.broad_year, ["house_type_name"], "year")

        time_then = self.season_of(train, train["date_sold"].dt.month.to_numpy()) + self.year_of_past(train)
        timeless = train.assign(sold_price=train["sold_price"].to_numpy() / np.exp(time_then))
        self.levels = features.LocationLevels(self.min_pocket_sales).fit(timeless)    # pockets from timeless prices
        if self.list_aware:
            self.remember_heat(train)
        X = self.all_features(train, time_then, train["date_sold"].dt.month.to_numpy())
        y = np.log(train["sold_price"] / train["list_price"]) if self.list_aware else np.log(timeless["sold_price"])
        self.model.fit(X, y)
        return self

    def predict(self, homes):
        month = self.sale_month_of(homes)
        time_now = self.season_of(homes, month) + self.year_guess_of(homes)
        out = self.model.predict(self.all_features(homes, time_now, month))
        if self.list_aware:
            return homes["list_price"].to_numpy() * np.exp(out)       # list price x predicted sold/list
        return np.exp(out + time_now)                                  # timeless price x time
