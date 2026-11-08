"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.
FeatureModel in models.py uses them; switch it on in make_model() to try your features.

Ideas to try:
- lot size: lot_front * lot_depth (empty for condos, which is fine for tree models)
- garage, house_style, municipality
- how pricey a community is: the median price per square foot of its past sales
  (work it out from the training sales only, never from the homes you predict)
- how prices have moved recently in the area or for that property type
- words in the remarks, such as "renovated" or "needs work"
- off-market and list-aware can use different features: list-aware has list_price
"""
import pandas as pd

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def off_market_features(homes):
    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    # Months since January 2021, so the model can follow prices over time.
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    # One 0/1 column per property type.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    return X


def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X


# ---------------------------------------------------------------------------
# MJ's location idea: neighbourhood first, then price pockets inside it.
# ---------------------------------------------------------------------------
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class LocationLevels:
    """Price level of each home's neighbourhood and of its "pocket" in it.

    1. Neighbourhood: every home belongs to a `community`.
    2. Pockets: inside each community, earlier sales that sit close together
       on the map AND sold at a similar price per square foot are grouped
       into a pocket. A small decision tree on latitude and longitude draws
       the pocket borders, so a new home (with no price yet) is placed into
       a pocket by its location alone.
    3. Level: the average price per square foot of the sales in that
       community or pocket during the year BEFORE the home was listed.
       Only earlier sales are ever used, so nothing leaks from the future.
       A pocket with too few recent sales falls back to its community.

    Prices are compared as log(price per square foot): size is taken out, so
    a pocket of big houses doesn't look pricey just because they are big.
    """

    def __init__(self, min_pocket_sales=150, window_days=365, min_recent_sales=10):
        self.min_pocket_sales = min_pocket_sales   # smallest pocket, in past sales
        self.window_days = window_days             # how far back "recent" goes
        self.min_recent_sales = min_recent_sales   # fewer than this: fall back

    def fit(self, sales):
        """Learn pocket borders and remember past sales. Uses sold prices."""
        sales = sales.assign(log_ppsf=np.log(sales["sold_price"] / square_feet(sales["house_area"])))
        self.pocket_trees = {}
        for community, group in sales.groupby("community"):
            if len(group) >= 2 * self.min_pocket_sales:
                tree = DecisionTreeRegressor(min_samples_leaf=self.min_pocket_sales, random_state=0)
                tree.fit(group[["latitude", "longitude"]], group["log_ppsf"])
                self.pocket_trees[community] = tree
        sales = sales.assign(pocket=self.pocket_of(sales))
        # For fast "average of sales between two dates": sort each group's sales
        # by date and keep a running total of log price per square foot.
        self.history = {}
        for column in ["community", "pocket"]:
            for key, group in sales.groupby(column):
                group = group.sort_values("date_sold")
                days = group["date_sold"].to_numpy("datetime64[D]").astype(np.int64)
                running_total = np.concatenate([[0.0], np.cumsum(group["log_ppsf"].to_numpy())])
                self.history[(column, key)] = (days, running_total)
        return self

    def pocket_of(self, homes):
        """Name of each home's pocket, found from its location alone."""
        pocket = homes["community"].astype(str).to_numpy(dtype=object)
        for community, tree in self.pocket_trees.items():
            here = (homes["community"] == community).to_numpy()
            if here.any():
                leaf = tree.apply(homes.loc[here, ["latitude", "longitude"]])
                pocket[here] = [f"{community} #{n}" for n in leaf]
        return pocket

    def recent_level(self, column, keys, listed_days):
        """Average log price per sq ft, and count, of each home's group's sales
        in the window_days before it was listed."""
        level = np.full(len(keys), np.nan)
        count = np.zeros(len(keys))
        for key in pd.unique(keys):
            if (column, key) not in self.history:
                continue
            days, running_total = self.history[(column, key)]
            rows = np.flatnonzero(keys == key)
            start = np.searchsorted(days, listed_days[rows] - self.window_days, side="left")
            end = np.searchsorted(days, listed_days[rows], side="left")   # strictly before listing
            n = end - start
            count[rows] = n
            with np.errstate(invalid="ignore", divide="ignore"):
                level[rows] = (running_total[end] - running_total[start]) / n
        level[count < self.min_recent_sales] = np.nan
        return level, count

    def transform(self, homes):
        """The new features for a table of homes (no sold price needed)."""
        listed_days = homes["date_listed"].to_numpy("datetime64[D]").astype(np.int64)
        X = pd.DataFrame(index=homes.index)
        community_level, community_count = self.recent_level(
            "community", homes["community"].astype(str).to_numpy(dtype=object), listed_days)
        pocket_level, pocket_count = self.recent_level("pocket", self.pocket_of(homes), listed_days)
        pocket_level = np.where(np.isnan(pocket_level), community_level, pocket_level)
        log_sqft = np.log(square_feet(homes["house_area"]).to_numpy())
        X["community_level"] = community_level
        X["community_recent_sales"] = community_count
        X["pocket_level"] = pocket_level
        X["pocket_recent_sales"] = pocket_count
        # The pocket's price per sq ft times this home's size, as a log price
        X["pocket_price_guess"] = pocket_level + log_sqft
        return X
