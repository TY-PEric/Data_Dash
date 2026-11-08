"""Index-adjusted gradient-boosted valuation model (comparable sales + price index).

TIME   A monthly price index for each home type, adjusted for location (a month where more homes
       sell in pricey neighbourhoods doesn't look like a price rise). The index is split into
       a repeating season and a deseasonalised level. Every past sale is converted to "index-free"
       dollars; the tree learns those; the prediction adds back the index projected forward:
       latest deseasonalised level (average of the last few months) + the season of the sale month.
LOCATION  Comparable sales: for every home, the index-free price per sq ft of its nearest sales
       of the same type (10 and 30 nearest), of its nearest sales of any type, and its
       neighbourhood x type level. For a training sale its own price is always left out.
TREE   HistGradientBoosting on log price, more trees and leaves than the starter's default.

Only the training sales are ever used to build any of this, so validation stays honest.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

import features

TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse", "Freehold Townhouse", "Co-Op Apt"]
INDEX_OF = {"Co-Op Apt": "Condo Apt"}      # co-ops are too few for their own index: they follow condos
KX = 0.723                                 # a degree of longitude is ~0.72 of a degree of latitude here
KM = 111.0


def month_number(dates):
    return (dates.dt.year * 12 + dates.dt.month - 1).to_numpy()


def size_range(homes):
    """(is_range, low, high, middle) from house_area: '622' is exact, '900-1099' is a range."""
    ha = homes["house_area"].astype(str)
    is_range = ha.str.contains("-").to_numpy()
    lo = pd.to_numeric(ha.str.split("-").str[0], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(ha.str.split("-").str[-1], errors="coerce").to_numpy(float)
    return is_range, np.where(is_range, lo, np.nan), np.where(is_range, hi, np.nan), features.square_feet(homes["house_area"]).to_numpy()


class FloorAreaImputer:
    """Estimates the true size of range-size homes from their other facts (never from prices).
    Learns from exact-size homes, each given a pretend range like the ones sellers use."""

    def _X(self, homes, lo, hi):
        X = pd.DataFrame({"bedrooms": homes["bedrooms"].to_numpy(float), "washrooms": homes["washrooms"].to_numpy(float),
                          "parking": homes["parking_total"].to_numpy(float), "garage": homes["garage"].to_numpy(float),
                          "lot_front": homes["lot_front"].to_numpy(float), "lot_depth": homes["lot_depth"].to_numpy(float),
                          "lat": homes["latitude"].to_numpy(), "lon": homes["longitude"].to_numpy(), "range_lo": lo, "range_hi": hi})
        for t in TYPES:
            X["is_" + t] = (homes["house_type_name"] == t).to_numpy().astype(np.int8)
        for st in self.styles:
            X["style_" + st] = (homes["house_style"] == st).to_numpy().astype(np.int8)
        return X

    def fit(self, s):
        self.styles = sorted(s["house_style"].unique())
        is_range, lo, hi, _ = size_range(s)
        buckets = pd.DataFrame({"t": s["house_type_name"].to_numpy()[is_range], "lo": lo[is_range], "hi": hi[is_range]}).value_counts().rename("n").reset_index()
        ex = s[~is_range]
        size = features.square_feet(ex["house_area"]).to_numpy()
        rng = np.random.default_rng(0)
        plo, phi = np.full(len(ex), np.nan), np.full(len(ex), np.nan)
        typ = ex["house_type_name"].to_numpy()
        for t in np.unique(typ):
            B = buckets[buckets.t == t]
            rows = np.flatnonzero(typ == t)
            if not len(B) or not len(rows):
                continue
            blo, bhi, bn = B.lo.to_numpy(), B.hi.to_numpy(), B.n.to_numpy(float)
            # every home x every range: weight = how often sellers use that range, if the home's size fits in it
            w = ((blo[None, :] <= size[rows, None]) & (bhi[None, :] >= size[rows, None])) * bn[None, :]
            tot = w.sum(axis=1)
            ok = tot > 0
            pick = (np.cumsum(w[ok], axis=1) / tot[ok, None] >= rng.random(ok.sum())[:, None]).argmax(axis=1)
            plo[rows[ok]], phi[rows[ok]] = blo[pick], bhi[pick]
        ok = ~np.isnan(plo)
        self.model = None                       # too few exact-size homes to learn from: use the middle of the range
        if ok.sum() >= 200:
            self.model = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(self._X(ex[ok], plo[ok], phi[ok]), np.log(size[ok]))
        return self

    def estimate(self, homes):
        is_range, lo, hi, mid = size_range(homes)
        out = mid.copy()
        if is_range.any() and self.model is not None:
            h = homes[is_range]
            out[is_range] = np.clip(np.exp(self.model.predict(self._X(h, lo[is_range], hi[is_range]))), lo[is_range], hi[is_range])
        return out


class MixAdjustedPriceIndex:
    """log price-per-sqft index by home type and month, adjusted for neighbourhood mix."""

    def __init__(self, level_months=3, smooth=3, season_by="group", trend=0.0):
        self.level_months, self.smooth, self.season_by, self.trend = level_months, smooth, season_by, trend

    def fit(self, s):
        typ = s["grp"].to_numpy()
        m = month_number(s["date_sold"])
        lp = s["lp"].to_numpy()
        key = s["community"].astype(str).to_numpy() + "|" + s["house_type_name"].to_numpy()
        idx = pd.Series(0.0, index=pd.MultiIndex.from_arrays([typ, m]))
        idx = idx[~idx.index.duplicated()]
        adj = np.zeros(len(s))
        for _ in range(3):                  # alternate: neighbourhood level <-> monthly index
            loc = pd.Series(lp - adj).groupby(key).transform("mean").to_numpy()
            raw = pd.Series(lp - loc).groupby([typ, m]).median()
            adj = raw.reindex(pd.MultiIndex.from_arrays([typ, m])).to_numpy()
        self.months = np.arange(m.min(), m.max() + 1)
        self.last = m.max()
        self.raw, self.smoothed, self.season, self.level = {}, {}, {}, {}
        for t in raw.index.get_level_values(0).unique():
            r = raw[t].reindex(self.months).interpolate(limit_direction="both")
            sm = r.rolling(self.smooth, center=True, min_periods=1).mean()
            # season: deviation from a centred 12-month average, averaged by calendar month
            trend = r.rolling(12, center=True, min_periods=12).mean()
            dev = (r - trend).dropna()
            cal = dev.index % 12
            season = dev.groupby(cal).mean().reindex(range(12)).fillna(0)
            season -= season.mean()
            self.raw[t], self.smoothed[t], self.season[t] = r, sm, season
        if self.season_by == "type":      # every tier of a type shares that type's season (steadier)
            for t in self.raw:
                same = [u for u in self.raw if u.split("|")[0] == t.split("|")[0]]
                self.season[t] = sum(self.season[u] for u in same) / len(same)
        for t, r in self.raw.items():
            desea = r - self.season[t].reindex(r.index % 12).to_numpy()
            self.level[t] = desea.iloc[-self.level_months:].mean()
            # recent trend: change per month of the deseasonalised level over the last 6 months
            last6 = desea.iloc[-6:]
            self.slope = getattr(self, "slope", {})
            self.slope[t] = np.polyfit(np.arange(len(last6)), last6.to_numpy(), 1)[0] if len(last6) >= 4 else 0.0
        return self

    def at_sale(self, homes):
        """index value for past sales, in their own sale month."""
        typ = homes["grp"].to_numpy()
        m = month_number(homes["date_sold"])
        out = np.zeros(len(homes))
        for t in np.unique(typ):
            mine = typ == t
            out[mine] = self.smoothed[t].reindex(m[mine]).to_numpy()
        return out

    def forecast(self, homes, month):
        """projected index for future months: latest deseasonalised level + that month's season."""
        typ = homes["grp"].to_numpy()
        out = np.zeros(len(homes))
        for t in np.unique(typ):
            mine = typ == t
            ahead = month[mine] - (self.last - (self.level_months - 1) / 2)     # months past the middle of the level window
            out[mine] = self.level[t] + self.trend * self.slope[t] * ahead + self.season[t].reindex(month[mine] % 12).to_numpy()
        return out


class IndexAdjustedGBM:
    def __init__(self, list_aware=False, level_months=3, tuned=False, loss="squared_error", comps=True,
                 index_tiers=1, season_by="group", extra_comps=False, n_iter=None, trend=0.0, ratio_target=False, pockets=False, remarks=False, extras=(), similar_weight=2.0, tree=None):
        self.list_aware, self.comps, self.index_tiers, self.extra_comps = list_aware, comps, index_tiers, extra_comps
        self.ratio_target = ratio_target and list_aware     # list-aware: learn sold/list, no index forecast
        self.pockets = pockets                               # also add neighbourhood and micro-market levels
        self.remarks = remarks                               # also add yes/no answers from the descriptions
        self.extras = set(extras)                            # remaining factors: lot, style, garage, muni, size_range, ratios
        self.similar_weight = similar_weight                 # how much size/lot similarity counts vs map distance
        self.index = MixAdjustedPriceIndex(level_months=level_months, season_by=season_by, trend=trend)
        params = dict(max_iter=1200, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=30,
                      l2_regularization=1.0, early_stopping=False) if tuned else {}
        if n_iter:
            params = dict(max_iter=n_iter, learning_rate=0.1 * 100 / n_iter * 2 if n_iter > 200 else 0.1)
        if tree:                                             # tree tuning: settings for the boosted trees
            params = dict(tree)
        self.model = HistGradientBoostingRegressor(loss=loss, random_state=0, **params)

    # ---------------- index groups: home type, optionally split into neighbourhood price tiers ----------------
    def _fit_groups(self, s):
        rel = s["lp"] - s.groupby(["house_type_name", s["date_sold"].dt.year])["lp"].transform("median")
        level = rel.groupby(s["community"]).median()
        self.tier_of = (pd.qcut(level.rank(method="first"), self.index_tiers, labels=False) + 1) if self.index_tiers > 1 else level * 0 + 1

    def _groups(self, homes):
        typ = homes["house_type_name"].map(lambda t: INDEX_OF.get(t, t))
        tier = homes["community"].map(self.tier_of).fillna(1).astype(int).astype(str)
        split = typ.isin(["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse"])
        return np.where(split, typ + "|" + tier, typ + "|1")

    # ---------------- location features ----------------
    def _xy(self, homes):
        return np.c_[homes["latitude"].to_numpy(), homes["longitude"].to_numpy() * KX]

    def _fit_location(self, s):
        self.sales = s
        self.nn_type, self.nn_sqft = {}, {}
        for t in TYPES:
            mine = (s["house_type_name"] == t).to_numpy()
            if mine.sum() > 40:
                self.nn_type[t] = (NearestNeighbors(n_neighbors=31).fit(self._xy(s[mine])), s.loc[mine, "adj"].to_numpy())
                self.nn_sqft[t] = np.log(features.square_feet(s.loc[mine, "house_area"])).to_numpy()
        self.type_mean = s.groupby("house_type_name")["adj"].mean()
        rel = s["adj"].to_numpy() - s["house_type_name"].map(self.type_mean).to_numpy()
        self.nn_all = (NearestNeighbors(n_neighbors=21).fit(self._xy(s)), rel)
        key = s["community"].astype(str) + "|" + s["house_type_name"]
        g = s.groupby(key)["adj"]
        self.ct_sum, self.ct_n = g.sum(), g.size()
        g2 = pd.Series(rel, index=s.index).groupby(s["community"])
        self.c_sum, self.c_n = g2.sum(), g2.size()

    def _location_features(self, homes, own):
        """own: index-free log ppsf of the home itself (training rows) or None (new homes)."""
        X = pd.DataFrame(index=homes.index)
        xy = self._xy(homes)
        typ = homes["house_type_name"].to_numpy()
        self_shift = 1 if own is not None else 0
        for name, k in [("comps10", 10), ("comps30", 30)]:
            X[name] = np.nan; X[name + "_km"] = np.nan
        for t, (nn, vals) in self.nn_type.items():
            mine = typ == t
            if not mine.any():
                continue
            dist, ind = nn.kneighbors(xy[mine], n_neighbors=30 + self_shift)
            dist, ind = dist[:, self_shift:], ind[:, self_shift:]      # leave the home itself out
            v = vals[ind]
            X.loc[mine, "comps10"] = np.median(v[:, :10], axis=1)
            X.loc[mine, "comps30"] = np.median(v, axis=1)
            X.loc[mine, "comps10_km"] = dist[:, :10].mean(axis=1) * KM
            X.loc[mine, "comps30_km"] = dist.mean(axis=1) * KM
            if self.extra_comps:
                X.loc[mine, "comps5"] = np.median(v[:, :5], axis=1)
                X.loc[mine, "comps10_logsqft"] = np.median(self.nn_sqft[t][ind[:, :10]], axis=1)
        nn, rel = self.nn_all
        dist, ind = nn.kneighbors(xy, n_neighbors=20 + self_shift)
        X["comps_any20"] = np.median(rel[ind[:, self_shift:]], axis=1) + pd.Series(typ).map(self.type_mean).to_numpy()
        key = homes["community"].astype(str) + "|" + homes["house_type_name"]
        s, n = key.map(self.ct_sum).to_numpy(), key.map(self.ct_n).to_numpy()
        if own is not None:
            s, n = s - own, n - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            X["hood_type_level"] = np.where(n >= 5, s / n, np.nan)
            X["hood_type_sales"] = n
        cs, cn = homes["community"].map(self.c_sum).to_numpy(), homes["community"].map(self.c_n).to_numpy()
        if own is not None:
            cs = cs - (own - pd.Series(typ).map(self.type_mean).to_numpy()); cn = cn - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            X["hood_level"] = cs / cn + pd.Series(typ).map(self.type_mean).to_numpy()
        return X

    # ---------------- similar comparables and the land price ----------------
    HOUSES = ["Detached", "Semi-Detached", "Freehold Townhouse"]

    def _similar_space(self, homes, t):
        """map position in km, plus house size and (for houses) lot size, scaled by the similarity weight."""
        w = self.similar_weight
        cols = [homes["latitude"].to_numpy() * KM, homes["longitude"].to_numpy() * KX * KM,
                w * np.log(features.square_feet(homes["house_area"]).to_numpy())]
        if t in self.HOUSES and np.isfinite(self.median_lot.get(t, np.nan)):
            lot = np.log((homes["lot_front"] * homes["lot_depth"]).to_numpy(float))
            cols.append(w * np.where(np.isfinite(lot), lot, self.median_lot[t]))   # missing lot -> the type's usual lot
        return np.c_[tuple(cols)]

    def _fit_similar(self, s):
        self.nn_similar, self.nn_land, self.median_lot = {}, {}, {}
        for t in TYPES:
            mine = (s["house_type_name"] == t).to_numpy()
            if mine.sum() <= 40:
                continue
            part = s[mine]
            if t in self.HOUSES:
                lot = np.log((part["lot_front"] * part["lot_depth"]).to_numpy(float))
                self.median_lot[t] = np.nanmedian(lot) if np.isfinite(lot).any() else np.nan
                # land price: time-adjusted price divided by lot area (in logs: adj + log sqft - log lot)
                land = part["adj"].to_numpy() + np.log(features.square_feet(part["house_area"]).to_numpy()) - lot
                ok = np.isfinite(land)
                if ok.sum() > 40:
                    self.nn_land[t] = (NearestNeighbors(n_neighbors=11).fit(self._xy(part[ok])), land[ok])
            self.nn_similar[t] = (NearestNeighbors(n_neighbors=11).fit(self._similar_space(part, t)), part["adj"].to_numpy())

    def _similar_features(self, homes, own):
        X = pd.DataFrame(index=homes.index, columns=["similar10", "similar10_gap", "land_price10"], dtype=float)
        typ, shift = homes["house_type_name"].to_numpy(), (1 if own is not None else 0)
        for t, (nn, vals) in self.nn_similar.items():
            mine = typ == t
            if not mine.any():
                continue
            dist, ind = nn.kneighbors(self._similar_space(homes[mine], t), n_neighbors=10 + shift)
            X.loc[mine, "similar10"] = np.median(vals[ind[:, shift:]], axis=1)
            X.loc[mine, "similar10_gap"] = dist[:, shift:].mean(axis=1)          # how far away / how different they are
            if t in self.nn_land:
                nl, land = self.nn_land[t]
                _, ind2 = nl.kneighbors(self._xy(homes[mine]), n_neighbors=10 + shift)
                X.loc[mine, "land_price10"] = np.median(land[ind2[:, shift:]], axis=1)
        return X

    # ---------------- time on market ----------------
    def _fit_dom(self, s):
        """Groups: 20 merged areas (nearby neighbourhoods) x home type x 4 size bands; fallback municipality x type.
        For each group, remember every past sale's days on market by sale date, to get the typical days on market
        in the year before any listing date (only sales finished before that date)."""
        from sklearn.cluster import KMeans
        cen = s.groupby("community")[["latitude", "longitude"]].mean()
        cen["longitude"] *= KX
        k = min(20, len(cen))
        self.area_of = pd.Series(KMeans(k, n_init=10, random_state=0).fit_predict(cen), index=cen.index)
        sq = features.square_feet(s["house_area"])
        self.size_cuts = {t: np.nanquantile(sq[s["house_type_name"] == t], [0.25, 0.5, 0.75]) for t in s["house_type_name"].unique()}
        self.dom_hist = {}
        for name, keys in [("fine", self._dom_keys(s, fine=True)), ("broad", self._dom_keys(s, fine=False))]:
            q = pd.DataFrame({"key": keys, "day": s["date_sold"].to_numpy("datetime64[D]").astype(np.int64),
                              "v": np.log1p(s["days_on_market"].to_numpy(float))}).dropna().sort_values("day")
            self.dom_hist[name] = {k: (g["day"].to_numpy(), np.concatenate([[0], np.cumsum(g["v"].to_numpy())])) for k, g in q.groupby("key")}

    def _dom_keys(self, homes, fine):
        typ = homes["house_type_name"].astype(str)
        if not fine:
            return (homes["municipality"].astype(str) + "|" + typ).to_numpy()
        sq = features.square_feet(homes["house_area"]).to_numpy()
        band = np.array([np.searchsorted(self.size_cuts.get(t, [np.inf] * 3), v) if np.isfinite(v) else 1 for t, v in zip(typ, sq)])
        area = homes["community"].map(self.area_of).fillna(-1).astype(int).astype(str)
        return (area + "|" + typ + "|" + band.astype(str)).to_numpy()

    def _typical_dom(self, homes, name, listed):
        keys = self._dom_keys(homes, fine=(name == "fine"))
        val, cnt = np.full(len(homes), np.nan), np.zeros(len(homes))
        for k in pd.unique(keys):
            if k not in self.dom_hist[name]:
                continue
            days, tot = self.dom_hist[name][k]
            rows = np.flatnonzero(keys == k)
            a, b = np.searchsorted(days, listed[rows] - 365), np.searchsorted(days, listed[rows])
            cnt[rows] = b - a
            with np.errstate(invalid="ignore", divide="ignore"):
                val[rows] = (tot[b] - tot[a]) / (b - a)
        return val, cnt

    def _dom_features(self, homes, own):
        X = pd.DataFrame(index=homes.index)
        listed_ts = homes["date_listed"]
        # how long it had already sat: until the start of its sale month (past sales) or until the first unseen month (new homes)
        if own is not None:
            start = homes["date_sold"].dt.to_period("M").dt.start_time
        else:
            start = pd.Series(pd.Period(year=self.first_unseen // 12, month=self.first_unseen % 12 + 1, freq="M").start_time, index=homes.index)
        sat = (start - listed_ts).dt.days.clip(lower=0).to_numpy(float)
        listed = listed_ts.to_numpy("datetime64[D]").astype(np.int64)
        fine, n = self._typical_dom(homes, "fine", listed)
        broad, _ = self._typical_dom(homes, "broad", listed)
        typical = np.where(n >= 20, fine, broad)                     # too few similar sales nearby -> municipality x type
        X["already_sat_days"] = sat
        X["typical_dom"] = np.expm1(typical)
        X["staleness"] = np.log1p(sat) - typical                     # how long it sat compared with what's normal for homes like it
        return X

    # ---------------- the remaining factors ----------------
    STYLES = ["Apartment", "2-Storey", "Bungalow", "3-Storey", "2 1/2 Storey", "Stacked Townhse", "Loft", "Bungalow-Raised",
              "1 1/2 Storey", "Multi-Level", "Backsplit 4", "Sidesplit 4", "Backsplit 3", "Sidesplit 3", "Backsplit 5", "1 Storey/Apt"]
    MUNIS = ["Toronto", "North York", "Scarborough", "Etobicoke", "York"]
    STYLE_GROUPS = ["Apartment", "2-Storey", "Bungalow", "3-Storey", "2 1/2 Storey", "Stacked Townhse", "Loft",
                    "Bungalow-Raised", "1 1/2 Storey", "Multi-Level", "Split-level"]
    STOREYS = {"Bungalow": 1, "Bungalow-Raised": 1, "1 1/2 Storey": 1.5, "Split-level": 1.5, "2-Storey": 2,
               "2 1/2 Storey": 2.5, "3-Storey": 3}          # apartments, lofts, stacked and multi-level: left blank

    def _extra_features(self, homes, own):
        X = pd.DataFrame(index=homes.index)
        sqft = features.square_feet(homes["house_area"]).to_numpy()
        if "lot" in self.extras:
            front, depth = homes["lot_front"].to_numpy(float), homes["lot_depth"].to_numpy(float)
            X["lot_front"], X["lot_depth"] = front, depth
            with np.errstate(invalid="ignore", divide="ignore"):
                X["log_lot"] = np.log(front * depth)
                X["lot_per_sqft"] = X["log_lot"] - np.log(sqft)          # land compared with floor area
            # lot compared with the nearest same-type neighbours' lots
            X["lot_vs_comps"] = np.nan
            xy, typ = self._xy(homes), homes["house_type_name"].to_numpy()
            shift = 1 if own is not None else 0
            for t, (nn, lots) in self.nn_lot.items():
                mine = typ == t
                if mine.any():
                    _, ind = nn.kneighbors(xy[mine], n_neighbors=10 + shift)
                    X.loc[mine, "lot_vs_comps"] = X.loc[mine, "log_lot"].to_numpy() - np.nanmedian(lots[ind[:, shift:]], axis=1)
            X = X.join(self._similar_features(homes, own))
        if "style" in self.extras:
            # grouped styles: the five side/back splits together, "1 Storey/Apt" with Apartment
            st = homes["house_style"].replace({"Sidesplit 3": "Split-level", "Sidesplit 4": "Split-level", "Backsplit 3": "Split-level",
                                               "Backsplit 4": "Split-level", "Backsplit 5": "Split-level", "1 Storey/Apt": "Apartment"})
            for g in self.STYLE_GROUPS:
                X["style_" + g] = (st == g).astype(np.int8).to_numpy()
            # number of storeys: the clearest pattern in the style table (fewer storeys -> more per sq ft)
            X["storeys"] = st.map(self.STOREYS).to_numpy(float)
        if "garage" in self.extras:
            X["garage_spaces"] = homes["garage"].to_numpy(float)
        if "muni" in self.extras:
            for mu in self.MUNIS:
                X["muni_" + mu] = (homes["municipality"] == mu).astype(np.int8).to_numpy()
        if "size_range" in self.extras or "size" in self.extras:
            X["size_is_range"] = homes["house_area"].astype(str).str.contains("-").astype(np.int8).to_numpy()
        if "size" in self.extras:
            is_range, lo, hi, _ = size_range(homes)
            X["range_lo"], X["range_hi"] = lo, hi                         # idea 1: the range itself
            est = self.sizer.estimate(homes)
            X["size_est"], X["log_size_est"] = est, np.log(est)          # idea 2: the estimate, as an extra column
        if "ratios" in self.extras:
            beds = homes["bedrooms"].to_numpy(float); baths = homes["washrooms"].to_numpy(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                X["sqft_per_bedroom"] = sqft / np.maximum(beds, 1)
                X["baths_per_bedroom"] = baths / np.maximum(beds, 1)
        return X

    # ---------------- the model ----------------
    def _features(self, homes, own, index_at_sale):
        X = features.off_market_features(homes).drop(columns=["months"])
        X["log_sqft"] = np.log(X["square_feet"])
        if self.comps:
            X = X.join(self._location_features(homes, own))
        if self.pockets:
            X = X.join(self.levels.transform(homes))
        if self.extras - {"dom"}:
            X = X.join(self._extra_features(homes, own))
        if "dom" in self.extras:
            X = X.join(self._dom_features(homes, own))
        if self.remarks:
            import remarks_features
            X = X.join(remarks_features.remark_answers(homes["remarks"]))
        if self.list_aware:
            # asking price in index-free dollars, and how it compares with the comps
            X["list_adj"] = np.log(homes["list_price"].to_numpy()) - index_at_sale
            if self.comps:
                X["list_vs_comps"] = X["list_adj"] - (X["comps10"] + X["log_sqft"])
            X["sale_month"] = self._sale_cal
            # market heat: average log(sold/list) of sales in the 90 days before this home was listed
            listed = homes["date_listed"].to_numpy("datetime64[D]").astype(np.int64)
            for name, (days, total) in self.heat.items():
                a, b = np.searchsorted(days, listed - 90), np.searchsorted(days, listed)
                with np.errstate(invalid="ignore", divide="ignore"):
                    X["heat_" + name] = np.where(b > a, (total[b] - total[a]) / np.maximum(b - a, 1), np.nan)
        return X

    def _fit_heat(self, s):
        self.heat = {}
        for name, part in [("city", s)] + [(t, s[s["house_type_name"] == t]) for t in ["Condo Apt", "Detached"]]:
            q = part.sort_values("date_sold")
            r = np.log(q["sold_price"] / q["list_price"]).to_numpy()
            self.heat[name] = (q["date_sold"].to_numpy("datetime64[D]").astype(np.int64), np.concatenate([[0], np.cumsum(r)]))

    def fit(self, train):
        s = train.copy()
        s["lp"] = np.log(s["sold_price"] / features.square_feet(s["house_area"]))
        if "size" in self.extras:
            # idea 3: price per sq ft (index, comparables, pockets) measured with the estimated size, so it's cleaner
            self.sizer = FloorAreaImputer().fit(s)
            s["lp"] = np.log(s["sold_price"].to_numpy() / self.sizer.estimate(s))
        self._fit_groups(s)
        s["grp"] = self._groups(s)
        self.index.fit(s)
        idx = self.index.at_sale(s)
        s["adj"] = s["lp"] - idx
        self.first_unseen = self.index.last + 1
        if self.comps:
            self._fit_location(s)
        if "lot" in self.extras:
            self._fit_similar(s)
            self.nn_lot = {}
            for t in ["Detached", "Semi-Detached", "Freehold Townhouse"]:
                mine = ((s["house_type_name"] == t) & s["lot_front"].notna()).to_numpy()
                if mine.sum() > 40:
                    self.nn_lot[t] = (NearestNeighbors(n_neighbors=11).fit(self._xy(s[mine])), np.log(s.loc[mine, "lot_front"] * s.loc[mine, "lot_depth"]).to_numpy())
        if "dom" in self.extras:
            self._fit_dom(s)
        if self.pockets:   # micro-market levels, built from index-free (time-adjusted) prices
            self.levels = features.MicroMarketLevels(min_pocket_sales=70).fit(s.assign(sold_price=np.exp(np.log(s["sold_price"]) - idx)))
        self._sale_cal = s["date_sold"].dt.month.to_numpy()
        if self.list_aware:
            self._fit_heat(s)
        # list-aware: index at the listing month (sellers price for the market they see then)
        idx_list = self._index_at_listing(s) if self.list_aware else None
        X = self._features(s, s["adj"].to_numpy(), idx_list)
        y = np.log(s["sold_price"].to_numpy()) - idx
        if self.ratio_target:
            y = np.log(s["sold_price"] / s["list_price"]).to_numpy()
        self.model.fit(X, y)
        return self

    def _index_at_listing(self, homes):
        m = month_number(homes["date_listed"])
        typ = homes["grp"].to_numpy()
        out = np.zeros(len(homes))
        for t in np.unique(typ):
            mine = typ == t
            sm = self.index.smoothed[t]
            mm = np.clip(m[mine], sm.index.min(), sm.index.max())
            out[mine] = sm.reindex(mm).to_numpy()
        return out

    def predict(self, homes):
        homes = homes.assign(grp=self._groups(homes))
        listed = month_number(homes["date_listed"])
        sale_m = np.maximum(listed, self.first_unseen)             # can't have sold before the data ends
        fc = self.index.forecast(homes, sale_m)
        self._sale_cal = sale_m % 12 + 1
        idx_list = self._index_at_listing(homes) if self.list_aware else None
        X = self._features(homes, None, idx_list)
        if self.ratio_target:
            return homes["list_price"].to_numpy() * np.exp(self.model.predict(X))
        return np.exp(self.model.predict(X) + fc)


class ListPriceEnsemble:
    """List-aware: average (in log terms) of two designs that miss in opposite directions:
    the index version (list price in index-free dollars, projected index added back) and the
    ratio version (learns sold / list directly, no projection)."""

    def __init__(self, **opt):
        self.a = IndexAdjustedGBM(list_aware=True, **opt)
        self.b = IndexAdjustedGBM(list_aware=True, ratio_target=True, **opt)

    def fit(self, train):
        self.a.fit(train); self.b.fit(train)
        return self

    def predict(self, homes):
        return np.exp((np.log(self.a.predict(homes)) + np.log(self.b.predict(homes))) / 2)
