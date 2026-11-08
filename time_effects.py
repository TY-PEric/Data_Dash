"""Time effects for the alternative time-adjusted design (TimeAdjustedModel), with shrinkage.

Step 1  tiers: each neighbourhood's price level vs the city (same type, same year),
        shrunk toward 1.00 by how sure we are of it, then ranked into tiers.
        Condo apt, detached, condo townhouse, semi: shared 5 tiers.
        Freehold townhouse: its own 3 tiers (equal sales). Co-op: 1 tier.
Step 2  in each tier x type group, the yearly difference method on log price per sq ft:
        season effect per calendar month, and a year effect per month (3-month window),
        each shrunk toward a broader group: w = n / (n + k).
        Broader group = the same home type city-wide; for co-ops, condo apartments city-wide.
Everything takes a cutoff date and only uses sales before it, so step 3 can rerun it as of
the end of May in earlier years.
"""
import numpy as np, pandas as pd
import features

SHARED = ["Condo Apt", "Detached", "Condo Townhouse", "Semi-Detached"]
OWN_TIERS = {"Freehold Townhouse": 3, "Co-Op Apt": 1}
BROADER = {"Co-Op Apt": "Condo Apt"}           # co-ops lean on condo apartments


def prep(sales):
    s = sales.copy()
    s["lp"] = np.log(s.sold_price / features.square_feet(s.house_area))
    s["yr"] = s.date_sold.dt.year
    s["m"] = s.date_sold.dt.to_period("M")
    return s


# ---------------- Step 1: tiers, with shrinkage on the neighbourhood levels ----------------
def neighbourhood_levels(s):
    rel = s.lp - s.groupby(["house_type_name", "yr"]).lp.transform("median")   # log of "vs city"
    g = rel.groupby(s.community)
    raw = g.median()
    noise_var = (1.25 * g.std()) ** 2 / g.size()          # how unsure each estimate is (median is ~1.25x noisier than a mean)
    real_var = max(raw.var() - noise_var.mean(), 1e-6)    # how much neighbourhoods truly differ
    w = (real_var / (real_var + noise_var)).fillna(0)     # shrinkage weight (a neighbourhood with 1 sale: no trust)
    return pd.DataFrame({"raw": np.exp(raw), "level": np.exp(w * raw), "w": w, "sales": g.size()})


def equal_sales_tiers(levels, counts, k):
    lv = levels[counts.index].sort_values()
    cum = counts[lv.index].cumsum().shift(fill_value=0) / counts.sum()
    return np.minimum((cum * k).astype(int) + 1, k)


def assign_tiers(s, n_shared=5):
    lv = neighbourhood_levels(s)
    shared = pd.qcut(lv.level.rank(method="first"), min(n_shared, len(lv)), labels=False) + 1
    tier = s.community.map(shared)
    for t, k in OWN_TIERS.items():
        mine = s.house_type_name == t
        own = equal_sales_tiers(lv.level, s[mine].community.value_counts(), k)
        tier[mine] = s[mine].community.map(own)
    return tier.astype(int), lv, shared


# ---------------- Step 2: yearly difference method + shrinkage ----------------
def monthly_table(s, keys):
    """median log price/sqft per group per month, using a 3-month window (month and its neighbours)."""
    rows = []
    for key, g in s.groupby(keys):
        by_m = {m: grp.lp.to_numpy() for m, grp in g.groupby("m")}
        for m in pd.period_range(s.m.min(), s.m.max(), freq="M"):
            win = [by_m[x] for x in (m - 1, m, m + 1) if x in by_m]
            single = by_m.get(m, np.array([]))
            if win:
                v = np.concatenate(win)
                rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))), "m": m,
                             "win": np.median(v), "n_win": len(v),
                             "one": np.median(single) if len(single) else np.nan, "n_one": len(single)})
    t = pd.DataFrame(rows)
    t["yr"], t["cal"] = t.m.dt.year, t.m.dt.month
    return t


def effects(t, keys, last_full_year, window_groups=()):
    """season effect per calendar month and year effect per month, both in log units.
    window_groups: groups (tuples of key values) whose season effect uses the 3-month window."""
    t = t.copy()
    if window_groups:
        use = t[keys].apply(tuple, axis=1).isin(set(window_groups))
        t.loc[use, "one"], t.loc[use, "n_one"] = t.loc[use, "win"], t.loc[use, "n_win"]
    base = t[t.yr <= last_full_year]
    same_month = base.groupby(keys + ["cal"]).agg(avg=("one", "mean"), n=("n_one", "sum")).reset_index()
    same_month["season"] = same_month.avg - same_month.groupby(keys).avg.transform("mean")
    t = t.merge(same_month[keys + ["cal", "avg"]], on=keys + ["cal"])
    t["year"] = t.win - t.avg
    return same_month, t


def shrink(own, broad, n, k):
    w = n / (n + k)
    return w * own + (1 - w) * broad, w


def build(sales, cutoff, k=50, n_shared=5, window_groups=()):
    """All time numbers as of `cutoff` (only sales before it). k=0 means no shrinkage."""
    s = prep(sales[sales.date_sold < cutoff])
    s["tier"], lv, shared = assign_tiers(s, n_shared)
    last_full_year = pd.Timestamp(cutoff).year - 1
    g_season, g_t = effects(monthly_table(s, ["house_type_name", "tier"]), ["house_type_name", "tier"], last_full_year, window_groups)
    b_season, b_t = effects(monthly_table(s, ["house_type_name"]), ["house_type_name"], last_full_year)
    g_season["broad_type"] = g_season.house_type_name.map(lambda x: BROADER.get(x, x))
    g_season = g_season.merge(b_season[["house_type_name", "cal", "season"]].rename(columns={"house_type_name": "broad_type", "season": "b_season"}), on=["broad_type", "cal"])
    g_season["season_s"], g_season["w"] = shrink(g_season.season, g_season.b_season, g_season.n, k)
    g_t["broad_type"] = g_t.house_type_name.map(lambda x: BROADER.get(x, x))
    g_t = g_t.merge(b_t[["house_type_name", "m", "year"]].rename(columns={"house_type_name": "broad_type", "year": "b_year"}), on=["broad_type", "m"], how="left")
    g_t["year_s"], g_t["w"] = shrink(g_t.year, g_t.b_year, g_t.n_win, k)
    return {"sales": s, "levels": lv, "shared_tiers": shared, "season": g_season, "year": g_t, "broad_season": b_season, "broad_year": b_t}
