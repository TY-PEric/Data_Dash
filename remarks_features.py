"""Text features: turn each listing description into yes/no answers.

The phrase lists were drafted with an AI assistant after reading the descriptions, which were themselves
written by a language model and reuse a limited set of phrases. Each question is a list of
phrases; a home gets 1 if its description contains any of them, otherwise 0.
Same rules for past and new homes, so the answers mean the same thing everywhere.
"""
import re
import numpy as np
import pandas as pd

QUESTIONS = {
    # ---- condition (the most important group) ----
    "cond_fully_redone": r"fully redone|fully renovated|completely renovated|top[- ]to[- ]bottom|turnkey|nothing to do but|\brenovated\b|redone with|gut[- ]renovated|reimagined|transformed|redo\b|completely redone|rebuilt",
    "cond_updated": r"\bupdated\b|refreshed|\bnewer\b|freshly painted|fresh paint throughout|recently (?:updated|replaced|redone)|upgraded|refinished|paint is fresh|new (?:countertops|flooring|vinyl|kitchen|bathroom|appliances|windows|carpet)",
    "cond_original_clean": r"original (?:finishes|kitchen|light)|finishes are original|classic original|sound and liveable|clean and liveable|clean and cared|good condition|good shape|well[- ]maintained|well[- ]kept|\bliveable\b|functional|cared for|clean and|tidy",
    "cond_needs_work": r"as-is|cosmetic overhaul|needs a full|full refresh|requires a full|full redo|blank canvas|bring your (?:vision|contractor)|roll up your sleeves|sweat equity|renovator|personal touch|modernize|worn flooring|\btired\b|\bdated\b|may need replac|needs? replac|needs waterproofing|wants fresh paint|has to go|need of updat|needs? updat|may be replaced|ready for (?:a|your) (?:refresh|update|reno)|fixer|handyman|project\b|your vision",
    "cond_as_is": r"as-is",
    "bones_mentioned": r"great bones|good bones|bones are (?:solid|sound|good)",
    # ---- big systems ----
    "new_systems": r"(?:new|newer|updated|upgraded|high-efficiency) (?:roof|furnace|windows|electrical|plumbing|central air)|mechanicals have been|serviced regularly|roof and windows|furnace and central",
    "dated_systems": r"dated electrical|older windows|original windows|older furnace|older appliances|electrical and plumbing (?:need|are original)",
    # ---- kitchen and bath ----
    "premium_counters": r"quartz|stone counter|granite|marble|custom kitchen|large island|built-in appliances|chef'?s kitchen",
    "stainless_appliances": r"stainless|s/s appliances",
    "luxury_bath": r"spa-like|heated (?:bathroom )?floors|frameless glass|rain shower|soaker",
    # ---- floors and finishes ----
    "hardwood": r"hardwood",
    "laminate_vinyl": r"laminate|vinyl plank|vinyl flooring",
    "parquet_broadloom": r"parquet|broadloom|carpet",
    "lighting_ceilings": r"pot lights|designer light|smooth ceilings",
    # ---- space ----
    "finished_basement": r"finished basement|basement is finished|basement apartment|in-law|income suite|basement suite",
    "unfinished_basement": r"unfinished basement|waterproofing and finishing|basement needs",
    "separate_entrance": r"separate (?:side )?entrance|side entrance",
    "home_office_den": r"home office|\bden\b",
    "walk_in_closet": r"walk-in closet",
    "family_room": r"family room",
    "open_concept": r"open-concept|open concept",
    # ---- outside ----
    "good_backyard": r"private backyard|fenced (?:yard|backyard)|south-facing backyard|deep lot|large backyard",
    "deck_patio": r"\bdeck\b|patio",
    "move_in_ready": r"move-in ready|ready to move|move right in",
    "trees_street": r"mature trees|tree-lined",
    "porch_curb": r"front porch|curb appeal|interlocking",
    "garage": r"garage",
    # ---- condo building ----
    "concierge": r"concierge",
    "amenities": r"\bgym\b|party room|guest suites|fitness",
    "rooftop_terrace": r"rooftop terrace",
    "view": r"unobstructed view|skyline view|lake view|\bviews?\b",
    "floor_to_ceiling": r"floor-to-ceiling",
    "balcony": r"balcony",
    "locker": r"locker",
    "ensuite_laundry": r"ensuite laundry|in-suite laundry",
    # ---- nearby places ----
    "near_transit": r"transit|\bttc\b|subway|\bgo station|streetcar",
    "near_highway": r"highway",
    "near_ravine_parks": r"ravine|parks|trails",
    "near_schools": r"schools?\b",
    "near_downtown": r"downtown",
    "near_shops": r"shops|restaurants|cafes|grocery",
    # ---- how it's sold ----
    "motivated_seller": r"seller is motivated|motivated seller|motivated and",
    "urgency": r"won'?t last|don'?t miss|act (?:fast|quickly)|see it this week|schedule a viewing soon",
    "investor_pitch": r"investors?\b|potential|opportunity",
}


_CACHE = None   # answers already worked out, by description text (same text -> same answers)


def remark_answers(remarks):
    """yes/no (1/0) answers for every question, one row per description."""
    global _CACHE
    text = remarks.fillna("")
    todo = pd.Index(text.unique())
    if _CACHE is not None:
        todo = todo.difference(_CACHE.index)
    if len(todo):
        new = _answer(pd.Series(todo, index=todo))
        _CACHE = new if _CACHE is None else pd.concat([_CACHE, new])
    out = _CACHE.reindex(text.to_numpy())
    out.index = remarks.index
    return out


def _answer(remarks):
    text = remarks.fillna("").str.lower()
    out = pd.DataFrame({name: text.str.contains(pat, regex=True).astype(np.int8) for name, pat in QUESTIONS.items()}, index=remarks.index)
    # when a description has both, the "needs work" signals win (e.g. "every wall wants fresh paint")
    out.loc[out["cond_needs_work"] == 1, "cond_updated"] = 0
    # one condition score from worst to best: needs work 0, original 1, updated 2, fully redone 3
    score = np.full(len(out), np.nan)
    score[out["cond_original_clean"].to_numpy() == 1] = 1
    score[out["cond_updated"].to_numpy() == 1] = 2
    score[out["cond_fully_redone"].to_numpy() == 1] = 3
    score[out["cond_needs_work"].to_numpy() == 1] = 0
    out["condition_score"] = score
    return out
