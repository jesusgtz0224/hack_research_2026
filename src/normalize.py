import unicodedata
from collections import Counter

# Marks that describe prosody, not the segment itself
STRESS = set("ˈˌ")
LENGTH = set("ːˑ")
BREAKS = set(".|‖‿")
TONE_LETTERS = set("˥˦˧˨˩")
TONE_DIGITS = set("⁰¹²³⁴⁵⁶⁷⁸⁹")

# Combining tone marks. Combining tilde (U+0303, nasalization) is deliberately NOT here.
COMBINING_TONE = {
    "\u0300",  # grave
    "\u0301",  # acute
    "\u0302",  # circumflex
    "\u0304",  # macron
    "\u030b",  # double acute
    "\u030c",  # caron
    "\u030f",  # double grave
}

TIE_BARS = {"\u0361", "\u035c"}  # t͡s, t͜s

STRIP_ALWAYS = STRESS | LENGTH | BREAKS | TONE_LETTERS | TONE_DIGITS | COMBINING_TONE


def normalize_segment(seg, affricates="keep"):
    """Return a list of 0, 1, or 2 cleaned segments.

    affricates: "keep"  -> leave tied affricates as one segment (t͡s)
                "split" -> turn t͡s into ["t", "s"]
    """
    seg = unicodedata.normalize("NFD", seg)
    seg = "".join(ch for ch in seg if ch not in STRIP_ALWAYS)
    seg = unicodedata.normalize("NFC", seg).strip()
    if not seg:
        return []
    if affricates == "split" and any(t in seg for t in TIE_BARS):
        for t in TIE_BARS:
            seg = seg.replace(t, "|")
        return [part for part in seg.split("|") if part]
    return [seg]


def normalize_word(segments, affricates="keep"):
    out = []
    for s in segments:
        out.extend(normalize_segment(s, affricates))
    return out


def normalize_words(words, affricates="keep", min_len=1):
    """Normalize a list of words and drop any that end up too short."""
    cleaned = (normalize_word(w, affricates) for w in words)
    return [w for w in cleaned if len(w) >= min_len]


def inventory(words):
    """Segment counts for a language, for eyeballing and for logging oddities."""
    return Counter(seg for w in words for seg in w)


def tie_bar_share(words):
    """Fraction of segments that are tied affricates, to compare transcription habits."""
    inv = inventory(words)
    total = sum(inv.values())
    tied = sum(n for s, n in inv.items() if any(t in s for t in TIE_BARS))
    return tied / total if total else 0.0