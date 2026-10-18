"""Cypriot-vs-other Greek dialect scoring for short verse (marker based)."""
import re
import unicodedata


def _norm(text):
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t.replace("ς", "σ").replace("’", "'").replace("΄", "'").replace("ʼ", "'").replace("`", "'")


# Strong Cypriot markers (on accent-stripped lowercase text). Each match = 1 point.
CY_PATTERNS = [
    r"(?<!ν)τζ",              # τζαι, τζι', γλυτζύς, τζιείνος, φαρμάτζιν (not SMG ντζ: νεραντζιά)
    r"\bτσ(?:ια|ιαι|ι')\b",   # τσιαι / τσ' (older spelling of τζαι)
    r"σι[ηιεαου]",            # palatal σ: έσιει, σιηλός, ψυσιή, χέρκα->σιέρκα
    r"ψι[ηι]",                # ψιη
    r"θκ",                    # πόθκια, μμάθκια, δκυο-type clusters
    r"\bδκ",                  # δκυο, δκιάβολος
    r"φκ",                    # φκαίνω, φκάλλω
    r"ρκι[αοεηου]|\bχωρκ|\bαρκο",  # χωρκό, καρκιά, αρκοπέζουνον (not αρκετά/Τούρκοι)
    r"\b(?:κκ|ππ|ττ|μμ|σσ|λλ)\w",  # word-initial geminates: κκελλέ, ττο, μμάθκια
    r"\w(?:νν|ττ|κκ|ππ)(?:ω|ει|εισ|ουν|ουσιν)\b",  # geminate verb endings: πίννω, κόφτει->κόττει
    r"\bενν?α\b",             # future εννά / ενά
    r"\bεν\b",                # copula εν
    r"\bουλλ",                # ούλλα, ούλλοι
    r"\bεσου\b|\bεγιω\b|'γιω\b",
    r"\bδαμε\b|\bτζιαμε\b|\bτωρα\b",
    r"\bποττε\b|\bμεν\b|\bεμ\b",
    r"\bλαλ(?:ω|εισ|ει|ουν|ουσιν|ησ)\b",
    r"\bκαμν",
    r"\bβουρ",
    r"\bγρον",
    r"(?:αμεν|ασιν|ουσιν|ουσειν|ησεν|ωσεν)\b",  # -μεν/-σιν verb endings
    r"\bπασ\s+(?:τ|στ)",      # πας τη ράσιη = on
    r"ρτκ|ρθκ|ρθτ",           # καρτκιά, καρθκιά
    r"κυπρ|λευκωσ|λεμεσ|αμμοχωστ|λαρνακ|τροοδ|κερυνει|ποιηταρ|τσιατ|τσιαττ",  # explicitly Cypriot topics
]
# Markers of Cretan / other dialects or Standard Modern Greek idiom.
NON_CY_PATTERNS = [
    r"κρητ", r"μαντιναδ", r"χανι[αω]", r"ψηλορειτ", r"σφακι", r"ρεθυμν", r"λασιθ", r"ηρακλει",
    r"\bηντα\b", r"\bντα\b", r"\bεπα\b", r"\bεδα\b", r"\bτση\b", r"\bτζη\b", r"\bτζοι\b", r"\bτσοι\b", r"\bχορευγ", r"\bκιανεισ\b", r"\bλυρα",
    r"\bθα\b",  # SMG future (Cypriot uses εννά); not decisive alone
    r"\bδεν\b",  # SMG negation (Cypriot verse often uses εν/δεν both); weak
]
WEAK_NON_CY = {r"\bθα\b", r"\bδεν\b"}

CY_RE = [re.compile(p) for p in CY_PATTERNS]
NON_RE = [(re.compile(p), p in WEAK_NON_CY) for p in NON_CY_PATTERNS]


# Final -ν on neuter/accusative nouns (ψουμίν, κορμίν, την κόρην) — SMG function words excluded.
FINAL_N = re.compile(r"\b\w{3,}(?:ιν|ον|ην|αν)\b")
SMG_N = {"οταν", "πριν", "μολον", "τουλαχιστον", "λοιπον", "μονον", "ποτεν", "καθεν", "αυτον", "αυτην", "τουτον", "εκεινον", "εκεινην", "ολον", "μιαν", "ποιον", "ποιαν", "κανεν", "παρον", "μελλον", "ανθρωπον"}


def cypriot_score(text):
    t = _norm(text)
    cy = sum(len(r.findall(t)) for r in CY_RE)
    final_n = sum(1 for w in FINAL_N.findall(t) if w not in SMG_N)
    cy += 0.5 * max(0, final_n - 1)
    strong_non = sum(len(r.findall(t)) for r, weak in NON_RE if not weak)
    weak_non = sum(len(r.findall(t)) for r, weak in NON_RE if weak)
    return cy - 3 * strong_non - 0.5 * weak_non, cy, strong_non


def is_cypriot(text, min_score=2.0):
    score, cy, strong_non = cypriot_score(text)
    return strong_non == 0 and score >= min_score
