"""Add category / category_norm / dialect / poet to every record in data/old/raw/web_*/extracted.jsonl (idempotent).
category      : the site's own label/tag/section (string; joined labels)
category_norm : tsiattista | distichs | poiitarika | folk_song | lyrics | modern_poetry | ancient | prose_other | unlabeled
dialect       : cypriot | smg | ancient | unknown (heuristic, see common.dialect_of)
poet          : poet name if known (signature, label, or blog author for own-verse blogs)"""
import sys, os, json, re, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dialect_of, ROOT

SITE_DEFAULT = {  # (category, category_norm, default poet)
    "tsiatista_wordpress": ("Κυπριακά τσιατιστά (user submissions)", "tsiattista", None),
    "kupriakatsiatismata": ("Κυπριακα Τσιατισματα", "tsiattista", None),
    "milomena": ("Μιλλομένα ποιήματα τζαι τσιαττιστά", "tsiattista", None),
    "cyprusalive": ("Ερωτικά κυπριακά δίστιχα", "distichs", None),
    "kypriaki_dialektos": ("Κυπριακά τσιατιστά", "tsiattista", None),
    "aftodioikisi": ("Τσιαττίσματα", "tsiattista", None),
    "simerini": ("ΤΣΙΑteaseΖΩ (Σημέλα Βωνιάτη)", "tsiattista", "Σημέλα Βωνιάτη"),
    "kyriakosorokkou": ("ΤΣΙΑteaseΖΩ (Σημέλα Βωνιάτη)", "tsiattista", "Σημέλα Βωνιάτη"),
}
SITE_FORCE = {  # single-topic pages: category_norm fixed regardless of title words
    "adipsona": ("Τσιαττιστά μαθητών Ε' τάξης (Τσικνοπέμπτη)", "tsiattista", None),
    "oldboy1983": ("ΤΣΙΑΤΤΙΣΤΑ", "tsiattista", None),
    "aneforiwn": ("ποιητάρικα (μυλλωμένα)", "poiitarika", None),
    "vrysoulesgnosis": ("Ποιήματα με κυπριακή διάλεκτο", "poiitarika", None),
    "stixoi": ("stixoi.info παραδοσιακά - Κύπρος", "folk_song", None),
    "kanellatou": ("Παραδοσιακό τραγούδι - Κύπρος", "folk_song", None),
    "authormanolis": ("Ερωτικά Κυπριακά τραγούδια (Ρίμες της αγάπης)", "folk_song", None),
}
CAT_RULES = [
    (re.compile(r"τσια|τσιά|τσατ|ΤΣΙΑ|ΤΣΙΆ|Τσια", re.I), "tsiattista"),
    (re.compile(r"δίστιχ|διστιχ|ΔΙΣΤΙΧ|μαντινάδ", re.I), "distichs"),
    (re.compile(r"ποιητάρ|ΠΟΙΗΤΑΡ|λαικ\w* ποιη|λαϊκ\w* ποιη|λαική ποιήτρια|λαικη ποιητρια|ποιητάρισσα", re.I), "poiitarika"),
    (re.compile(r"δημοτικ|κάλαντα|ανωνύμων|παραδοσιακ|ΔΗΜΟΤΙΚ|ΚΑΛΑΝΤΑ", re.I), "folk_song"),
    (re.compile(r"στιχουργ|ύμνοι|τραγούδ|ΤΡΑΓΟΥΔ", re.I), "lyrics"),
    (re.compile(r"αρχαί|στασίνος|εύκλος|σώπατρος|ερμείας", re.I), "ancient"),
]
NONNAME = re.compile(r"ανακοιν|ιστορ|ποίηση|εκδηλ|διαγων|βραβ|κύπρος|ένωση|όμιλος|ημέρα|διάφορα|e-book|παραστ|στατιστ|δημοκρατ|αγγλοκρ|φίλοι|ποιήτριες|ποιητές|ποιήματα|συνταγ|facebook|γυμνασ|σχολ|ταξη|εορτ|γιορτ|ευχ|ΤΑ |ΜΑ$", re.I)


def looks_like_name(lbl):
    core = re.sub(r"[\(\[].*?[\)\]]", "", lbl).strip(" -–")
    toks = [t for t in re.split(r"[\s\-–]+", core) if t]
    if not (2 <= len(toks) <= 5) or ":" in lbl or NONNAME.search(lbl):
        return False
    return all(t[0].isupper() for t in toks if t[0].isalpha()) and not lbl.isupper()


def norm_from(texts):
    j = " ".join(t for t in texts if t)
    for rx, n in CAT_RULES:
        if rx.search(j):
            return n
    return None


for f in sorted(glob.glob(os.path.join(ROOT, "web_*", "extracted.jsonl"))):
    site = os.path.basename(os.path.dirname(f))[4:]
    recs = [json.loads(l) for l in open(f, encoding="utf-8")]
    for r in recs:
        # signature-based duel detection is only reliable on shapxar (Facebook duel transcripts);
        # elsewhere poet-name lines at the end of poems create false 'duels'
        if r.get("duel") and site != "shapxar":
            r.pop("duel", None); r.pop("reply_to", None)
        labels = r.get("labels") or []
        dflt = SITE_DEFAULT.get(site)
        cat_labels = [l for l in labels if not looks_like_name(l)]
        name_labels = [l for l in labels if looks_like_name(l)]
        r["category"] = "; ".join(cat_labels) if cat_labels else (dflt[0] if dflt else (r.get("post_title") or r.get("thread_title")))
        n = norm_from(labels) or norm_from([r.get("post_title"), r.get("thread_title")])
        if site in SITE_FORCE:
            r["category"], n = SITE_FORCE[site][0], SITE_FORCE[site][1]
        if site == "cyprushunting":
            r["category"] = "forum thread: " + (r.get("thread_title") or "")
            n = n or "forum_verse"
        if not n and dflt:
            n = dflt[1]
        if not n:
            if site == "poetryfromcyprus":
                n = "modern_poetry" if name_labels else "prose_other" if any(re.search("ανακοιν|ιστορ|εκδηλ|στατιστ", l, re.I) for l in labels) else "unlabeled"
            else:
                n = "unlabeled"
        r["category_norm"] = n
        r["dialect"] = dialect_of(r.get("text", ""))
        poet = None
        if name_labels:
            poet = re.sub(r"\s+", " ", name_labels[0]).strip()
        elif r.get("author") and r.get("kind") == "couplet" and r.get("source_type") != "comment" and site != "tsiatista_wordpress":
            poet = r["author"]  # signature line in a duel post
        elif r.get("source_type") == "comment" or site == "tsiatista_wordpress":
            poet = r.get("author")  # commenter (often the composer or submitter)
        elif dflt and dflt[2]:
            poet = dflt[2]
        elif site == "shapxar":
            poet = r.get("blog_author")  # blog of Χάρης Σιαπ. (own verses) unless signed otherwise
        r["poet"] = poet
    with open(f, "w", encoding="utf-8") as o:
        for r in recs:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(site, len(recs), dict(Counter(r["category_norm"] for r in recs)), dict(Counter(r["dialect"] for r in recs)))
