"""Write SOURCE.md for every data/old/raw/web_<site>/ directory (counts computed from extracted.jsonl)."""
import os, json, glob, sys, datetime
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT

TODAY = datetime.date.today().isoformat()
COMMON_METHOD = """Text extraction: HTML stripped (br/p -> line breaks), Greek-script verse-like lines kept, sung refrains
(repeats of the previous line's end) dropped, then lines are *reflowed* into 15-syllable couplets with a dynamic
program over words (scripts/collect/web/common.py: reflow_couplets) that prefers 12-19-syllable verses that rhyme
(tail from the last stressed vowel) and verse ends on original line breaks. So `text` = the two reflowed verses, while
`raw_lines` = the original lines as written (spelling untouched). `rhyme_ok` = heuristic rhyme check;
`dialect` / `category_norm` / `poet` = heuristics added by tag_records.py. Non-rhyming pairs are kept (rhyme_ok=false)
and are often prose fragments -> filter downstream."""

META = {
    "tsiatista_wordpress": ("https://tsiatista.wordpress.com/ (all posts, pages and 56 comments)",
        "WordPress.com public REST API v1.1 (/sites/tsiatista.wordpress.com/comments, /posts). Comments are user-submitted couplets on the page 'Προσθέστε στη λίστα' (2009-2016). No threaded replies exist (parent=null for all) - comments are independent submissions, ordered by date (`position`).",
        "User-submitted folk couplets; no explicit license. One comment is labeled 'Κρήτης' (Cretan variant)."),
    "kupriakatsiatismata": ("https://kupriakatsiatismata.blogspot.com/ (all 98 posts)",
        "Blogger JSON feed /feeds/posts/default?alt=json (paged). Posts mix tsiattista, poiitarika by named folk poets (e.g. Χαμπής Αχνιώτης), folk songs, a dialect glossary and recipes.",
        "Blog content, no explicit license."),
    "shapxar": ("https://shapxar.blogspot.com/ ('ΑΠΑΘΕΙΑ ΤΕΛΟΣ', Χάρης Σιαπ., all 147 posts + 165 comments)",
        "Blogger JSON feeds. Many posts ('ΤΑ ΤΣΙΑΤΤΙΣΤΑ...ΤΗΣ ΚΟΝΤΡΑΣ No N', 'ΔΙΣΤΙΧΑ...του facebook') are transcripts of Facebook tsiattista duels: each turn is followed by a signature line (e.g. 'Χάρης Σιαπ.', 'Θεόδωρος Θ.'). Signatures are detected and attached as `author`; in posts with >=2 distinct signers `duel`=true and the first couplet of each turn has `reply_to` = id of the first couplet of the previous turn. Wrapped Facebook lines (8+7 syllables) are reflowed.",
        "Personal blog; verses by the blogger and his friends. No explicit license. Some Cretan mantinades appear in exchanges with Cretan friends (Chania school twinning)."),
    "milomena": ("http://milomena.blogspot.com/ (12 posts)", "Blogger JSON feed.", "Bawdy ('μυλλωμένα') Cypriot verses and jokes. No license."),
    "poetryfromcyprus": ("https://poetryfromcyprus.blogspot.com/ ('Κυπρίων Ποίηση', all 4892 posts + 152 comments)",
        "Blogger JSON feeds, mode=both: one `kind=poem` record per post (verse-like lines) plus reflowed `kind=couplet` records. Labels are mostly poet names ('Surname Name (Ποιητάρης)') -> `poet`; label/role -> `category_norm` (poiitarika for Ποιητάρης/Λαϊκός ποιητής labels, folk_song for Δημοτική Ποίηση/Κάλαντα/Ανωνύμων, modern_poetry otherwise). MOST of this blog is modern Standard-Greek free verse: use `dialect=='cypriot'` and `category_norm in (poiitarika, tsiattista, folk_song)` to select.",
        "Anthology blog of Cypriot poets; many poems are copyrighted works of living poets -> research use only, do not redistribute."),
    "cyprushunting": ("https://www.cyprushuntingmagazine.com/forum/ (phpBB 'Hunt & Shoot' forum). FULL CRAWL (scripts/collect/web/phpbb_crawl.py, state in crawl_state.json, log crawl.log) of sections 4,5,45,33,3,32,42,53,52,8 and subforums 43 (καφενείο),44 (ονομαστικές εορτές),40,49,13,14,16,17,15,18,19,41,50,38,30, plus most of 31 (marketplace; budget stop). NOT crawled: rest of 31, 39,35,34,36,55,56,48 (dog subforums), 12,20,37 (guns/ammo subforums) - low verse density expected. Raw: html/topics/t<tid>_s<start>.html.gz = topic title + original <div class=post> elements only (site chrome trimmed), html/listing/*.gz; older full pages from the first pass also in html/*.html.gz.",
        "Only posts containing >=1 rhyming reflowed couplet are kept (all units of such posts written, rhyme_ok per unit). Fields: post_id, post_index (order in thread), author (username), date, reply_to_user (user quoted in blockquote; quote text removed), reply_to_post_id (if the quote links a post), prev_post_id / prev_post_author / prev_post_has_verse (immediately preceding post in the thread - for call->reply pairing), forum_id, thread_title. Earlier pass (phpbb.py) records kept only for topics not covered by the crawl. Each post -> couplets with `author` (username), `date`, `post_index` (thread order), `reply_to_user` (user quoted in a blockquote, quotes removed from text). Hunters exchange verses in reply to each other, so thread order is meaningful dialogue.",
        "Forum user posts; no explicit license. Some posts all-caps."),
    "cyprusalive": ("https://www.cyprusalive.com/el/erwtika-kypriaka-distixa", "Single article page.", "Quotes couplets from Κ. Μόντης-Α. Χριστοφίδης 'Κυπριακή Ανθολογία' (1965) and Κ.Π. Χατζηιωάννου 'Κυπριακά διαλεκτικά κείμενα' (1961)."),
    "aftodioikisi": ("aftodioikisi.com.cy article 'Τσιαττίσματα: Πειράγματα τζι ανθίσματα'", "Single article page.", "Book review with a few quoted quatrains (Λάζαρος Λαζάρου)."),
    "laografikosomilosktima": ("https://www.laografikosomilosktima.com/laografia/tsiattista", "Single page.", "Folklore society page; a few examples."),
    "kyriakosorokkou": ("https://kyriakosorokkou.wordpress.com/2017/06/30/ (review of 'Τσια-tease-ζω')", "Single page.", "Quoted quatrains by Σημέλα Βωνιάτη (copyrighted book)."),
    "simerini": ("https://simerini.sigmalive.com/article/2017/1/14/tsiattizei-kai-to-goustarei/", "Single page.", "Interview with Σημέλα Βωνιάτη with quoted quatrains."),
    "authormanolis": ("authormanolis.wordpress.com 2016/01/04 'Ερωτικά Κυπριακά τραγούδια'", "Single page.", "Rimes d'amour (16th c. Cypriot) excerpts + folk couplets."),
    "kypriaki_dialektos": ("https://kypriaki-dialektos.blogspot.com/2012/10/blog-post_335.html", "Single page (.post-body).", "Few examples incl. Γεώργιος Πανατσιάς Κογκονιάς."),
    "kypriaki_dialektos_blog": ("https://kypriaki-dialektos.blogspot.com/ (all 58 posts)", "Blogger JSON feed (comments feed is 403). Folk-song posts -> kind=poem; tsiattista post -> couplets.", "Blog, no license."),
    "oldboy1983": ("http://oldboy1983.blogspot.com/2011/09/blog-post_08.html", "Single page.", "Traditional (bawdy) tsiattista."),
    "adipsona": ("http://adipsona.blogspot.com/p/blog-page_7619.html", "Single page.", "Tsiattista written by primary-school pupils (Ε' τάξη)."),
    "aneforiwn": ("http://aneforiwn.blogspot.com/2009/07/352009-9.html", "Single page.", "Bawdy poiitarika poem 'Του βίλλου με τον πούττον' etc."),
    "vrysoulesgnosis": ("vrysoulesgnosis.wordpress.com 2016/06/12 'ΤΡΙΑ ΠΟΙΗΜΑΤΑ με Κυπριακή διάλεκτο'", "Single page.", "Three dialect poems."),
    "kanellatou": ("https://kanellatou.gr/el/paradosiako/cyprus/* (13 Cypriot traditional song lyrics)", "Single pages.", "Traditional songs (folk_song)."),
    "stixoi": ("https://stixoi.info/paradosiaka/region/10 (62 Cypriot traditional songs, /songs/<id>)", "Single song pages (article text).", "Traditional song lyrics (folk_song); some may be Standard-Greek renderings."),
    "terrabook": ("https://cyprus.terrabook.com/el/cyprus/page/tsiattista/", "Single page.", "Few examples."),
    "alphanews": ("alphanews.live WP REST search (τσιαττιστά etc.)", "WP REST /wp-json/wp/v2/posts?search=; 19 matching news posts.", "News rarely quote verses; ~0 useful."),
    "rembetiko": ("https://rembetiko.gr/t/10709 (Discourse thread 'Τα τσιαττιστά της Κύπρου στην UNESCO')", "Discourse JSON API.", "Discussion only; no real couplets (junk records)."),
}
FAILED = {
    "musicheaven": "musicheaven.gr forum 'Κυπριακά τσιατιστά' (topic 1835): Cloudflare challenge (403), Wayback 429. NOT scraped.",
    "tovivlio": "tovivlio.net 'Ας μιλήσουμε για τσιαττιστά': 403. NOT scraped.",
    "satsupreme": "satsupreme.com 'Κυπριακό Καφενείο' thread: 403. NOT scraped.",
}

for d in sorted(glob.glob(os.path.join(ROOT, "web_*"))):
    site = os.path.basename(d)[4:]
    f = os.path.join(d, "extracted.jsonl")
    recs = [json.loads(l) for l in open(f, encoding="utf-8")] if os.path.exists(f) else []
    kinds = Counter(r.get("kind", "couplet") for r in recs)
    cps = [r for r in recs if r.get("kind", "couplet") == "couplet"]
    good = [r for r in cps if r.get("rhyme_ok")]
    cy = [r for r in good if r.get("dialect") == "cypriot"]
    reply = sum(1 for r in recs if r.get("reply_to") or r.get("reply_to_user"))
    threaded = sum(1 for r in cps if r.get("author") and (r.get("post_index") is not None or r.get("duel") or r.get("comment_id")))
    nfiles = len(os.listdir(os.path.join(d, "html"))) if os.path.isdir(os.path.join(d, "html")) else 0
    m = META.get(site)
    lines = [f"# web_{site}", ""]
    if m:
        lines += [f"**Source:** {m[0]}", "", f"**How scraped:** {m[1]}", "", f"**License / notes:** {m[2]}", ""]
    elif site in FAILED:
        lines += [f"**Status:** {FAILED[site]}", ""]
    else:
        lines += ["**Source:** survey page(s) only (see html/); no dedicated extractor (little or no verse).", ""]
    lines += [f"Scraped: {TODAY} by scripts/collect/web/ (User-Agent 'mougia-research/0.1', >=1s between requests per host).", "",
              "## Counts", "",
              f"- raw files in html/: {nfiles}",
              f"- records: {len(recs)} {dict(kinds)}",
              f"- couplet records: {len(cps)}; rhyme_ok: {len(good)}; rhyme_ok & dialect=cypriot: {len(cy)}",
              f"- records with reply link (reply_to / reply_to_user): {reply}; couplets with author+thread order: {threaded}",
              f"- category_norm: {dict(Counter(r.get('category_norm') for r in recs))}",
              f"- dialect: {dict(Counter(r.get('dialect') for r in recs))}", "",
              "## Method", "", COMMON_METHOD, ""]
    open(os.path.join(d, "SOURCE.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(site, len(recs), len(good), len(cy))
