# Mougia: a fruit-fly brain that answers in Cypriot τσιαττιστά

> *Εκάτσαμεν να μάθουμεν της μούγιας να τσιαττίζει,*
> *τζι εκείνη πάνω στο φαΐν γυρίζει τζιαι βουρνίζει!*

Mougia trains a **connectome-constrained neural network** ([the wiring of the *Drosophila* brain](#the-fly-connectome)) to do what a Cypriot ποιητάρης does in a τσιαττιστά duel. You give it a rhyming 15-syllable couplet, and it answers with one.

This repository holds:

1. **A dataset**: what we believe is the largest assembled corpus of Cypriot-dialect tsiattista and folk couplets, including real call→reply duels.
2. **A dataset browser**: a small web app for reading the corpus and its battles in the order they were stored.
3. *(next)* **The fly-brain model**: see [PLAN.md](PLAN.md). A medium GPT-2 baseline comes after the fly-brain model works.

## The dataset at a glance

| | Count |
|---|---|
| Unique Cypriot couplets | **6,668** |
| …passing strict 15-syllable meter + rhyme | 3,377 |
| Call → reply pairs from real duels | 179 (64 strict) |
| Battle threads (≥ 2 poets exchanging verse) | 100 |
| Raw records kept with provenance | 51,495 |
| Cypriot dialect text for LM pretraining | ≈ 1.75 M words |

Counts are as of 2026-10-18. Re-run `scripts/build_dataset.py` to refresh them.

### Sources

| Family | Main sources | Notes |
|---|---|---|
| Books | Sakellarios, *Τα Κυπριακά* vol. 2 (1891): 457 numbered δίστιχα plus 78 folk songs; Cypriot Wikisource; Ορθόδοξος Κήρυξ; a CD booklet by Ττερλίκκας; a school textbook; a UoM thesis | Sakellarios was transcribed from the page scans and checked against the book's line numbers |
| Web | Κυπρίων Ποίηση blog, Χάρης Σιαπ. written duels, Cyprus Hunting forum verse threads, stixoi.info Cypriot songs, tsiatista.wordpress, several dialect blogs | Forum and blog threads keep their poet names and reply order |
| Corpora | GRDD / GRDD+ Cypriot sections (Zenodo/GitHub), MilaMou transcripts, the Tsiattista Muse gold set | Couplets were mined from running text with a meter and rhyme detector |
| YouTube | 4,133 videos catalogued (855 of them tsiattista); couplets taken from descriptions and comments | Whisper transcripts of the sung duels are too noisy (CER ≈ 0.35). They are stored, but kept out of the dataset |

Each `data/raw/<source>/SOURCE.md` records the URLs, method, license notes and counts for that source.

### Cypriot only

Every couplet must read as Cypriot dialect. `scripts/dialect.py` scores the evidence for and against:
- **Cypriot markers:** τζ(ι)αι, έσιει, πόθκια, φκαίνω, εννά, εν, ούλλα, κκελλέ-style geminates, final -ν, and -ουσιν endings.
- **Cretan / Standard Greek markers:** τση/τζη, ήντα, εδά, Κρήτη, μαντινάδα, and so on.

Calibration: about **0.6 %** of Cretan mantinades pass the filter, while **89 %** of gold tsiattista are kept. Curated Cypriot folk collections such as Sakellarios are only rejected on explicit non-Cypriot evidence, because their archaic spelling hides modern markers. Non-Cypriot corpora live apart in `data/noncypriot/` and are never merged.

### Files

```
data/
  raw/<source>/            untouched downloads + extracted.jsonl + SOURCE.md   (git-ignored)
  noncypriot/              Cretan / Standard Greek corpora, kept apart          (git-ignored)
  processed/
    couplets.jsonl         one unique Cypriot couplet per line
    pairs.jsonl            call → reply couplet pairs
    records.jsonl          every raw record in stored order + what happened to it (kept / duplicate / filtered + reason)
scripts/
  build_dataset.py         merge → filter → dedupe → pair; writes data/processed/
  greek_meter.py           syllable counting (synizesis, synaloepha), rhyme keys, dedupe keys
  dialect.py               Cypriot-vs-other dialect scoring
  collect/{web,books,corpora,youtube}/   the scrapers and extractors for each source family
app/                       dataset browser (stdlib Python server + vanilla JS)
```

A record in `couplets.jsonl` looks like this:

```json
{"id": "c005112", "lines": ["Επόλλυνε η όρεξη τζιαι σύραμε κουρτέλλαν", "τζιαι μείνανε τα πιάτα μας, όφτζιερα, πασαρέλλα."],
 "syllables": [13, 15], "meter_ok": true, "rhyme_ok": false, "source": "web_shapxar", "author": "Άνδρος",
 "category": "tsiattista", "cy_score": 6.0, "url": "https://shapxar.blogspot.com/2015/05/9.html", ...}
```

## Rebuilding

```bash
python3 -I scripts/build_dataset.py          # re-merges data/raw/* into data/processed/*
```

The scrapers can be re-run on their own. For example, `scripts/collect/web/run_all.sh` rebuilds the web extraction from the cached HTML.

## Dataset browser

```bash
python3 -I app/server.py --host 0.0.0.0 --port 4000
# → http://<this-machine>:4000
```

The browser has no dependencies beyond the Python standard library.
- **Battles:** threads where two or more poets exchange verse. The first poet's turns sit on the left and the second poet's on the right. Each turn shows "answers #n" and the syllable count of every line, and the rhyme word is highlighted when the couplet rhymes.
- **Call → reply pairs:** the training signal.
- **Sources:** every thread or document from every source.
- **Search:** ignores accents, across verses and poet names. Press `/` to focus it.
- **"Kept" vs "Everything stored":** switches between the cleaned dataset and every raw record. Filtered records are labelled with the reason they were dropped.
- **Training:** live convergence for every run, refreshing every 30 seconds.
  - Charts of training and validation loss, perplexity, meter/rhyme/Cypriot rates of the replies, learning rate, gradient norm, throughput, GPU memory, plus any other number the trainer logs.
  - One line per run, with a hover readout and a table of the latest values.
  - **Replies during training:** every 15 minutes of training, each model answers the same held-out couplets. The prompt and the reply are shown side by side with their scores.

### Run logs (for the paper)

Every training run writes to `runs/<run_id>/` through `mougia_train/runlog.py`. The files are append-only and flushed on every write:

| File | Contents |
|---|---|
| `config.json` | full config, seed, git commit, host, library and GPU versions |
| `status.json` | running / finished / failed, current step and training time |
| `metrics.jsonl` | per-step: train loss, LR, grad norm, throughput, GPU memory, … |
| `eval.jsonl` | per-evaluation: val loss/perplexity, meter, rhyme and Cypriot rates of generated replies |
| `samples.jsonl` | prompt + reply couplets generated every 15 min of training time |

Time spent generating samples is excluded from `elapsed_s`, so "training time" means the same thing for every run. Point the browser at another log directory with `MOUGIA_RUNS=/path/to/runs`.

The page follows Apple's Human Interface Guidelines ([apple-design-skill](https://github.com/dickwu/apple-design-skill)):
- system type for the interface, a Greek serif for the verse;
- light and dark appearance that follows the OS;
- 44 px touch targets and text contrast of at least 4.5:1;
- a translucent toolbar, and a copper accent, because Cyprus is named for copper.

The server has **no authentication**. Bound to `0.0.0.0`, it is visible to anyone who can reach the port.

## The fly connectome

There are two fly-brain ποιητάρηδες, one built from each sex's wiring:
- **Kyriakou** is the female, built from FlyWire.
- **Markos** is the male, built from the male CNS.

They train on the same data and then duel each other (names are set in `mougia_train/poets.py`).

| | Publication | Data / repository |
|---|---|---|
| **Male CNS connectome**, brain + nerve cord, ~166k neurons, ~125M synapses. *The map the "fly brain as an LLM" projects use.* | Berg et al., "Sexual dimorphism in the complete *Drosophila* male central nervous system connectome", *Cell* 189(18), 2026. [doi:10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015) | [male-cns.janelia.org](https://male-cns.janelia.org/download/) (CC-BY; neuPrint dataset `male-cns:v1.0`, bucket `gs://flyem-male-cns/v1.0/`), with [neuprint-python](https://github.com/connectome-neuprint/neuprint-python) for access |
| **FlyWire whole-brain connectome**, adult female brain, ~139k neurons, ~50M connections | Dorkenwald et al., "Neuronal wiring diagram of an adult brain", *Nature* 634, 124–138, 2024. [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y) | [Zenodo v783 release](https://doi.org/10.5281/zenodo.10676866), [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations), [codex.flywire.ai](https://codex.flywire.ai) |

**The connectome trained as a language model.** PitchFly was built by Will Knight at WIRED: he vibe-coded headline generation through the open fly connectome. See ["I Trained a Fly's Brain to Generate WIRED Story Ideas"](https://www.wired.com/story/i-trained-a-fly-on-wired-story-ideas/) (WIRED, 2026) and the live site [pitchfly.fun](https://www.pitchfly.fun/). As far as we could find (2026-10-18), its source code is **not public**, so Mougia implements its own connectome-constrained model (see [PLAN.md](PLAN.md)).

## Licensing

The collected texts belong to their authors and publishers. Sakellarios (1891) and Wikisource texts are public domain. Most web and corpus material carries no license and is used here for **non-commercial research** only, so do not redistribute `data/raw/`. The per-source details are in each `SOURCE.md`.

## Status

- [x] Step 1: data collection, Cypriot filtering, dataset browser
- [ ] Step 2: connectome-constrained model. Convergence logs for every run will be kept for the paper.
- [ ] Baseline: GPT-2 medium, trained on the same data
