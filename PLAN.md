# Mougia — a fruit-fly-brain Tsiattista bot

Goal: give the model a tsiattisto (a 15-syllable rhyming Cypriot couplet) and have it answer with a couplet the way a ποιητάρης would.

## Step 1: Data (current step)

Tsiattista come from an oral tradition, so no single dataset exists. We collect from several source families, keep the raw downloads with their sources, and then normalize everything.

| Family | Sources | Expected yield |
|---|---|---|
| A. Web | tsiatista.wordpress.com (user-posted couplets in comment threads, which are real call/response exchanges), kypriaki-dialektos.blogspot, cyprusalive, news reports on tsiattista festivals and competitions, forums | thousands of couplets, some as dialogue pairs |
| B. Digitized books | Sakellarios *Τα Κυπριακά* vol. 2 (1891, public domain, archive.org), Anemi (Univ. of Crete), Pergamos/searchculture, Cyprus Pedagogical Institute PDFs (sch.cy), CUT Agapiou archive (helates), openbook.gr | thousands of δίστιχα (OCR needs cleanup) |
| C. NLP corpora | GRDD / GRDD+ Cypriot portion (GitHub/Zenodo), HuggingFace, other Cypriot corpora | about 2M words of dialect text for language pretraining, plus whatever couplets we can mine from it |
| D. Audio/video | YouTube tsiattista duels (festivals, RIK, Cyprus Music Archive): descriptions, captions, and Whisper transcription for a quality test | real duels in true dialogue order, but transcription will be noisy |

Layout:
- `data/raw/<source>/`: untouched downloads and their extracted text, plus `SOURCE.md` (URLs, license, date).
- `data/processed/couplets.jsonl`: one couplet per line: `{id, text, lines[2], source, url, poet?, reply_to?, syllables[2], rhyme_ok}`.
- `data/processed/pairs.jsonl`: (call → reply) couplet pairs, used for training the "reply".
- `data/processed/cypriot_lm.txt`: general Cypriot dialect text for pretraining.

**Cypriot only.** `scripts/dialect.py` scores Cypriot markers (τζαι, σιη, θκ, φκ, εννά, εν, ούλλα, geminates, final -ν, -μεν/-σιν endings, etc.) and penalizes Cretan markers (τση/τζη, ήντα, εδά, Κρήτη, μαντινάδα).
- Calibration: at threshold 1.5, 0.6% of Cretan mantinades pass and 89% of the gold tsiattista pass.
- Curated Cypriot folk collections (Sakellarios and similar) are rejected only on explicit non-Cypriot markers.
- Couplets in Cretan-themed threads need a higher score.
- Non-Cypriot corpora are kept out of the dataset, in `data/noncypriot/`.

Quality filters: Greek script only; dedupe (normalize accents and τσ/τζ spelling); each line about 13–17 syllables after synizesis; the two lines rhyme on their last stressed syllable.

Target: at least 5k unique couplets, at least 500 call/reply pairs, and at least 1M words of dialect text.

## Step 2: The fly brain

The pasted GCN snippet is not a real design. It builds a random graph rather than the fly connectome, and it treats tokens as graph nodes. What we will build instead:

1. **Connectome**: both sexes. The male CNS v1.0 (Berg et al., *Cell* 2026; about 166k neurons; CC-BY; male-cns.janelia.org). The female FlyWire FAFB v783 (Dorkenwald et al., *Nature* 2024; about 139k neurons; Zenodo). Start with a subgraph that fits a 12 GB GPU, for example the central brain or the mushroom body plus the central complex, about 10–30k neurons.
2. **Model**: a *connectome-constrained recurrent network*. The neurons are the hidden units. The recurrent weight matrix is sparse and fixed to the real synapses: the sign comes from the neurotransmitter, the initial magnitude from the synapse count, and only a per-edge or per-cell-type gain is learned. Token embeddings are injected into sensory neurons and read out from descending neurons. We unroll a few "ticks" per syllable.
3. **Tokenization**: a syllable-level tokenizer for Cypriot Greek, with a vocabulary of about 2–4k syllables. This makes meter (15 syllables) and rhyme (final syllables) easy to learn.
4. **Training**: (a) next-syllable language modeling on dialect text, then (b) fine-tuning on couplets, then (c) training on call→reply pairs. At generation time we use constrained decoding that forces 15 syllables per line and rhyme between the lines.
5. **Male vs. female**: train the same architecture on both sexes' wiring.
   - **Female:** FlyWire FAFB v783 (brain only), or BANC (brain plus nerve cord).
   - **Male:** the male CNS v1.0.
   - **Fair comparison:** use the brain-only parts of both. Berg et al. matched cell types across the two sexes (7,205 types are the same in both), so both flies can get identical input (sensory) and output (descending) neuron types. Then only the wiring differs.
   - **The fun part, a "Fly Tsiattista Battle":** the two models duel, each answering the other's last couplet. Each turn is scored on meter, rhyme and Cypriot score, plus human votes in the browser.
6. **Baselines**: the same parameter budget with a random sparse graph of the same density, plus a tiny transformer. This shows whether the fly wiring actually matters.
