# Experimental procedures

These experiments test whether changing one metadata field's history changes the same document's native retrieval score, and whether the available responses suffice to accept repair. The primary tolerance is `epsilon = 1e-6`. Results cover the specified queries, targets and states. See [RESULTS.md](RESULTS.md) for the paper claim–evidence mapping and [README.md](README.md) for saved-data recomputation.

## Shared measurement procedure

1. Fix document identities, corpus membership, permitted content, query text/vectors, encoder, scoring function, eligibility and insertion order. Construct two histories that differ only in the declared metadata intervention.
2. Record the embedding inputs, writes and actual serving reader for each state. Bind public retrieval and target scoring to the same state, query and eligible target identities. The service experiments use trusted local startup and acquisition.
3. Collect ordinary top-k responses. A visible unrounded native score is a singleton interval. A target omitted by verified exact search has the returned cutoff as an upper bound, optionally intersected with the known score range. An ANN cutoff alone is not an omission bound.
4. For phase intervals `[l0,u0]` and `[l1,u1]`, compute the difference interval `[l1-u0,u1-l0]`. Report **supported effect** if it lies strictly outside `[-epsilon,epsilon]`, **no detectable effect** if it lies wholly inside, and **insufficient evidence** otherwise.
5. Compute complete native target scores separately for evaluation. Compare interval containment, decisive verdicts and coverage against the same scorer's complete-score oracle. Report all registered conditions, including insufficient and zero-gain cases.

A score verdict concerns the declared finite comparisons. It does not establish deletion, nonprocessing or generated-answer privacy. Shared queries, states and depths are reused measurements rather than independent observations.

## Model and software

The principal encoder is `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with normalized float32 embeddings. The recorded native environment uses Python 3.12.3, Haystack 2.31.0, NumPy 2.5.3 and, for approximate/exact vector-index experiments, `faiss-cpu` 1.15.1. Encoding provenance records Sentence Transformers 6.0.1, Transformers 5.15.1 and PyTorch 2.13.0. The service ingestion experiments additionally use Hayhooks 1.24.0. The upstream regression compares `llama-index-core` 0.12.27 and 0.12.28 with identical other dependencies.

Saved-evidence verification reconstructs scores where vectors are supplied and reclassifies the recorded score intervals and summaries. The attribution and precision checks start from the supplied per-comparison records; they do not recreate censored or rounded intervals from raw HTTP responses. It does not rerun encoding or contact the original services. The optional native LlamaIndex reproduction is described below. A fresh collection requires the native dependencies, pinned model, fixed input selection and the collection procedures below; it produces a separate measurement, not a replacement for the supplied results.

## Metadata-update regression in LlamaIndex

Evidence: [data/upstream_update](data/upstream_update).

**Purpose:** determine whether a metadata-only edit reaches stored representations through native update detection.

1. In each official release, reproduce a minimal metadata update with `Document(id_="0", text="Hello", metadata={"A":1})`, then add `B=2`. Use `IngestionPipeline`, `SimpleDocumentStore` and `transformations=[]` to isolate update handling.
2. In a separate two-document, one-query fixture, use `SimpleDocumentStore`, `SimpleVectorStore` and explicit `UPSERTS`. Encode the four predefined strings once with pinned MiniLM on CPU. An exact-string `BaseEmbedding` adapter supplies those saved vectors; native hashing, update decisions, vector storage and cosine retrieval remain unchanged.
3. Remove only the target's `Title`, retaining its ID and body. Submit a freshly constructed edited document, then submit that same edited document again as an unchanged-update control. Independently build a fresh index containing the edited corpus.
4. Compare processed-node counts, metadata, hashes, stored vectors and full-depth target scores with the clean rebuild. Independently reconstruct all 16 recorded cosine outputs.

The native regression can also be run in separate environments containing the corresponding official version (0.12.27 or 0.12.28):

```bash
python scripts/run_upstream_update.py --embeddings data/upstream_update/embeddings.json --output reproduced/llamaindex-0.12.27
```

Use a different output directory for the fixed version. This runs native update/store/query operations with the supplied MiniLM vectors; it does not perform fresh model encoding.

**Expected result:** 0.12.27 skips the edit and retains target score 0.910012, compared with 0.112141 after a clean rebuild (residual 0.797871). Version 0.12.28 processes the edit and has zero residual against rebuilding. This tests handling of a subsequent edit; upgrading a package alone is not the repair intervention.

## Metadata exclusion, replacement and serving state

Evidence: [data/migration](data/migration) and [data/reader_repair](data/reader_repair).

**Purpose:** distinguish a changed input policy, replacement writes and the representations used to answer retrieval requests.

**Data:** two worlds of 192 synthetic DICOM records each: 24 targets, 24 unchanged-null records and 144 near-topic backgrounds. Twenty-four queries cover twelve fictional medical themes. Each query addresses two targets and two null records, yielding 48 target and 48 null comparisons per stage. Pair records by `SOPInstanceUID`; preserve permitted text, UIDs, pixels, file metadata, membership and order. Only target `PatientComments` differ between worlds. Source URLs and Haystack IDs are excluded from embedding inputs.

1. Upload the synthetic records to Orthanc. Fetch simplified-tags JSON through Haystack, convert `TextValue` to document content, embed and retrieve using exact cosine.
2. Ingest with comments included. Then exclude `PatientComments` from embedding while deliberately using duplicate policy `SKIP`. Finally use `OVERWRITE` to replace all 384 vectors. Capture embedding inputs, write receipts, ordinary top-ten responses and private complete-192 scores at each stage.
3. Restore old and replacement vectors into separate native stores. Hold successful replacement writes and record inventories fixed while pointing the reader to the old store; then correct only the reader binding.
4. On the repaired state, compare retrieval depths 10, 20, 50 and 192 with retrieval restricted to the two declared eligible target IDs per query. Each channel uses 48 calls. Record resolved comparisons, acquired scores and response-body bytes; keep the later score-only projection separate from bytes actually acquired.
5. For the state-binding measurement, expose ordinary and target routes through the same captured retriever/store handle in a separate process for each world and generation. Freeze queries and check descriptors, epoch, query identity/vector digest and target eligibility around requests. Collect all 192 requests across the four states and two channels.

**Expected result:** `SKIP` leaves all 48 target effects (absolute differences 0.336–0.718); replacement removes every complete-score effect. With the reader still using the old store, all 48 effects remain detectable despite successful writes. Correcting the reader makes all 48 differences exactly zero, but ordinary top ten leaves all 48 insufficient. Depths 10/20/50/192 resolve 0/2/24/48 repaired comparisons. Full depth acquires 9,216 scores and 11,171,866 response-body bytes; the target route acquires 96 scores and 121,694 bytes, a 98.91% reduction at the same call budget. These native bodies contain text and metadata; headers are excluded.

## Equal-access direct-rescoring baseline

Evidence: [data/equal_access](data/equal_access).

**Purpose:** compare decision rules when both receive the same sufficient evidence.

Use the four unchanged 192-document stores (two worlds × old/corrected generation). Both methods inspect the actual reader and descriptor and request the same native target scores for the same 24 queries. One directly tests `abs(s1-s0) > epsilon`; the other applies the interval rule to singleton scores.

For each reader state, run one warmup batch per method, then six measured batches per method, alternating AB/BA order three times. This yields four warmup and 24 measured batches; each batch contains 48 calls, 96 scores and 15,336 request-body bytes. Measure batch elapsed time on one native-scoring thread. Exclude shared startup, preparation, output writing and history acquisition.

**Expected result:** both methods detect all 48 old-state effects and accept all 48 corrected-state score comparisons. Their response volumes are identical: 34,020 bytes for old and 34,034 for corrected states, including binding envelopes. Median direct/interval times are 5.469/5.471 seconds before correction and 5.365/5.378 seconds afterward; ranges overlap. These local timings do not establish a production speed advantage. The format differs from the full-document responses in the acquisition experiment.

## Target-score attribution and numerical precision

### Attribution

Evidence: [data/attribution](data/attribution).

Create eight fresh 608-document worlds across two field policies, two insertion orders and two phases. Each contains 440 public nonclinical RFC 9110 paragraphs, 72 synthetic maintenance backgrounds and 96 synthetic JSON probes. Comments are padded to 192 ASCII bytes. The two insertion orders use a seeded shuffle (seed 101018) and its reversal. Twelve themes supply 36 queries and four paired cohorts per theme. The four cohorts are: swapping target/reference comments; an unchanged-null pair; changing only the reference comment; and unchanged comments with unequal target/reference bodies. Body inequality in the last cohort is between documents, not a change across phases. Both policies parse, store and return `PatientComments`; only embedding inclusion changes. Freeze identities, permitted bodies, queries and membership across paired phases.

Collect exact top-ten and top-fifty responses and apply six rules to the same data: rank change, reference-pair score-gap change, nonzero target-score change, guarded point comparison, target-score intervals, and intervals with an unchanged-null gate. Assign an absent rank `k+1`, abstain if both ranks are absent, and require no detectable effect on the unchanged control for the null gate. The original collection independently reconstructed the 17,280 public scores after classification. The distributed checker reclassifies the saved per-comparison scores, ranks and intervals.

**Expected result:** at top ten, intervals support all 72 true effects, compared with 45 for visible-score point comparison, with no false supports among 504 score nulls. Rank and pair-gap rules falsely support 107 and 25 nulls, respectively. Strict-null gating reduces effect detection to 43 and increases insufficiency from 185 to 301. At top fifty, target-score intervals resolve all 576 episodes.

### New queries, depth and rounding

Evidence: [data/precision](data/precision).

Use sixteen queries from eight equipment themes, disjoint from the attribution/development queries. Apply six comment edits: unchanged, case-only, date, scheduling, frequency and adverb. Cross two field policies and two phases with nested corpus sizes 128/512/1,024 and depths 5/10/50. Keep 80 synthetic topical records (32 neighbors and 48 edit targets) fixed while adding 48/432/944 RFC backgrounds. Use normalized unique 25–100-word paragraphs from RFC 9110 and RFC 9111, interleaved in source order and checked against the 256-token encoder limit. The largest corpus shares 440 of its 944 background paragraphs with the attribution workload; query disjointness is not corpus independence. The resulting 1,728 paired episodes contain 576 effects and 1,152 score nulls.

Compare full-precision responses with predetermined HALF_EVEN rounding to 6/4/3/2 decimal places. Use exact rational half-step intervals around each rounded score and the original unrounded native-score oracle. These precision views reuse saved responses without additional service calls.

**Expected result:** full-precision intervals yield 531 supported effects, 999 no-effect verdicts and 198 insufficiencies. Six decimals preserve these counts; four decimals yield 501/0/1,227. Treating rounded scores as exact creates 18/45/354 false no-effect verdicts at 4/3/2 decimals. At top five, intervals add 15 supports per corpus size; top ten and fifty resolve all episodes at full precision.

For a separate near-threshold stress test, use one saved target/query pair per query at corpus size 512. Interpolate float32 target vectors toward score changes of `0`, `±0.25`, `±0.5`, `±1`, `±2`, and `±4` times epsilon; retain achieved changes and classify by actual native scores. The 192 native calls yield 528 depth projections: 261 effects and 267 nulls. Full-precision counts are 218/221/89; six decimals give 80/139/309. These engineered changes test numerical sensitivity rather than natural metadata effects. A separate saved-data sensitivity analysis reclassifies all cases at tolerances 0, 1e-8, 1e-7, 1e-6, 1e-5 and 1e-4.

## SciFact exact and approximate retrieval

### Fixed public workload and exact scorers

Evidence: [data/scifact](data/scifact).

Select the first sixteen numeric BEIR SciFact test-query IDs with positive qrels and up to their first two numeric positive documents: eighteen query–target pairs and fifteen distinct targets. Retain these targets and the earliest remaining document IDs for nested corpora of 128/512/5,183 documents. Use prefixes of 160 body and 48 title WordPiece tokens. Change only target titles from `Administrative record.` to their original titles; evaluate title-included and title-excluded policies at depths 5/10/20/50.

Run Haystack cosine and separately normalized float32 FAISS `IndexFlatIP`. Use each scorer's own complete native scores as its oracle. Check all 1,536 public responses, 384 complete responses and 864 paired cells.

**Expected result:** for each engine, intervals improve determinate coverage from 385/432 to 389/432, with four added effect supports. Nine of twelve corpus-size/depth conditions show no gain. The two engines share inputs; their matching gains are not independent replications.

### HNSW cutoff validity

Evidence: [data/ann](data/ann).

Use the full 5,183-document workload and saved normalized vectors. Build four `IndexHNSWFlat` graphs with `M=16`, `efConstruction=100`, seed 20260918, numeric-ID insertion and one thread. Cross `efSearch=8/32/128` with depths 5/10/20/50 for all sixteen queries and four policy/phase states, producing 768 responses and 432 paired cells.

Enumerate every native score for each query as privileged evaluation evidence. Measure recall and whether the largest omitted score exceeds the returned cutoff by more than epsilon. Compare the ordinary unknown-omission interval, the deliberately naive ANN-cutoff substitution, scan-derived valid omission bounds and direct target scores.

**Expected result:** increasing efSearch raises mean recall from 82.83% to 99.72%, while cutoff failures decrease from 163/256 to 22/256 responses. Naive cutoff substitution causes twelve target-interval containment failures, although no decisive label is wrong in this fixed grid. Scan-derived omission bounds resolve nine additional cells; direct target scoring resolves all 216 effects and 216 nulls. The scan certifies the measured states, not future HNSW omissions.

## Cross-query geometry

Evidence: [data/geometry](data/geometry).

**Purpose:** test whether narrower missing-score intervals produce additional audit decisions.

1. Use all 284 remaining positive SciFact test queries after the initial sixteen, their first two positive documents, and all 5,183 corpus documents: 305 pairs and 254 targets. Preserve numeric-ID ordering and the same text-prefix construction. Encode 284 query inputs and both 254-target title phases on CPU, one thread, batch size 32. All 792 fresh inputs fit the encoder limit (maximum 209 tokens). Reuse unchanged document vectors and the common excluded-policy vectors.
2. Build FlatIP and HNSW indexes for four policy/phase states. Use HNSW `M=16`, `efConstruction=100`, `efSearch=32`, seed 20260919. Collect all queries at depths 5/10/20/50: 9,088 responses and 4,880 paired cells.
3. Give every method the same query coordinates and response packet. Select anchors only when another query visibly returns the target under the same engine, policy, phase and depth. Keep complete target scores out of anchor selection and interval construction.
4. Compare A (per-query bounds), B (single-anchor geometry), P (multi-anchor projection) and C (verified interval certificates). For C, use zero/projection initializations and bounded L-BFGS-B with at most 200 iterations, retaining the baseline candidates.
5. Treat post-FAISS float32 coordinates as exact stored numbers. Apply the fixed norm bound `R=1.0001`, native-to-real-dot allowance `eta=1e-4` and exact rational tolerance `1/1,000,000`. Enlarge anchor constraints and inferred native scores by the stated error allowances; only FlatIP omissions use cutoffs. Independently verify certificate support sums and residual squared norms with exact arithmetic and outward rounding. Invalid proposals retain baseline bounds and cannot earn gains.
6. Evaluate against separate complete native scores: 2,272 scans covering 11,775,776 scores. Report every engine/depth stratum, interval containment, additional decisions and interval-width changes.

**Expected result:** all four methods yield 1,957 supported effects, 1,943 no-effect verdicts and 980 insufficiencies. Of 1,957 omitted phase scores, 998 have anchors and narrower bounds, but none changes a verdict. All 980 unresolved difference intervals contain zero; their median width is 2.91579. Independent checking of 5,988 certificates finds no containment or decisive-label errors. This negative result distinguishes interval tightening from useful decision gain. The numerical contract and disclosed query coordinates are additional assumptions/access beyond an ordinary score-only API.
