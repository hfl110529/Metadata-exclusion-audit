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

The commands in [README.md](README.md) replay saved evidence. The procedures below describe collection; fresh service/model execution produces new measurements. Each subsection identifies the supplied inputs and expected result.

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

Evidence: [data/migration](data/migration) (aggregate stage results) and [data/reader_repair](data/reader_repair) (native snapshots and responses).

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

## Payload removal and policy-only refresh

Evidence: [data/backend_updates](data/backend_updates). The [protocol](data/backend_updates/protocol.json) contains the exact 36 document bodies, two protected-note histories, twelve queries and model revisions.

1. Encode the fixed strings with MiniLM and `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`. Use CPU, four threads, float32, batch size four and L2 normalization. MiniLM uses attention-mask mean pooling; BGE-M3 uses the dense CLS representation. Apply no query prefix and reject truncation.
2. For each model, create two persistent worlds in Qdrant Client 1.19.1 local mode and Chroma 1.5.9. Preserve all twelve targets, 24 backgrounds, insertion order, IDs and permitted bodies. Qdrant uses native exhaustive cosine; Chroma uses cosine HNSW with batch/sync thresholds 3, efSearch 128 and one thread.
3. Run the six registered stages: included input; unchanged repeat; audit-tag-only update; delete `protected_note` and mark body-only input policy; replace vectors with body-only embeddings; edit the excluded note again. Read back every vector/payload and request native top-five, top-ten and all 36 candidate scores for every query at each stage.
4. Compare vector bytes before/after metadata deletion, native paired scores, background controls and membership. Reconstruct all full scores from returned vectors. Treat Chroma top-k membership as approximate; use complete native scores for the score-effect reference.

**Expected result:** all twelve effects persist after metadata removal in every configuration. Median absolute differences are 0.026695/0.027475 for MiniLM/BGE-M3. Replacement makes all paired candidate differences zero. The two backends share inputs; these are dependent comparisons. Qdrant local search can renormalize vectors in place, with observed coordinate drift up to 2.98e-8.

For the separate LlamaIndex 0.12.28 diagnostic, use document IDs 1/101/201 and query `q01` from the same fixture. Supply captured MiniLM vectors through an exact-string adapter. Use text template `{content}\n{metadata_str}`, metadata template `{key}: {value}`, chunk size 512 and zero overlap. Run unchanged refresh, LLM-exclusion-only refresh, embedding-exclusion-only refresh, explicit `update_ref_doc`, then unchanged repaired refresh. Record incoming/stored hashes, rendered embedding inputs, embedding calls, native receipts and all three scores. Policy-only refresh returns false and retains the 0.047511 effect; explicit update embeds once per world and makes the difference zero.

## Enforced access: PostgreSQL and Qdrant

Inputs: [reader snapshots](data/reader_repair/snapshots/) and [target identities](data/reader_repair/public_manifest.json). Both deployments reuse the 192-document, 24-query, 48-comparison fixture without new encoding.

### PostgreSQL 16.2

1. Create an isolated local database `auditstudy` with login roles `limited_auditor` and `score_auditor`; use SCRAM authentication. As its trusted initializer, apply [schema.sql](data/access_sql/schema.sql), which creates the nonlogin owner, private tables and functions with pinned search paths.
2. Load four document states into `private.documents`, queries into `private.queries`, query–target mappings into `private.targets`, and trusted state digests into `private.epochs`. Use epoch names `old_w0`, `old_w1`, `new_w0`, `new_w1`. The scorer computes exhaustive double-precision cosine, ordering by descending score and ascending UID.
3. The limited role can call only approved-query top-ten and state-receipt functions. Exercise table/vector reads, ID lookup, COPY, system-file access, role escalation, private/privileged routines, writes, protected-schema creation and unsupported query/epoch requests. Retain the exact SQL and errors. Outer filtering, offsets and batching operate on already limited results.
4. The stronger role has three added function grants: target scores, full scores and vector export. Measure each route sequentially and batched across both worlds/all queries. Also filter full scores to the declared pairs and exported vectors to the target union. Batch SQL, parameters and native results are retained in [responses.jsonl.gz](data/access_sql/responses.jsonl.gz).
5. Use one warmup and three measured rounds per state/route, rotating route order. Count calls, returned scores and UTF-8 compact JSON value bytes. Record median/range elapsed time including collection, decoding and route-specific vector reconstruction; exclude common startup/load/validation. Compare every route to complete native scores and check unchanged loaded state and grants.

**Expected result:** fourteen privilege probes and five unsupported requests fail. Top ten detects all old effects and accepts no repairs. Batched target scores accept 48/48 with 96 scores/8,187 bytes in one call (median 0.017 s); filtered full scoring returns the same bytes but takes 1.918 s; full output returns 9,216 scores/786,073 bytes. These grants express a configured application policy, not PostgreSQL defaults.

### Qdrant server 1.15.4

1. Start an isolated server with an admin API key and a native `read_only_api_key`. Create four cosine collections from the same snapshots; bind native point IDs to synthetic UIDs using [identity.json](data/access_qdrant/identity.json). Freeze state during collection and compare full vector readbacks before/afterward.
2. Use the read-only key for all measured routes: depths 10/20/50/192, exact top ten, ID/payload-filtered target queries, filtered queries with vectors, batched target queries, target fetch and full scroll. Set `exact=true` for exact/target/full validation. These small collections are not an ANN-quality experiment.
3. Attempt eight writes with the read-only key, and reads with missing/invalid keys. Probe the twenty read capabilities and classify REST operations in [API_INVENTORY.csv](data/access_qdrant/API_INVENTORY.csv). Preserve response status and exact body bytes.
4. Run old then corrected states, three repetitions per route in the [registered order](data/access_qdrant/protocol.json), then world/query order. Compare returned scores with exhaustive native scores. Count HTTP calls, body bytes, scores, vector coordinates and client/server elapsed time; report medians and ranges, excluding setup and probes.

**Expected result:** eight writes are denied, twenty read probes succeed, and 31/71 classified REST operations are exercised. Batched target queries use two requests and 4,654 response bytes to accept all 48 repairs (median 0.011 s). Read-only access permits score/vector retrieval; it is not a top-k-only boundary. Corrected depth 20/50 accepts 5/25 comparisons, compared with Haystack's 2/24 on the same vectors.

## Private EHR study

The private EHR dataset was used with authorization from the data custodian. Its directory contains only the [data availability statement](data/ehr/README.md). The study's methods and results are described in Section V-E and Supplement G of the paper; this experiment is excluded from public reproduction.

## Paired-score retention

Evidence: [data/score_retention](data/score_retention); source responses are the [state-bound target observations](data/equal_access/binding/).

1. For each old/corrected state and world, retain two native target scores for each of 24 fixed queries. Bind each response to its query digest, eligible target/native IDs, scorer, state descriptor and epoch under the supplied contract.
2. Serialize compact sorted-key JSON. Chain each entry's sequence, previous hash and payload with SHA256; bind the initial hash to the contract and log ID. Flush and fsync at each stage end; retain independent count/root checkpoints after records 48 and 96.
3. Verify chain, checkpoints, declared coverage and state/query/target binding. Compare payloads with the separately retained source responses. Compute within-stage paired differences; do not substitute old-stage scores for missing new-stage evidence.
4. Replay three local serialization/append/fsync/verification runs. Record log, contract and checkpoint bytes separately. Exercise score edits, dropped/duplicated/reordered entries, wrong query/target/native ID/world/state/epoch/descriptor, and rewritten chains with original or replaced checkpoints.

**Expected result:** one log has 96 records/192 scores/123,341 bytes; contract/checkpoints add 19,302/470 bytes. Median local processing is 10.258 ms, excluding source preparation and acquisition. Paired new logs accept all 48 repairs; old logs plus new top-ten observations accept none. Thirteen negative controls fail. A rewritten chain with replaced checkpoints passes internal integrity but fails source comparison; the mechanism does not authenticate a hostile source.

## Additional analyses of the saved observations

- **Rank margin:** for each of 24 queries, compare all 192 candidate scores across worlds and the baseline gap between ranks ten and eleven. Acceptance requires maximum absolute change <= epsilon and gap > 2 epsilon. Reclassify the five migration/reader stages; these are repeated observations.
- **SciFact repair transitions:** join included/excluded states on engine, depth, query and target. Count visibility (0/1/2 histories) and T/N/I transition matrices for every saved condition. These are separately rebuilt states, not a live repair execution.
- **Recall-to-cutoff bound:** for each response calculate missed true neighbors `X=k(1-R)`. At each fixed k, compare cutoff failure frequency with `min(1, mean(X))`; pooled depths use `mean(K(1-R))`. Retain all twelve efSearch/depth strata. The fixed query grid does not satisfy an iid calibration contract.
- **Double omissions:** select geometry cells whose target is absent in both phases, recount labels and interval-width changes for every method, and retain engine/depth strata. All 932 double omissions remain insufficient, although 492 difference intervals narrow.

These calculations are implemented in [extensions.py](scripts/extensions.py). [precision.py](scripts/precision.py) separately reclassifies every supplied precision view at the six declared tolerances using exact rational endpoints.
