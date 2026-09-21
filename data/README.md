# Experimental data

| Directory | What it contains | Unit of analysis |
|---|---|---|
| `upstream_update/` | Two-document fixture, four saved MiniLM embeddings, native document/vector-store states and outputs for two versions | Document score and processed update |
| `migration/` | Three-stage write/score results and paired identities | Target/query pair within a migration stage |
| `reader_repair/` | Native vector snapshots, query/identity maps, raw response strings, write evidence and channel results | 48 target/query pairs per channel, with unchanged controls |
| `equal_access/` | Matched baseline batches, native snapshots and binding descriptors/observations | 48 calls per batch; four warmup and 24 measured batches |
| `attribution/` | 1,152 reconstructed comparison records with target/reference scores, ranks, visibility, intervals and truth | One query/cohort/policy/order/depth comparison |
| `precision/` | All natural/engineered interval views, reported counts, scientific protocol and RFC attribution texts | 1,728 natural episodes or 528 engineered depth projections, reused across precision views |
| `scifact/` | Original BEIR archive, ID-based selection, composed inputs, captured vectors, complete native score matrices, public responses and decisions | 18 query/target pairs across policy, corpus size, depth and scorer |
| `ann/` | Native full-score arrays, HNSW response rows, scan/target evidence, fixed ID mapping and decisions | 768 responses and 432 paired cells |
| `geometry/` | Query/response packet, interval certificates, analysis, protocol and separate target oracle | 305 pairs across two engines, two policies and four depths; 4,880 cells |

Native measurements and recorded response bodies are retained. Reader/baseline files omit redundant parsed copies and local acquisition-path metadata; checks validate retained raw body lengths/hashes and their contents. Experimental IDs remain stable to support joins.

The scientific protocols retain the controlled intervention, fixed grid, model/software versions and numerical assumptions. They describe the original measurement procedure. The supplied computations operate on saved evidence; see [EXPERIMENTS.md](../EXPERIMENTS.md) for collection steps and [RESULTS.md](../RESULTS.md) for the claim mapping.
