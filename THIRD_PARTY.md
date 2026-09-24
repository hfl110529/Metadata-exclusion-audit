# Data and software attribution

| Material | Source and terms |
|---|---|
| SciFact corpus, queries, qrels, transformed title/body inputs and derived retrieval evidence | David Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020, pp. 7534–7550, [DOI](https://doi.org/10.18653/v1/2020.emnlp-main.609). [BEIR data](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip); the upstream dataset card declares [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/). Original citation and license-source records are in `data/scifact/`. |
| RFC text used as background content | RFC 9110 and RFC 9111, copyright 2022 IETF Trust and named authors; [BCP 78 / IETF Trust Legal Provisions](https://trustee.ietf.org/license-info). Original RFC documents with their notices are in `data/precision/`. |
| MiniLM encoder outputs | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; [model repository](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Model weights are not redistributed. |
| BGE-M3 encoder outputs | `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181`; [model repository](https://huggingface.co/BAAI/bge-m3). Model weights are not redistributed. |
| Native retrieval/update software used in the experiments | LlamaIndex core 0.12.27/0.12.28, Haystack 2.31.0, Hayhooks 1.24.0, FAISS 1.15.1, Qdrant Client 1.19.1, Qdrant server 1.15.4, Chroma 1.5.9 and PostgreSQL 16.2. Install official packages under their upstream licenses. This repository contains experiment-specific checks rather than their runtime distributions. |

The DICOM examples and medical themes are synthetic. Body-prefix extraction, synthetic title replacement and metadata interventions are described in [EXPERIMENTS.md](EXPERIMENTS.md). A qrel-positive SciFact target is a cited document; it need not support the query's scientific claim.

Original paper/code licensing is not specified. The third-party notices above retain their own terms and do not grant a new license over the complete repository.
