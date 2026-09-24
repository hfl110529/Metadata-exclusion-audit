# Metadata Exclusion Audit

Supporting code, results and experimental procedures for **Metadata Exclusion in Retrieval Services: Detection and Repair Evidence**.

[Paper](paper/main.pdf) · [Supplement](paper/supplement.pdf) · [Results](RESULTS.md) · [Experimental procedures](EXPERIMENTS.md)

The paper studies whether excluded metadata still affects native retrieval scores, and what evidence is sufficient to accept a repair. Experiments cover update detection, vector replacement, serving-reader selection, access permissions, numerical precision and approximate retrieval.

## Reproduce the supplied results

Use Python 3.12:

```bash
python -m pip install -r requirements.txt
python reproduce.py --all --output reproduced/results.json
```

This recomputes results from saved observations, response bodies, vectors and interval certificates. It does not launch services or encode documents. Without NumPy, run `python reproduce.py`; this omits the SciFact vector and ANN full-score checks. Individual checks also run directly, for example `python scripts/reader_repair.py`.

[EXPERIMENTS.md](EXPERIMENTS.md) specifies the collection steps, inputs, software versions, parameters and expected outcomes. [RESULTS.md](RESULTS.md) links each claim to its evidence and calculation. File checksums are in `SHA256SUMS`.

## Repository contents

| Location | Contents |
|---|---|
| [data/](data/) | Experimental inputs, observations and result aggregates, grouped by study |
| [scripts/](scripts/) | Result calculations, native-score reconstruction and synthetic examples |
| [results/](results/) | Numerical tables, figure data and plots |
| [paper/](paper/) | Main article and supplementary material |

The private EHR dataset was used with authorization from the data custodian and is not publicly distributed. Its directory contains only a [data availability statement](data/ehr/README.md); public reproduction covers the other experiments.

The studies concern finite score comparisons under controlled histories and trusted acquisition. They establish neither deployment-wide failure rates nor deletion or generated-answer privacy. See [THIRD_PARTY.md](THIRD_PARTY.md) for data terms and attribution.
