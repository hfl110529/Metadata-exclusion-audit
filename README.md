# Metadata Exclusion Audit

Code, data and experimental procedures for **Metadata Exclusion in Retrieval Services: Detection and Repair Evidence**.

[Paper](paper/main.pdf) · [Supplement](paper/supplement.pdf) · [Experimental procedures](EXPERIMENTS.md) · [Results and evidence](RESULTS.md)

The paper studies whether an excluded metadata field still affects a document's retrieval score, and what evidence is sufficient to verify repair. The experiments distinguish input-policy changes, stored-vector replacement, serving-reader state and the visibility of target scores.

## Reproduce the results

Use Python 3.12, matching the recorded environment. To run every saved-data check:

```bash
python -m pip install -r requirements.txt
python reproduce.py --all --output reproduced/results.json
```

Without NumPy, `python reproduce.py` runs the upstream regression, migration, reader repair, equal-access baseline, attribution, precision and geometry checks. Each experiment can also be checked directly, for example `python scripts/reader_repair.py`.

The checks recompute results from recorded responses, per-comparison evidence, saved vectors and geometric certificates. They do not call a model or start a service. [EXPERIMENTS.md](EXPERIMENTS.md) gives the original collection procedures, fixed inputs, software versions and parameters. Input checksums are in `SHA256SUMS`.

## Findings

| Question | Result |
|---|---|
| Can an input-policy change leave stored influence? | The reproduced LlamaIndex update regression leaves a 0.797871 target-score residual; the corrected version has zero residual. |
| Do successful writes establish repair? | A stale reader retains all 48 effects. Correcting the reader makes all 48 complete-score differences zero. |
| Does disappearance from top-k establish repair? | Repaired top-10 evidence leaves all 48 comparisons unresolved. Tested depths 20/50/192 resolve 2/24/48. |
| Which evidence resolves the obligation efficiently? | At 48 calls, target scoring uses 121,694 response-body bytes versus 11,171,866 at full depth: 98.91% fewer bytes. |
| Does the interval rule outperform direct scoring with equal access? | Both give the same verdicts and evidence volume; timing ranges overlap. |
| Does high ANN recall certify omitted scores? | At 99.72% mean recall, 22/256 cutoffs are invalid; no incorrect decisive labels were observed. |
| Do tighter geometric intervals add decisions? | Across 4,880 dependent cells, 998 missing-score intervals narrow, with zero additional decisions. |

The results concern finite score obligations under controlled histories and trusted state binding. They do not establish data deletion or generated-answer privacy. Insufficient and negative results remain in the supplied data.

## Contents

- `data/`: evidence grouped by experiment, including observations, inputs and reported results.
- `scripts/`: readable computations and checks for each experiment.
- `results/`: paper figure data, plots and table exports.
- `paper/`: main paper and supplementary material.

See [THIRD_PARTY.md](THIRD_PARTY.md) for SciFact, RFC and software attribution and data terms. No additional license for the authors' original code or manuscript is declared here.
