# Roxy

Classical protein sequence descriptors for machine learning.

Roxy sits between **PRISM** (data curation) and **Sylphy** (sequence encoders and embeddings). It focuses exclusively on
computing classical numerical descriptors from amino-acid sequences.

## Install

```bash
pip install roxy
# or with uv:
uv add roxy
```

## Quickstart

```python
from roxy import read_fasta, DESCRIPTOR_REGISTRY

# Load sequences from FASTA
records = read_fasta("sequences.fasta")  # [(id, seq), ...]
sequences = [seq for _, seq in records]

# Compute a descriptor family (once families are registered)
descriptor = DESCRIPTOR_REGISTRY["aac"]()
features = descriptor.compute(sequences)  # polars.DataFrame
```

## CLI

```bash
roxy --help
roxy list                          # show registered descriptor families
```

## Descriptor Families

| Family          | Descriptors                                                              |
|-----------------|--------------------------------------------------------------------------|
| composition     | AAC, DPC, grouped, k-mers, compositional bias                            |
| physicochemical | global basic, hydrophobicity, charge, structural propensity              |
| autocorrelation | Moran, Geary, normalized Moreau-Broto                                    |
| CTD             | composition/transition/distribution                                      |
| pseudo          | PseAAC, QSO, sequence order                                              |
| complexity      | Shannon entropy, linguistic complexity, local repetition, run blockiness |
| motif           | pattern matching, user regex, spacing, functional residue content        |
| positional      | normalized positional, sliding window, terminal                          |
| aaindex         | AAIndex-based mean properties                                            |

## Related Projects

- [Sylphy](https://github.com/kren-ai-lab/sylphy) — sequence encoders and pretrained model embeddings
