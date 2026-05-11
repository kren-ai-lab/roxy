# Roxy

Classical protein sequence descriptors for machine learning.

Roxy computes classical numerical descriptors from amino-acid sequences and
returns tabular features as `polars.DataFrame` objects.

## Install

```bash
pip install roxylib
# or with uv:
uv add roxylib
```

## Quickstart

```python
from roxy import read_fasta, DESCRIPTOR_REGISTRY

# Load sequences from FASTA
records = read_fasta("sequences.fasta")  # [(id, seq), ...]
sequences = [seq for _, seq in records]
ids = [seq_id for seq_id, _ in records]

descriptor = DESCRIPTOR_REGISTRY["aac"]()
features = descriptor.compute(sequences, ids=ids)  # polars.DataFrame
```

For direct Python use, descriptors are importable by family:

```python
from roxy.descriptors.aaindex import AAIndexDescriptor
from roxy.descriptors.composition import KmerFullAlphabetDescriptor
from roxy.descriptors.physicochemical import ChargeDescriptor

kmer = KmerFullAlphabetDescriptor(k=2)
aaindex = AAIndexDescriptor()
charge = ChargeDescriptor()

kmer_features = kmer.compute(["ACDEFGHIK"])
aaindex_features = aaindex.compute(["ACDEFGHIK"])
charge_features = charge.compute(["ACDEFGHIK"])
```

## CLI

```bash
roxy --help
roxy --version

# List and inspect descriptors
roxy list
roxy list --family composition
roxy describe aac

# Compute descriptors from FASTA, CSV, or Parquet input
roxy compute sequences.fasta -d aac -d charge -o features.csv
roxy compute sequences.fasta -f composition -o composition.parquet
roxy compute sequences.csv -d aac --seq-col sequence --id-col id -o features.csv
roxy compute sequences.fasta --all -o all_features.parquet

# Generate and use a YAML config with descriptor parameters
roxy init-config -d aac -d charge -o roxy.yaml
roxy init-config -f composition -o composition.yaml
roxy init-config --all -o all_descriptors.yaml
roxy compute sequences.fasta --config roxy.yaml -o configured_features.csv
```

`roxy compute` requires exactly one descriptor selector: `--config`, `--all`,
`--descriptor/-d`, or `--family/-f`. CSV and Parquet inputs use `sequence` as
the default sequence column; pass `--seq-col` and `--id-col` when needed.

## Examples

The repository ships a few plain Python scripts under [`examples/`](examples):

- `examples/basic_api.py` for direct descriptor use from Python
- `examples/configured_descriptors.py` for parameterized descriptor instances
- `examples/cli_roundtrip.py` for an end-to-end CLI config and compute flow

Run them with:

```bash
bash examples/run_ci_examples.sh
```

## Descriptor Families

Registry keys are the names used by `roxy compute -d`, config files, and output
column prefixes.

| Family          | Descriptor keys                                                                       |
|-----------------|---------------------------------------------------------------------------------------|
| aaindex         | `aaindex`                                                                             |
| autocorrelation | `autocorrelation`                                                                     |
| complexity      | `entropy_complexity`, `local_repetition`, `run_blockiness`                            |
| composition     | `aac`, `compositional_bias`, `dpc`, `grouped`, `kmer_full_alphabet`, `reduced_kmer`   |
| ctd             | `ctd_classic`, `distribution`                                                         |
| hybrid          | `family_summary`                                                                      |
| motif           | `functional_residue_content`, `pattern`, `spacing`, `user_regex`                      |
| physicochemical | `charge`, `global_basic`, `hydrophobicity`, `order_disorder`, `structural_propensity` |
| positional      | `normalized`, `sliding_window`, `terminal`                                            |
| pseudo          | `pseaac`, `qso`, `sequence_order`                                                     |

## Development

```bash
uv sync --all-extras
uv run pytest -q
uv run task lint
uv run roxy --help
```

## Related Projects

- [Sylphy](https://github.com/kren-ai-lab/sylphy) — sequence encoders and pretrained model embeddings
