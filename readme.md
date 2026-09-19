# PropMol

PropMol is an **attribute-guided molecule generation project**. It uses a pretrained property–molecule joint model to generate new molecules from input SMILES based on user-specified "shielded properties", and evaluates the generated results across 56 molecular properties.

## References

If PropMol is helpful to your research, please cite the appropriate papers：`Broadening molecular design chemical space by attenuating human historical design bias and preferences in generative AI`

## Features

- **Attribute-guided generation**: Specify the molecular properties to be shielded/optimized via a config file, and the model generates new molecules accordingly.
- **PV→SMILES decoding**: A stochastic beam-search decoding strategy.
- **56-property evaluation**: Automatically computes the normalized root-mean-square error (nRMSE) of generated molecules across 56 molecular properties.
- **Visualized output**: Draws a structure image (PNG) for each valid generated molecule and exports a complete property report as JSON.

## Directory Structure

```
PropMol/
├── generate_shielded.py        # Main entry script
├── PropMol_models.py           # PropMol model definition
├── dataset.py                  # SMILES dataset loader
├── calc_property.py            # 56-property calculator (with shield_list)
├── d_pv2smiles_single.py       # Single-sample PV→SMILES decoding
├── xbert.py                    # Custom BERT implementation
├── spacial_score.py            # Spatial score (SPS) calculation
├── checkpoint_final.ckpt       # Pretrained model weights
├── vocab_bpe_300.txt           # BPE vocabulary (300 tokens)
├── config_bert.json            # Text BERT config
├── config_bert_property.json   # Property BERT config
├── property_name.txt           # List of 56 property names
├── train_all/
│   └── normalize_1.pkl         # Property normalization params (mean/std)
├── scheduler/                  # Learning rate schedulers
│   ├── __init__.py
│   ├── cosine_lr.py
│   ├── step_lr.py
│   ├── tanh_lr.py
│   ├── plateau_lr.py
│   └── scheduler_factory.py
├── pysmilesutils/              # SMILES data augmentation toolkit
│   ├── __init__.py
│   ├── augment.py
│   ├── tokenize.py
│   ├── datautils.py
│   └── analyze.py
└── out/
    ├── config.json             # Run configuration
    └── exa.txt                 # Input SMILES file
```

## Installation

```bash
conda create -n PropMol python=3.9
conda activate PropMol
pip install -r requirements.txt
```

> Note: `rdkit` must include the `Contrib.SA_Score` module (used for SA Score calculation). The standard conda-installed `rdkit` typically ships with it.

### Run

```bash
python generate_shielded.py <output_path> <config_json>
```

**Example:**

```bash
python generate_shielded.py ./out ./out/config.json
```

After running, the `out/` directory will contain:
- `mol_0.png`, `mol_1.png`, ... — structure images of each valid generated molecule
- `smiles.json` — the full generation result and evaluation report

## Configuration

Edit [out/config.json](PropMol/out/config.json) to set the run parameters:

```json
{
    "input_file": "F:/save/PropMol/out/exa.txt",
    "k": 2,
    "n_generate": 10,
    "properties": ["FractionCSP3"]
}
```

| Field | Type | Description |
|---|---|---|
| `input_file` | string | Path to the input SMILES file (one SMILES per line) |
| `k` | int | Beam search width |
| `n_generate` | int | Number of molecules to generate per input sample |
| `properties` | string[] | List of property names to shield (must be from `property_name.txt`) |

> **Meaning of shielded properties**: Properties listed in `properties` are "shielded" during generation — the model does not use them directly as conditions during the generation process. The report provides nRMSE separately for shielded and non-shielded properties, which can be used to evaluate whether the model genuinely "bypasses" the shielded attributes.

## Supported Properties (56)

Available `properties` names are listed in [property_name.txt](PropMol/property_name.txt), including:

```
BalabanJ, BertzCT, Chi0, Chi0n, Chi0v, Chi1, Chi1n, Chi1v,
Chi2n, Chi2v, Chi3n, Chi3v, Chi4n, Chi4v, ExactMolWt,
FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3,
FractionCSP3, HallKierAlpha, HeavyAtomCount, HeavyAtomMolWt,
Kappa1, Kappa2, Kappa3, LabuteASA, MaxAbsEStateIndex,
MaxEStateIndex, MinAbsEStateIndex, MinEStateIndex, MolLogP,
MolMR, MolWt, NHOHCount, NOCount, NumAliphaticCarbocycles,
NumAliphaticHeterocycles, NumAliphaticRings,
NumAromaticCarbocycles, NumAromaticHeterocycles,
NumAromaticRings, NumHAcceptors, NumHDonors, NumHeteroatoms,
NumRadicalElectrons, NumRotatableBonds, NumSaturatedCarbocycles,
NumSaturatedHeterocycles, NumSaturatedRings, NumValenceElectrons,
RingCount, TPSA, Ipc, sps, nsps, sa_score
```

## Output Description

`smiles.json` contains the following fields:

| Field | Description |
|---|---|
| `Shielded properties` | List of property names shielded in this run |
| `Mean nRMSE (all 56 properties)` | Mean normalized RMSE across all 56 properties |
| `Mean nRMSE (non-shielded, N props)` | Mean nRMSE over non-shielded properties |
| `Mean nRMSE (shielded, N props)` | Mean nRMSE over shielded properties |
| `Validity` | Fraction of generated SMILES that are valid molecules |
| `Uniqueness` | Fraction of unique structures among valid molecules (after deduplication) |
| `generate` | List of generated molecules, each containing SMILES, structure image filename, and the 56 property values |

## How It Works

1. **Load config**: Read the input file, beam width, generation count, and shielded properties from `config.json`.
2. **Update shield_list**: Write the shielded properties into the `shield_list` variable in `calc_property.py`.
3. **Load model**: Initialize the PropMol model and load `checkpoint_final.ckpt`.
4. **PV→SMILES generation**: For each input sample, use property embeddings to guide beam-search decoding and generate new molecules.
5. **Property evaluation**: Use `calculate_property` to compute the 56 properties of each generated molecule, and compute nRMSE against the reference molecule.
6. **Output results**: Save molecule structure images and a `smiles.json` containing the evaluation metrics.
7. **note**:The initial model used in this paper is the `checkpoint_final.ckpt`, which is based on 56 property vectors. If your project needs to investigate the influence of other molecular properties on the generative model, you will need to retrain this base model. For more details, please refer to SPMM.
## Acknowledgements

We thank the following open-source projects for their contributions to this research:

**Generative models**
- [SPMM](https://github.com/jinhojsk515/SPMM)
- [BIMODAL](https://github.com/ETHmodlab/BIMODAL)
- [REINVENT 3/4](https://github.com/MolecularAI/Reinvent)

**Retrosynthesis analysis**
- [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder)

**Molecule generation evaluation**
- [MOSES](https://github.com/molecularsets/moses)
