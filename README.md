# AdvQFuse Research Package
<img width="4389" height="2764" alt="fig_problem_formulation-1" src="https://github.com/user-attachments/assets/a90bce2b-a9f4-44a8-a47f-144dbe9a8943" />


## Scientific position

This repository is a  remote-sensing-specific AdvQFuse-RS concept. Remote sensing remains a demanding transfer and multimodal-fusion domain, but it is not the title, exclusive scope, or central claim.

The broad question is:

> Can a model's prediction trajectory across controlled precision interventions reveal sample-specific failure risk, and can that signal support compute-aware, risk-controlled multimodal inference across unseen model families, tasks, datasets, precisions, and shifts?

Quantization is treated as a controlled parameter intervention. The resulting **precision path** is represented through predictive drift, rank changes, path curvature, semantic answer changes, hidden-state alignment, and cross-modal counterfactual evidence.

The study design requires:

- at least four independent VLM families;
- BF16/FP16, INT8, INT4, and lower-precision path points where supported;
- at least three task families and eight datasets;
- clean, corruption, adversarial-image, adversarial-text, and image-text conflict evaluation;
- leave-one-model-family-out, leave-one-dataset-out, leave-one-attack-out, and unseen-precision tests;
- action-conditioned risk prediction for accept, reobserve, escalate, ensemble, and abstain;
- finite-sample group-wise selective-risk calibration;
- matched-compute accounting for every precision point, prompt, view, and intervention;
- results that remain positive when Bonsai and all remote-sensing datasets are removed.

## Dataset support

The package includes audited layout detection and normalized manifest adapters for:

- EarthVQA;
- FloodNet VQA, including DatasetNinja/Hugging Face annotations with separate question and answer tags;
- RSVQA-HR and RSVQA-LR;
- UAV-OBB;
- SEN12MS structure validation with mandatory optical, SAR, and ground-truth land-cover evidence.

The UAV-OBB-QA generator supports normalized corner OBBs, center-angle OBBs, and DOTA-style corners. It loads class names from `data.yaml`, preserves split-qualified identities, generates both positive and negative existence questions, and adds dominant-class and orientation questions.


## Core method

1. Evaluate aligned model checkpoints across a controlled precision ladder.
2. Compute precision-path geometry and cross-modal counterfactual features.
3. Encode the path with a lightweight set encoder or calibrated tabular baseline.
4. Estimate error and action-conditioned residual risk.
5. Select accept, reobserve, escalate, ensemble, or abstain under a calibrated risk bound.
6. Test transfer to held-out model families, datasets, attacks, and precision edges.


## Quick start: current Colab workspace

```bash
python -m pip install -e ".[dev,plots]"
python scripts/scan_existing_datasets.py --search-root /content --output configs/datasets.local.yaml
python scripts/validate_datasets.py --config configs/datasets.local.yaml --report results/dataset_validation.json
python scripts/build_all_manifests.py --config configs/datasets.local.yaml --output-dir data/derived/manifests --include-rsvqa-lr
pytest -q
```

Generate the expanded visual suite:

```bash
python scripts/generate_v4_extended_suite.py
python scripts/build_full_figure_gallery.py
```

