# XGBoost Robustness to Missing and Incorrect Data

This ECS 171 project investigates how missing values and incorrect training data affect classification performance. We compare XGBoost with random forest, logistic regression, and a neural network on Adult Income and AI4I Predictive Maintenance.

The central experiment trains two groups of models: one on the original data and one on corrupted copies of the training and validation data. Both groups are evaluated on the same original test set.

## Models and datasets

Each dataset will have four model families:

- XGBoost
- Random forest
- Logistic regression
- Neural network

| Dataset | Prediction target | Cleaned rows in saved EDA | Input features |
|---|---|---:|---:|
| Adult Income | Whether income exceeds $50,000 | 41,439 | 12 |
| AI4I 2020 Predictive Maintenance | Whether a machine fails | 10,000 | 6 |

Adult's cleaned count differs from the source dataset's 48,842 rows because the current loader drops columns and removes duplicate feature combinations. Review and document this rule, including any repeated feature combinations with conflicting labels, before freezing the dataset.

AI4I identifiers and the five failure-mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are excluded from model inputs. The failure-mode flags reveal information about the target.

## Current implementation status

| Component | Status |
|---|---|
| `project-notebooks/eda.ipynb` | EDA for both datasets, with saved findings and plots |
| `src/data.py` | Dataset loading, cleaning, feature selection, stratified 80/20 train/test split, and basic preprocessing implemented |
| `src/corrupt.py` | Function interfaces exist; all four functions currently raise `NotImplementedError` |
| Training, tuning, and evaluation for the four models | To implement |
| Experiment runner and comparison results | To implement |
| Saved model and preprocessing artifacts | To implement |
| Web prediction interface | To implement |
| Pinned dependencies and verified execution instructions | To add |

The sections below describe planned work. The full experiment and web demo are not runnable yet.

## Experimental design: normal versus corrupted data

Split each dataset before injecting corruption. Preserve the original partitions and generate corrupted copies.

| Model group | Training partition | Validation partition | Test partition |
|---|---|---|---|
| Normal baseline | Original training data | Original validation data | Original shared test data |
| Corrupted condition | Corrupted training copy | Corrupted validation copy | The same original shared test data |

Both groups include all four model families. There are eight baseline models across the two datasets, plus eight models for each corruption condition and experimental seed. A baseline can be reused across corruption rates for the same dataset, model configuration, and seed.

### Test-set isolation

- Corrupt only training and validation copies. The experiment runner must never pass test data into a corruption function.
- Preserve test feature values, labels, row identities, and row order. Check that they remain unchanged after experiment preparation.
- "Original" or "normal" means no injected corruption. Adult's naturally missing values remain in every group's original test data.
- Use identical raw test rows and labels for all models. Each model applies its own training-fitted preprocessing to those rows.
- Use validation data for model selection, early stopping, and threshold selection. Freeze these decisions before evaluating on the test set.
- The existing EDA examined the full datasets. Document that limitation and avoid further test-guided development decisions.

### Corruption consistency

- For a given dataset, corruption type, rate, and seed, all four models receive the same corrupted training and validation copies.
- Training and validation use the same corruption definition and rate, with independently sampled corruption within each partition.
- Feature corruption leaves labels unchanged. Label-noise experiments flip only the designated training and validation labels; test labels never change.
- Each model group fits preprocessing on its own training data and uses its corresponding validation data.
- Use comparable tuning budgets across groups and record any differences.

This experiment measures how learning from degraded training and validation data changes performance on an unchanged test population. Corrupting inference-time test inputs is outside this experiment.

## Dataset splitting

The current `split(X, y, seed=42)` function in `src/data.py` shuffles the data and creates a stratified 80/20 training/test split. Stratification approximately preserves the label proportions.

| Dataset | Current training: 80% | Current testing: 20% |
|---|---:|---:|
| Adult Income | 33,151 | 8,288 |
| AI4I | 8,000 | 2,000 |

The proposed validation strategy keeps that 20% test partition and takes 25% of the remaining training partition for validation, also stratified. This produces approximately 60/20/20 overall:

| Dataset | Proposed training: 60% | Proposed validation: 20% | Shared testing: 20% |
|---|---:|---:|---:|
| Adult Income | 24,863 | 8,288 | 8,288 |
| AI4I | 6,000 | 2,000 | 2,000 |

These counts depend on the current cleaning rules. The validation split still needs implementation. Save split indices and the split seed so every experiment uses identical partitions. Keep split randomness separate from corruption and model-training randomness.

## Corruption implementation plan

Apply feature corruption to the original feature columns before encoding or scaling. Every function must return a copy and preserve row alignment.

| Function | Intended behavior |
|---|---|
| `inject_missing_mcar` | Uniformly sample a requested fraction of eligible, observed cells and replace them with `NaN`; optionally restrict eligible columns |
| `inject_missing_mnar` | In a selected numeric column, sample values with probabilities favoring the upper or lower tail, then replace the selected values with `NaN` |
| `flip_labels` | For symmetric noise, sample a fraction of binary labels and swap 0 and 1; define and document asymmetric behavior separately |
| `perturb_features` | Sample observed numeric cells and add zero-mean Gaussian measurement noise, scaled by feature-specific training standard deviations |

Implementation requirements:

- Validate rates, column names, numeric eligibility, and noise settings.
- Define the rate denominator explicitly. For missingness, use the fraction of eligible, originally observed cells newly erased, preserving existing missing values.
- Report achieved corruption and final missingness rates. Single-column MNAR rates and whole-table MCAR rates describe different scopes.
- Derive numeric noise scales from original training features and reuse them for validation; do not estimate them from the test set.
- Document permissible value bounds and handling of discrete features. Ensure measurement noise cannot produce invalid inputs for downstream log transforms.
- Record seeds and corruption settings. Prefer nested random masks across rates where practical.
- Verify zero-rate identity, reproducibility, achieved counts, unchanged source data, preserved schema/index alignment, and unchanged test data.

For the presentation, prioritize random missingness and numeric measurement noise. Add MNAR and label-noise comparisons after the core experiment works. Keep corruption types separate in results unless a combined-corruption experiment is explicitly defined.

## Preprocessing and model development

Use the EDA decisions as the starting point:

- Retain outliers in both datasets; they contain useful signal.
- Let XGBoost handle missing numeric values natively, with a documented categorical encoding strategy.
- Use an imputation-based comparison pipeline for random forest, logistic regression, and the neural network. Add fixed missing-value indicators so they retain information about missingness.
- Scale numeric inputs for logistic regression and the neural network.
- Evaluate `log1p` for Adult's `capital_gain` and `capital_loss`, and optionally AI4I's `Rotational_speed`, for logistic regression and the neural network. Select these choices using validation data.
- Fit imputation values, category mappings, scaling statistics, and any learned transformations only on the relevant training partition.
- Address class imbalance, especially AI4I's 339 failures out of 10,000 rows. Calculate any class weights from the training labels available to that model group.

For the neural network, start with a small feedforward candidate such as encoded inputs -> 64 hidden units -> 32 hidden units -> one binary output. Validate the architecture, use early stopping, and train a separate model for each dataset and condition.

Start with a small validation-based tuning budget. Freeze the main comparison configurations across corruption rates, while using each run's matching validation partition for the declared early-stopping policy. Record settings for all four model families.

## Experiments and evaluation

Initial core experiment grid:

- Both datasets and all four model families.
- Normal baseline plus random missingness and numeric measurement noise.
- Additional corruption rates of 0%, 10%, 20%, 30%, and 40%.
- A fixed, recorded noise strength for the initial measurement-noise experiment.
- Three experimental seeds if runtime permits, using the same saved dataset split.

At 0% injected corruption, original missing values remain. Tune using validation data, then evaluate the frozen runs on the shared original test set.

Report accuracy, precision, recall, F1, and a precision-recall summary. Choose and document one primary precision-recall calculation for every model. Average precision (AP) and trapezoidal PR-AUC are different metrics and must be labeled separately.

Accuracy alone is insufficient: an AI4I model that always predicts "no failure" already achieves 96.61% accuracy while detecting no failures.

Save one result row per run with dataset, model family, normal/corrupted group, corruption type, rate and scope, noise strength, split seed, corruption/training seeds, hyperparameters, threshold, metrics, and training time. Save per-example test predictions for paired comparisons. Summarize repeated runs with means and variability; repeated seeds on one fixed test split do not measure uncertainty across new test populations.

Presentation outputs:

- Baseline comparison table for all four model families on both datasets.
- Test performance versus additional missingness.
- Test performance versus measurement-noise rate.
- Normal-versus-corrupted comparisons within each model family.
- Selected confusion matrices and a concise account of findings and limitations.

## Web and live-demo plan

The interface should demonstrate actual inference using saved models and fitted preprocessing.

1. Select Adult Income or AI4I.
2. Select a model family and normal or corrupted training condition.
3. Load a prepared example or enter feature values, allowing supported missing inputs.
4. Run prediction and display the predicted class and clearly labeled model score.
5. Show the corresponding experiment settings and comparison results.

Save model weights, fitted preprocessing, feature order, input schema, class mapping, and decision threshold together. Load artifacts when the application starts. Keep dataset downloads and model training outside the live prediction flow.

Verify that web predictions match the saved-model inference path. Check missing fields, unknown categories, invalid numeric inputs, and each selectable artifact. Prepare locally available examples and a version that runs without internet. Confirm whether the presentation also requires a publicly accessible deployment.


## Schedule for the November 12, 2026 presentation

| Date | Required milestone |
|---|---|
| November 8 | Confirm ownership and existing work; freeze cleaning/split definitions; implement and check core corruption; get a small end-to-end training run working |
| November 9 | Run normal baselines for all four model families on both datasets; add the validation split; complete core corruption checks; connect a saved model to the web interface |
| November 10 | Finish the main corruption comparisons; save artifacts and results; generate plots; integrate the remaining models into the demo |
| November 11 | Freeze configurations and results; complete slides and README execution instructions; test a fresh application startup and offline demo; rehearse within five minutes |
| November 12 | Verify the frozen application and examples before presenting; demonstrate real predictions and the main experimental findings |

Before presenting, have:

- Reproducible normal and corrupted comparisons for all four model families on both datasets.
- Verified isolation of the shared test set.
- Saved models, preprocessing, configurations, predictions, and results.
- A working web interface, prepared examples, and a backup recording.
- Pinned dependencies and tested setup, experiment, and demo launch instructions.
- A five-minute presentation with approximately one minute reserved for the live demo.

## Research reference

Chen, T., and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of KDD '16. <https://doi.org/10.1145/2939672.2939785>

The project is motivated by XGBoost's handling of missing values. Expected advantages are hypotheses to test, not conclusions established by the EDA.
