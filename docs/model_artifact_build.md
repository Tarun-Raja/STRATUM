
# STRATUM — Model Artifact Build

## Prerequisites

- STRATUM project repository.
- Compatible Python environment.
- Dependencies installed from requirements.txt.
- Original verified Infineon AlCu dataset.
- Preserved research result JSON files.

The dataset must be placed at:

data/raw/apc_spc_AlCu_inputs_outputs.csv

## Artifact Generation

1. Open the STRATUM repository in VS Code.
2. Activate the project virtual environment.
3. Install the requirements.
4. Select the virtual environment as the notebook kernel.
5. Open notebooks/17_model_serialization.ipynb.
6. Execute the notebook sequentially.
7. Verify final held-out metric reproduction.
8. Generate the model bundle.
9. Verify artifact checksums.
10. Execute the automated inference tests.
11. Verify the independent Python inference process.

The initial model bundle is generated at:

ml/artifacts/1.0.0/

The inference test fixture is generated at:

ml/artifacts/inference_fixture.json

## Automated Verification

From the repository root:

python -m pytest tests/test_inference.py -v

## Version Policy

Existing released model bundles must not
be silently overwritten.

A changed model configuration or training
population requires a new appropriately
versioned bundle and evaluation evidence.

The application must load an explicitly
configured model version.

## Security

Only trusted, locally generated joblib
artifacts may be loaded.

User-uploaded model files are not supported.

Artifact checksums detect file modifications
but do not establish trusted origin.

## Deployment

The deployed backend requires the complete
generated model bundle.

The raw research dataset is not required
for ordinary inference after artifact generation.

The deployment environment must match
the recorded model runtime versions.

Binary model artifacts and the raw dataset
are excluded from Git.

Artifact generation currently uses the
serialization notebook.

A non-interactive artifact build command
should be implemented during application
development for automated deployment.