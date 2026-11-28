
"""
STRATUM - Versioned Model Registry.

Manage trusted, locally generated fitted
model bundles and associated research metadata.
"""

import hashlib
import json
import platform
import re
import shutil
import tempfile

from pathlib import Path
from importlib.metadata import version

import joblib


TARGETS = (
    "mean_thickness",
    "mean_resistance",
)

MODEL_FILES = {
    "mean_thickness": "thickness.joblib",
    "mean_resistance": "resistance.joblib",
}

METADATA_FILES = (
    "metadata.json",
    "input_schema.json",
    "input_warning_profile.json",
    "global_explainability.json",
    "evaluation.json",
)


def sha256_file(path):

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


def read_json(path):

    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def write_json(path, data):

    Path(path).write_text(
        json.dumps(
            data,
            indent=4,
            allow_nan=False,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def runtime_versions():

    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "joblib",
    ]

    return {
        "python": platform.python_version(),

        **{
            package: version(package)

            for package in packages
        },
    }


def validate_version(model_version):

    if not isinstance(model_version, str):

        raise ValueError(
            "Model version must be a string."
        )

    if not re.fullmatch(
        r"\d+\.\d+\.\d+",
        model_version,
    ):

        raise ValueError(
            "Expected a semantic version such as 1.0.0."
        )

    return model_version


def verify_bundle(
    bundle_path,
    check_runtime=True,
):

    """
    Verify bundle contents and recorded checksums.

    This does not authenticate an untrusted bundle.
    """

    bundle_path = Path(bundle_path)

    manifest = read_json(
        bundle_path / "manifest.json"
    )

    expected_files = (
        set(MODEL_FILES.values())
        | set(METADATA_FILES)
    )

    if set(manifest["files"]) != expected_files:

        raise ValueError(
            "Unexpected artifact manifest contents."
        )

    for filename, expected_hash in (
        manifest["files"].items()
    ):

        artifact = bundle_path / filename

        if not artifact.is_file():

            raise FileNotFoundError(artifact)

        if sha256_file(artifact) != expected_hash:

            raise ValueError(
                f"Artifact checksum mismatch: {filename}"
            )

    metadata = read_json(
        bundle_path / "metadata.json"
    )

    if metadata["model_version"] != manifest[
        "model_version"
    ]:

        raise ValueError(
            "Model version metadata mismatch."
        )

    if check_runtime:

        current = runtime_versions()

        recorded = metadata["runtime_versions"]

        differences = {
            package: {
                "expected": expected,

                "installed": current.get(package),
            }

            for package, expected in recorded.items()

            if current.get(package) != expected
        }

        if differences:

            raise RuntimeError(
                f"Incompatible model runtime: {differences}"
            )

    return manifest


def save_model_bundle(
    artifact_root,
    model_version,
    models,
    metadata,
    input_schema,
    warning_profile,
    global_explainability,
    evaluation,
):

    """
    Save a complete immutable model version.

    Stage files in a temporary directory before
    publishing the final bundle directory.
    """

    model_version = validate_version(
        model_version
    )

    if set(models) != set(TARGETS):

        raise ValueError(
            "Both material-property models are required."
        )

    artifact_root = Path(artifact_root)

    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle_path = (
        artifact_root / model_version
    )

    if bundle_path.exists():

        raise FileExistsError(
            f"Model version already exists: {bundle_path}"
        )

    staging_path = Path(
        tempfile.mkdtemp(
            prefix=".stratum-build-",
            dir=artifact_root,
        )
    )

    try:

        for target in TARGETS:

            joblib.dump(
                models[target],

                staging_path / MODEL_FILES[target],

                compress=3,
            )

        bundle_metadata = {
            **metadata,

            "model_version": model_version,

            "runtime_versions": runtime_versions(),
        }

        documents = {
            "metadata.json": bundle_metadata,

            "input_schema.json": input_schema,

            "input_warning_profile.json": (
                warning_profile
            ),

            "global_explainability.json": (
                global_explainability
            ),

            "evaluation.json": evaluation,
        }

        for filename, document in documents.items():

            write_json(
                staging_path / filename,
                document,
            )

        artifact_files = sorted(
            set(MODEL_FILES.values())
            | set(METADATA_FILES)
        )

        manifest = {
            "project": "STRATUM",

            "model_version": model_version,

            "checksum_algorithm": "SHA256",

            "files": {
                filename: sha256_file(
                    staging_path / filename
                )

                for filename in artifact_files
            },
        }

        write_json(
            staging_path / "manifest.json",
            manifest,
        )

        verify_bundle(
            staging_path,
            check_runtime=True,
        )

        staging_path.rename(bundle_path)

    except Exception:

        shutil.rmtree(
            staging_path,
            ignore_errors=True,
        )

        raise

    return bundle_path


def load_model_bundle(bundle_path):

    """
    Load trusted local artifacts after verifying
    checksums and compatible runtime versions.

    Never use this function on user-uploaded
    or otherwise untrusted model files.
    """

    bundle_path = Path(bundle_path)

    manifest = verify_bundle(
        bundle_path,
        check_runtime=True,
    )

    models = {
        target: joblib.load(
            bundle_path / MODEL_FILES[target]
        )

        for target in TARGETS
    }

    return {
        "manifest": manifest,

        "models": models,

        "metadata": read_json(
            bundle_path / "metadata.json"
        ),

        "input_schema": read_json(
            bundle_path / "input_schema.json"
        ),

        "warning_profile": read_json(
            bundle_path / "input_warning_profile.json"
        ),

        "global_explainability": read_json(
            bundle_path / "global_explainability.json"
        ),

        "evaluation": read_json(
            bundle_path / "evaluation.json"
        ),
    }