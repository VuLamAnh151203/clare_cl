#!/usr/bin/env python3
"""Standalone CLARE continual-learning workflow for X-VLA on LIBERO.

This file intentionally does not modify or import the existing LoRA workflow
runner. It keeps CLARE-specific training/evaluation isolated so the original
`libero_continual_2.py` and replay scripts remain untouched.
"""

from __future__ import annotations

import importlib.util
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TRAIN_CHILD_FLAG = "--clare-train-child"
EVAL_CHILD_FLAG = "--clare-eval-child"


class ConfigError(ValueError):
    """Raised when an environment variable cannot be parsed."""


class TeeStream:
    """Mirror writes to the original stream and a log file."""

    def __init__(self, stream: Any, log_file: Any) -> None:
        self.stream = stream
        self.log_file = log_file
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, text: str) -> int:
        written = self.stream.write(text)
        self.log_file.write(text)
        return written

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def parse_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean value, got {raw!r}")


def parse_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def parse_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def parse_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float, got {raw!r}") from exc


def parse_csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ConfigError(f"{name} must contain at least one value")
    return values


def parse_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    try:
        if raw.strip().startswith("["):
            values = json.loads(raw)
        else:
            values = [item.strip() for item in raw.split(",") if item.strip()]
        parsed = [int(value) for value in values]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{name} must be a comma-separated integer list or JSON list") from exc
    if not parsed:
        raise ConfigError(f"{name} must contain at least one task id")
    return parsed


def parse_json_map(name: str, default: dict[str, str]) -> dict[str, str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return dict(default)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} must be valid JSON, got {raw!r}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a JSON object")
    if not all(isinstance(key, str) and isinstance(val, str) for key, val in value.items()):
        raise ConfigError(f"{name} must map strings to strings")
    return value


def parse_shape(name: str, default: list[int]) -> list[int]:
    values = parse_int_list(name, default)
    if not values:
        raise ConfigError(f"{name} must not be empty")
    return values


def cli_bool(value: bool) -> str:
    return str(value).lower()


def task_ids_arg(task_ids: list[int]) -> str:
    return json.dumps(task_ids, separators=(",", ":"))


def command_to_string(command: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


@dataclass(frozen=True)
class Config:
    workdir: Path
    dataset_input_root: Path
    dataset_work_root: Path
    output_root: Path
    run_name: str
    run_output_root: Path
    results_file: Path
    eval_results_file: Path
    run_log_file: Path
    libero_config_path: Path
    libero_datasets: Path
    libero_benchmark_root: str
    write_libero_config: bool
    base_model: str
    eval_policy_path: str
    suites: list[str]
    convert_suites: list[str]
    dataset_repo_ids: dict[str, str]
    train_task_ids: list[int]
    test_task_ids: list[int]
    train_steps: int
    batch_size: int
    num_workers: int
    device: str
    control_mode: str
    n_eval_episodes: int
    eval_batch_size: int
    install_deps: bool
    copy_datasets: bool
    convert_datasets: bool
    run_train: bool
    run_eval: bool
    train_all_suites: bool
    dry_run: bool
    cudnn_benchmark: str
    cublas_workspace_config: str
    env_max_parallel_tasks: int
    eval_freq: int
    policy_push_to_hub: bool
    policy_dtype: str
    optimizer_lr: str
    scheduler_warmup_steps: int
    scheduler_decay_steps: int
    scheduler_decay_lr: str
    action_mode: str
    policy_use_proprio: bool
    policy_num_image_views: int | None
    policy_empty_cameras: int | None
    freeze_vision_encoder: bool
    freeze_language_encoder: bool
    train_policy_transformer: bool
    train_soft_prompts: bool
    merge_lora_between_suites: bool
    rename_map: dict[str, str]
    tokenizer_name: str
    tokenizer_max_length: int
    tokenizer_task_key: str
    domain_id: int
    normalizer_eps: str
    image_shape: list[int]
    state_shape: list[int]
    action_shape: list[int]
    preprocessor_device: str
    postprocessor_device: str
    clare_root: Path
    clare_config_path: Path
    clare_regenerate_config: bool
    clare_target_regex: str
    clare_max_target_modules: int
    clare_adapter_hidden_dim: int
    clare_discriminator_hidden_dim: int
    clare_discriminator_latent_dim: int
    clare_expand_threshold: float
    clare_detect_steps: int
    clare_detect_batch_size: int
    clare_detect_num_workers: int
    clare_train_discriminator_steps: int
    clare_train_discriminator_batch_size: int
    clare_train_discriminator_num_workers: int


def load_config() -> Config:
    workdir = Path(env("WORKDIR", "/kaggle/working")).expanduser()
    output_root = Path(env("OUTPUT_ROOT", str(workdir / "outputs" / "continual_learning_clare_xvla"))).expanduser()
    run_name = env("RUN_NAME", datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_output_root = output_root / "runs" / run_name
    dataset_work_root = Path(env("DATASET_WORK_ROOT", str(workdir / "IPEC-COMMUNITY"))).expanduser()
    libero_dataset_parent = Path(env("LIBERO_DATASET_PARENT", str(workdir / "libero"))).expanduser()

    dataset_repo_ids = {
        "libero_spatial": env(
            "LIBERO_SPATIAL_REPO_ID",
            "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
        ),
        "libero_goal": env(
            "LIBERO_GOAL_REPO_ID",
            "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
        ),
        "libero_10": env(
            "LIBERO_10_REPO_ID",
            "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
        ),
        "libero_object": env(
            "LIBERO_OBJECT_REPO_ID",
            "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
        ),
    }

    suites = parse_csv("SUITES", ["libero_spatial", "libero_goal", "libero_10", "libero_object"])
    convert_suites = parse_csv(
        "CONVERT_SUITES",
        ["libero_10", "libero_object", "libero_goal", "libero_spatial"],
    )
    missing_suites = [suite for suite in set(suites + convert_suites) if suite not in dataset_repo_ids]
    if missing_suites:
        raise ConfigError("Missing dataset repo id env vars for suites: " + ", ".join(sorted(missing_suites)))

    merge_lora_between_suites = parse_bool("MERGE_LORA_BETWEEN_SUITES", False)
    if merge_lora_between_suites:
        raise ConfigError("CLARE-X-VLA does not support MERGE_LORA_BETWEEN_SUITES=1")

    clare_root = Path(env("CLARE_ROOT", str(Path(__file__).resolve().parent / "clare"))).expanduser()
    clare_config_env = env("CLARE_CONFIG_PATH", "")
    clare_config_path = (
        Path(clare_config_env).expanduser()
        if clare_config_env
        else run_output_root / "clare_xvla_config"
    )

    return Config(
        workdir=workdir,
        dataset_input_root=Path(
            env(
                "DATASET_INPUT_ROOT",
                "/kaggle/input/datasets/anhvlm/libero-dataset/libero_dataset/IPEC-COMMUNITY",
            )
        ).expanduser(),
        dataset_work_root=dataset_work_root,
        output_root=output_root,
        run_name=run_name,
        run_output_root=run_output_root,
        results_file=Path(env("RESULTS_FILE", str(output_root / "results.json"))).expanduser(),
        eval_results_file=Path(env("EVAL_RESULTS_FILE", str(output_root / "evaluation_results.json"))).expanduser(),
        run_log_file=Path(env("RUN_LOG_FILE", str(output_root / "run.log"))).expanduser(),
        libero_config_path=Path(env("LIBERO_CONFIG_PATH", str(workdir / ".libero"))).expanduser(),
        libero_datasets=Path(env("LIBERO_DATASETS", str(libero_dataset_parent / "datasets"))).expanduser(),
        libero_benchmark_root=env("LIBERO_BENCHMARK_ROOT", ""),
        write_libero_config=parse_bool("WRITE_LIBERO_CONFIG", True),
        base_model=env("BASE_MODEL", "lerobot/xvla-base"),
        eval_policy_path=env("EVAL_POLICY_PATH", ""),
        suites=suites,
        convert_suites=convert_suites,
        dataset_repo_ids=dataset_repo_ids,
        train_task_ids=parse_int_list("TRAIN_TASK_IDS", list(range(8))),
        test_task_ids=parse_int_list("TEST_TASK_IDS", [8, 9]),
        train_steps=parse_int("TRAIN_STEPS", 8000),
        batch_size=parse_int("BATCH_SIZE", 16),
        num_workers=parse_int("NUM_WORKERS", 0),
        device=env("DEVICE", "cuda"),
        control_mode=env("CONTROL_MODE", "absolute"),
        n_eval_episodes=parse_int("N_EVAL_EPISODES", 10),
        eval_batch_size=parse_int("EVAL_BATCH_SIZE", 1),
        install_deps=parse_bool("INSTALL_DEPS", True),
        copy_datasets=parse_bool("COPY_DATASETS", True),
        convert_datasets=parse_bool("CONVERT_DATASETS", True),
        run_train=parse_bool("RUN_TRAIN", True),
        run_eval=parse_bool("RUN_EVAL", True),
        train_all_suites=parse_bool("TRAIN_ALL_SUITES", False),
        dry_run=parse_bool("DRY_RUN", False),
        cudnn_benchmark=env("CUDNN_BENCHMARK", "0"),
        cublas_workspace_config=env("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        env_max_parallel_tasks=parse_int("ENV_MAX_PARALLEL_TASKS", 1),
        eval_freq=parse_int("EVAL_FREQ", 0),
        policy_push_to_hub=parse_bool("POLICY_PUSH_TO_HUB", False),
        policy_dtype=env("POLICY_DTYPE", ""),
        optimizer_lr=env("OPTIMIZER_LR", "1e-4"),
        scheduler_warmup_steps=parse_int("SCHEDULER_WARMUP_STEPS", 1000),
        scheduler_decay_steps=parse_int("SCHEDULER_DECAY_STEPS", 30000),
        scheduler_decay_lr=env("SCHEDULER_DECAY_LR", "2.5e-6"),
        action_mode=env("ACTION_MODE", "auto"),
        policy_use_proprio=parse_bool("POLICY_USE_PROPRIO", True),
        policy_num_image_views=parse_optional_int("POLICY_NUM_IMAGE_VIEWS"),
        policy_empty_cameras=parse_optional_int("POLICY_EMPTY_CAMERAS"),
        freeze_vision_encoder=parse_bool("FREEZE_VISION_ENCODER", False),
        freeze_language_encoder=parse_bool("FREEZE_LANGUAGE_ENCODER", False),
        train_policy_transformer=parse_bool("TRAIN_POLICY_TRANSFORMER", True),
        train_soft_prompts=parse_bool("TRAIN_SOFT_PROMPTS", True),
        merge_lora_between_suites=merge_lora_between_suites,
        rename_map=parse_json_map("RENAME_MAP", {"observation.images.wrist_image": "observation.images.image2"}),
        tokenizer_name=env("TOKENIZER_NAME", "facebook/bart-large"),
        tokenizer_max_length=parse_int("TOKENIZER_MAX_LENGTH", 50),
        tokenizer_task_key=env("TOKENIZER_TASK_KEY", "task"),
        domain_id=parse_int("DOMAIN_ID", 3),
        normalizer_eps=env("NORMALIZER_EPS", "1e-08"),
        image_shape=parse_shape("IMAGE_SHAPE", [3, 224, 224]),
        state_shape=parse_shape("STATE_SHAPE", [8]),
        action_shape=parse_shape("ACTION_SHAPE", [20]),
        preprocessor_device=env("PREPROCESSOR_DEVICE", env("DEVICE", "cuda")),
        postprocessor_device=env("POSTPROCESSOR_DEVICE", "cpu"),
        clare_root=clare_root,
        clare_config_path=clare_config_path,
        clare_regenerate_config=parse_bool("CLARE_REGENERATE_CONFIG", False),
        clare_target_regex=env("CLARE_TARGET_REGEX", ""),
        clare_max_target_modules=parse_int("CLARE_MAX_TARGET_MODULES", 24),
        clare_adapter_hidden_dim=parse_int("CLARE_ADAPTER_HIDDEN_DIM", 1024),
        clare_discriminator_hidden_dim=parse_int("CLARE_DISCRIMINATOR_HIDDEN_DIM", 256),
        clare_discriminator_latent_dim=parse_int("CLARE_DISCRIMINATOR_LATENT_DIM", 128),
        clare_expand_threshold=parse_float("CLARE_EXPAND_THRESHOLD", 1.0),
        clare_detect_steps=parse_int("CLARE_DETECT_STEPS", 20),
        clare_detect_batch_size=parse_int("CLARE_DETECT_BATCH_SIZE", parse_int("BATCH_SIZE", 16)),
        clare_detect_num_workers=parse_int("CLARE_DETECT_NUM_WORKERS", 0),
        clare_train_discriminator_steps=parse_int("CLARE_TRAIN_DISCRIMINATOR_STEPS", 200),
        clare_train_discriminator_batch_size=parse_int(
            "CLARE_TRAIN_DISCRIMINATOR_BATCH_SIZE", parse_int("BATCH_SIZE", 16)
        ),
        clare_train_discriminator_num_workers=parse_int("CLARE_TRAIN_DISCRIMINATOR_NUM_WORKERS", 0),
    )


def safe_config_for_results(config: Config) -> dict[str, Any]:
    payload = dict(config.__dict__)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def print_config(config: Config) -> None:
    print("========== CLARE-X-VLA continual run config ==========")
    for key, value in safe_config_for_results(config).items():
        print(f"{key}: {value}")
    print("======================================================")


def run_command(command: list[str | Path], config: Config, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + command_to_string(command))
    if config.dry_run:
        return subprocess.CompletedProcess([str(part) for part in command], 0, stdout="", stderr="")
    if capture:
        try:
            return subprocess.run(
                [str(part) for part in command],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            output = exc.stdout or exc.output
            if output:
                print(output)
            raise
    return subprocess.run([str(part) for part in command], check=True, text=True)


def install_dependencies(config: Config) -> None:
    package_groups = [
        ["robosuite==1.4.1"],
        ["bddl", "easydict", "cloudpickle", "num2words"],
        ["imageio[ffmpeg]"],
        ["lerobot[libero]"],
        ["lerobot[xvla]"],
    ]
    for packages in package_groups:
        run_command([sys.executable, "-m", "pip", "install", *packages], config)

    if parse_bool("CLARE_UPGRADE_TRANSFORMERS", True):
        transformers_version = os.environ.get("CLARE_TRANSFORMERS_VERSION", "4.53.3").strip()
        if transformers_version:
            transformers_spec = (
                transformers_version
                if any(op in transformers_version for op in ("=", "<", ">", "~", "!"))
                else f"transformers=={transformers_version}"
            )
        else:
            transformers_spec = "transformers>=4.53.0"
        run_command([sys.executable, "-m", "pip", "install", "--upgrade", transformers_spec], config)
    else:
        print(
            "Skipping transformers upgrade because CLARE_UPGRADE_TRANSFORMERS=0. "
            "CLARE PEFT may require transformers with HybridCache support."
        )

    peft_project = config.clare_root / "peft_lsy"
    if parse_bool("CLARE_INSTALL_EDITABLE", False):
        if not peft_project.exists():
            raise FileNotFoundError(f"CLARE PEFT project directory does not exist: {peft_project}")
        if not ((peft_project / "setup.py").exists() or (peft_project / "pyproject.toml").exists()):
            raise FileNotFoundError(
                "CLARE_INSTALL_EDITABLE=1 was requested, but the PEFT directory is not installable: "
                f"{peft_project}. Expected setup.py or pyproject.toml."
            )
        run_command([sys.executable, "-m", "pip", "install", "-e", str(peft_project)], config)
    else:
        print(
            "Skipping editable install for CLARE PEFT. The runner loads it directly from "
            f"{peft_project / 'src'}. Set CLARE_INSTALL_EDITABLE=1 to force `pip install -e`."
        )


def find_libero_benchmark_root() -> Path:
    try:
        spec = importlib.util.find_spec("libero.libero")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cannot find the 'libero' package. Install dependencies or set INSTALL_DEPS=1."
        ) from exc
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate libero.libero package files.")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def ensure_libero_config(config: Config) -> None:
    config_file = config.libero_config_path / "config.yaml"
    if not config.write_libero_config and config_file.exists():
        return
    benchmark_root = (
        Path(config.libero_benchmark_root).expanduser().resolve()
        if config.libero_benchmark_root
        else find_libero_benchmark_root()
    )
    if config.dry_run:
        print(f"DRY_RUN enabled; would write LIBERO config at {config_file}")
        return
    config.libero_config_path.mkdir(parents=True, exist_ok=True)
    config.libero_datasets.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                f"datasets: {config.libero_datasets}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Created LIBERO config: {config_file}")


def dataset_root(config: Config, suite: str) -> Path:
    return config.dataset_work_root / config.dataset_repo_ids[suite].split("/")[-1]


def require_dataset_root(config: Config, suite: str) -> Path:
    root = dataset_root(config, suite)
    if config.dry_run:
        return root
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist for {suite}: {root}. "
            "Set COPY_DATASETS=1 or DATASET_WORK_ROOT correctly."
        )
    return root


def copy_datasets(config: Config) -> None:
    for suite, repo_id in config.dataset_repo_ids.items():
        name = repo_id.split("/")[-1]
        source = config.dataset_input_root / name
        target = config.dataset_work_root / name
        if target.exists():
            print(f"Dataset already present: {target}")
            continue
        if config.dry_run:
            print(f"DRY_RUN enabled; would copy {source} -> {target}")
            continue
        if not source.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying dataset {source} -> {target}")
        shutil.copytree(source, target)


def dataset_codebase_version(root: Path) -> str | None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        with info_path.open("r", encoding="utf-8") as file:
            info = json.load(file)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Dataset info file is not valid JSON: {info_path}") from exc
    version = info.get("codebase_version")
    return str(version) if version is not None else None


def convert_datasets(config: Config) -> None:
    for suite in config.convert_suites:
        root = require_dataset_root(config, suite)
        version = dataset_codebase_version(root)
        if version == "v3.0":
            print(f"Dataset already converted to v3.0; skipping conversion for {suite}: {root}")
            continue
        if version and version != "v2.1":
            raise ValueError(
                f"Dataset {suite} has unsupported codebase_version '{version}' at {root}. "
                "This workflow can convert v2.1 datasets to v3.0, or skip datasets already at v3.0."
            )
        if version is None:
            print(
                f"Dataset version could not be detected for {suite} at {root}; "
                "running the LeRobot converter and letting it validate the dataset."
            )
        run_command(
            [
                sys.executable,
                "-m",
                "lerobot.scripts.convert_dataset_v21_to_v30",
                f"--repo-id={config.dataset_repo_ids[suite]}",
                "--push-to-hub",
                "False",
                "--root",
                str(root),
            ],
            config,
        )


def ensure_clare_peft_path(clare_root: Path) -> Path:
    explicit_peft_src = env("CLARE_PEFT_SRC", "").strip()
    peft_src = Path(explicit_peft_src).expanduser() if explicit_peft_src else clare_root / "peft_lsy" / "src"
    if not peft_src.exists():
        existing = []
        if clare_root.exists():
            existing = sorted(path.name for path in clare_root.iterdir())
        hint = (
            "Set CLARE_PEFT_SRC directly to the folder containing the peft package, for example "
            "`/kaggle/input/<dataset>/peft_lsy/src`, or upload the missing `clare/peft_lsy` folder."
            if explicit_peft_src
            else "Upload the full clare folder including `peft_lsy`, or set CLARE_PEFT_SRC to a folder "
            "containing `peft`."
        )
        raise RuntimeError(
            f"Cannot find local CLARE PEFT source: {peft_src}. "
            f"CLARE_ROOT is {clare_root}. Entries there: {existing}. "
            f"{hint}"
        )
    if not (peft_src / "peft").exists():
        raise RuntimeError(
            f"CLARE PEFT source is missing the peft package: {peft_src / 'peft'}. "
            "CLARE_PEFT_SRC must point to the `src` directory, not to `peft_lsy` itself. "
            "Expected layout: CLARE_PEFT_SRC/peft/tuners/clare/..."
        )
    peft_src_str = str(peft_src.resolve())
    if peft_src_str not in sys.path:
        sys.path.insert(0, peft_src_str)
    return peft_src


def patch_clare_runtime() -> None:
    """Patch CLARE's class-level adapter list into an instance-level list."""
    from peft.tuners.clare.model import CLAREModel

    if getattr(CLAREModel, "_xvla_instance_patch", False):
        return
    original_init = getattr(CLAREModel, "__init__", None)

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self._clare_layers = []
        if original_init is not None:
            original_init(self, *args, **kwargs)
        if not hasattr(self, "_clare_layers"):
            self._clare_layers = []

    CLAREModel.__init__ = patched_init
    CLAREModel._xvla_instance_patch = True


def validate_runtime_imports(config: Config) -> None:
    ensure_clare_peft_path(config.clare_root)
    try:
        from peft import CLAREConfig, PeftModel  # noqa: F401
    except Exception as exc:
        if "HybridCache" in str(exc) or "transformers" in str(exc):
            raise RuntimeError(
                "Local CLARE PEFT was found, but it is not compatible with the installed transformers package. "
                "CLARE's PEFT fork imports transformers.HybridCache, which is missing in older Kaggle images. "
                "Use INSTALL_DEPS=1 with CLARE_UPGRADE_TRANSFORMERS=1, or run "
                "`python -m pip install --upgrade \"transformers==4.53.3\"` before this script. "
                "If a notebook imported transformers earlier, restart the Kaggle session after upgrading."
            ) from exc
        raise RuntimeError(
            "Local CLARE PEFT is not importable. Check CLARE_ROOT and peft_lsy installation."
        ) from exc
    patch_clare_runtime()
    try:
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Official LeRobot X-VLA is not importable. Install with `pip install \"lerobot[xvla]\"`."
        ) from exc
    try:
        from lerobot.envs.factory import make_env  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "LeRobot LIBERO environment support is not importable. Install with `pip install \"lerobot[libero]\"`."
        ) from exc


def import_lerobot_runtime_helpers() -> tuple[Any, Any]:
    import logging

    import torch

    try:
        from lerobot.utils.utils import init_logging as lerobot_init_logging
    except ImportError:

        def lerobot_init_logging() -> None:
            logging.basicConfig(level=logging.INFO)

    try:
        from lerobot.utils.utils import get_safe_torch_device as lerobot_get_safe_torch_device
    except ImportError:

        def lerobot_get_safe_torch_device(device: str | torch.device | None, log: bool = False) -> torch.device:
            requested = str(device or "cuda")
            torch_device = torch.device(requested)
            if torch_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device was requested ({requested}), but torch.cuda.is_available() is false. "
                    "Set DEVICE=cpu or enable a GPU runtime."
                )
            if torch_device.type == "mps":
                mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                if not mps_available:
                    raise RuntimeError("MPS device was requested, but it is not available.")
            if log:
                logging.info("Using torch device: %s", torch_device)
            return torch_device

    return lerobot_get_safe_torch_device, lerobot_init_logging


def update_dataclass_type_hints(dataclass_type: type[Any], hints: dict[str, Any]) -> None:
    dataclass_type.__annotations__.update(hints)
    dataclass_fields = getattr(dataclass_type, "__dataclass_fields__", {})
    for name, hint in hints.items():
        if name in dataclass_fields:
            dataclass_fields[name].type = hint


def metadata_feature_mappings(meta: Any) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    seen: set[int] = set()
    features = getattr(meta, "features", None)
    if isinstance(features, dict):
        mappings.append(features)
        seen.add(id(features))
    info = getattr(meta, "info", None)
    if isinstance(info, dict):
        info_features = info.get("features")
        if isinstance(info_features, dict) and id(info_features) not in seen:
            mappings.append(info_features)
    return mappings


def metadata_stats_mappings(meta: Any) -> list[dict[str, Any]]:
    stats = getattr(meta, "stats", None)
    return [stats] if isinstance(stats, dict) else []


def apply_rename_map_to_mapping(mapping: dict[str, Any], rename_map: dict[str, str]) -> None:
    for source, target in rename_map.items():
        if source not in mapping:
            continue
        if target not in mapping:
            mapping[target] = copy.deepcopy(mapping[source])
        del mapping[source]


def add_empty_camera_features(mapping: dict[str, Any], empty_cameras: int) -> None:
    if empty_cameras <= 0:
        return
    template = mapping.get("observation.images.image")
    if template is None:
        template = next(
            (value for key, value in mapping.items() if key.startswith("observation.images.")),
            None,
        )
    if template is None:
        return
    for camera_idx in range(empty_cameras):
        key = f"observation.images.empty_camera_{camera_idx}"
        if key not in mapping:
            mapping[key] = copy.deepcopy(template)


def apply_policy_feature_compatibility(meta: Any, policy_cfg: Any, rename_map: dict[str, str]) -> None:
    for features in metadata_feature_mappings(meta):
        apply_rename_map_to_mapping(features, rename_map)
        add_empty_camera_features(features, int(getattr(policy_cfg, "empty_cameras", 0) or 0))
    for stats in metadata_stats_mappings(meta):
        apply_rename_map_to_mapping(stats, rename_map)


def make_policy_compatible_metadata(meta: Any, policy_cfg: Any, rename_map: dict[str, str]) -> Any:
    policy_meta = copy.deepcopy(meta)
    apply_policy_feature_compatibility(policy_meta, policy_cfg, rename_map)
    return policy_meta


def apply_rename_map_to_batch(batch: dict[str, Any], rename_map: dict[str, str]) -> dict[str, Any]:
    for source, target in rename_map.items():
        if source not in batch:
            continue
        if target not in batch:
            batch[target] = batch[source]
        del batch[source]
    return batch


def prefix_regex_pattern(pattern: str, prefix: str) -> str:
    escaped_prefix = re.escape(prefix)
    if pattern.startswith("^"):
        return "^" + escaped_prefix + pattern[1:]
    return escaped_prefix + pattern


def count_matching_clare_targets(peft_cfg: Any, model: Any) -> int:
    get_module_config = getattr(peft_cfg, "get_module_config", None)
    if get_module_config is None:
        return 0
    count = 0
    for name, _module in model.named_modules():
        if get_module_config(name) is not None:
            count += 1
    return count


def rebuild_clare_module_configs(peft_cfg: Any) -> None:
    if hasattr(peft_cfg, "_module_configs"):
        peft_cfg._module_configs = {}
    process = getattr(peft_cfg, "_process_module_configs", None)
    if process is not None:
        process()


def prefix_clare_target_modules(peft_cfg: Any, prefix: str) -> None:
    target_modules = getattr(peft_cfg, "target_modules", None)
    if isinstance(target_modules, str):
        peft_cfg.target_modules = prefix_regex_pattern(target_modules, prefix)
    elif isinstance(target_modules, dict):
        prefixed: dict[str, Any] = {}
        for pattern, module_cfg in target_modules.items():
            prefixed_pattern = prefix_regex_pattern(str(pattern), prefix)
            module_cfg = copy.deepcopy(module_cfg)
            if hasattr(module_cfg, "pattern"):
                module_cfg.pattern = prefixed_pattern
            prefixed[prefixed_pattern] = module_cfg
        peft_cfg.target_modules = prefixed
    elif isinstance(target_modules, list):
        prefixed_list = []
        for module_cfg in target_modules:
            module_cfg = copy.deepcopy(module_cfg)
            if isinstance(module_cfg, dict):
                pattern = module_cfg.get("pattern")
                if pattern is not None:
                    module_cfg["pattern"] = prefix_regex_pattern(str(pattern), prefix)
            elif hasattr(module_cfg, "pattern"):
                module_cfg.pattern = prefix_regex_pattern(str(module_cfg.pattern), prefix)
            prefixed_list.append(module_cfg)
        peft_cfg.target_modules = prefixed_list
    rebuild_clare_module_configs(peft_cfg)


def ensure_clare_targets_match_wrapped_model(peft_cfg: Any, model: Any, wrapper_prefix: str = "policy.") -> int:
    match_count = count_matching_clare_targets(peft_cfg, model)
    if match_count > 0:
        return match_count
    original_targets = copy.deepcopy(getattr(peft_cfg, "target_modules", None))
    original_module_configs = copy.deepcopy(getattr(peft_cfg, "_module_configs", {}))
    prefix_clare_target_modules(peft_cfg, wrapper_prefix)
    match_count = count_matching_clare_targets(peft_cfg, model)
    if match_count > 0:
        print(
            "Adjusted CLARE target module patterns for wrapped X-VLA policy "
            f"using prefix {wrapper_prefix!r}; matched {match_count} modules."
        )
        return match_count
    peft_cfg.target_modules = original_targets
    if hasattr(peft_cfg, "_module_configs"):
        peft_cfg._module_configs = original_module_configs
    examples = [name for name, _module in list(model.named_modules())[:40]]
    raise RuntimeError(
        "CLARE target modules do not match the wrapped X-VLA model. "
        f"Example wrapped module names: {examples}. "
        "Regenerate the CLARE config or set CLARE_TARGET_REGEX to match wrapped policy modules."
    )


def module_is_excluded(name: str) -> bool:
    lowered = name.lower()
    excluded_tokens = (
        "vision",
        "visual",
        "image",
        "language",
        "token",
        "embed",
        "normalizer",
        "preprocess",
        "postprocess",
        "head",
        "output",
        "projector",
    )
    return any(token in lowered for token in excluded_tokens)


def module_is_in_default_scope(name: str) -> bool:
    lowered = name.lower()
    include_tokens = ("policy_transformer", "transformer", "blocks", "layers", "flow", "dit")
    return any(token in lowered for token in include_tokens)


def generate_clare_config(config: Config) -> Path:
    config_dir = config.clare_config_path
    adapter_config = config_dir / "adapter_config.json"
    if adapter_config.exists() and not config.clare_regenerate_config:
        print(f"Using existing CLARE config: {adapter_config}")
        return config_dir
    if config.dry_run:
        print(f"DRY_RUN enabled; would generate CLARE config at {adapter_config}")
        return config_dir

    validate_runtime_imports(config)
    import torch.nn as nn
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    print(f"Loading X-VLA base model for CLARE target discovery: {config.base_model}")
    policy = XVLAPolicy.from_pretrained(config.base_model)
    target_regex = re.compile(config.clare_target_regex) if config.clare_target_regex else None

    selected: list[dict[str, Any]] = []
    linear_examples: list[str] = []
    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if len(linear_examples) < 20:
            linear_examples.append(name)
        if module_is_excluded(name):
            continue
        if target_regex is not None:
            if not target_regex.search(name):
                continue
        elif not module_is_in_default_scope(name):
            continue
        in_features = int(getattr(module, "in_features", 0))
        out_features = int(getattr(module, "out_features", 0))
        if in_features <= 0 or out_features <= 0:
            continue
        selected.append(
            {
                "name": name,
                "peft_name": f"policy.{name}",
                "in_features": in_features,
                "out_features": out_features,
            }
        )
        if len(selected) >= config.clare_max_target_modules:
            break

    del policy
    if not selected:
        raise RuntimeError(
            "No safe X-VLA nn.Linear target modules were selected for CLARE. "
            f"Examples of linear modules: {linear_examples}. "
            "Set CLARE_TARGET_REGEX to a regex matching transformer linear modules."
        )

    print(f"Selected {len(selected)} CLARE target modules.")
    for item in selected[:10]:
        print(f"  - {item['peft_name']} ({item['in_features']} -> {item['out_features']})")

    target_modules: dict[str, dict[str, Any]] = {}
    for item in selected:
        pattern = "^" + re.escape(item["peft_name"]) + "$"
        target_modules[pattern] = {
            "feature_dim": item["in_features"],
            "out_feature_dim": item["out_features"],
            "batch_first": True,
            "use_trainable_copy": False,
            "add_zero_init_conv_layer": False,
            "discriminator_cfg": {
                "type": "autoencoder",
                "batch_first": True,
                "feature_dim": item["in_features"],
                "feature_fusion": False,
                "fused_feature_dim": None,
                "hidden_dim": config.clare_discriminator_hidden_dim,
                "latent_dim": config.clare_discriminator_latent_dim,
                "num_tokens": None,
                "lora_rank": 32,
                "lora_alpha": 32,
                "use_lora": False,
                "use_momentum": True,
                "momentum": 0.1,
                "max_batches_tracked": 2000,
            },
            "func_adapter_cfg": {
                "hidden_dim": config.clare_adapter_hidden_dim,
                "lora_rank": 32,
                "lora_alpha": 32,
                "use_lora": False,
            },
        }

    first = selected[0]
    payload = {
        "peft_type": "CLARE",
        "task_type": None,
        "auto_mapping": {"base_model_class": "PeftWrapperPolicy", "parent_library": "__main__"},
        "base_model_name_or_path": None,
        "revision": None,
        "target_modules": target_modules,
        "inference_mode": True,
        "batch_first": True,
        "num_learned_task": 0,
        "feature_dim": first["in_features"],
        "out_feature_dim": first["out_features"],
        "use_trainable_copy": False,
        "add_zero_init_conv_layer": False,
        "structure": {},
        "discriminator_cfg": target_modules[next(iter(target_modules))]["discriminator_cfg"],
        "func_adapter_cfg": target_modules[next(iter(target_modules))]["func_adapter_cfg"],
    }

    config_dir.mkdir(parents=True, exist_ok=True)
    adapter_config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (config_dir / "target_modules_summary.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote CLARE config: {adapter_config}")
    return config_dir


def checkpoint_path(config: Config, suite: str) -> Path:
    return config.run_output_root / f"train_{suite}" / "checkpoints" / "last" / "adapter"


def build_common_train_args(config: Config, suite: str, output_dir: Path) -> list[str]:
    root = require_dataset_root(config, suite)
    rename_map = json.dumps(config.rename_map, separators=(",", ":"))
    args = [
        f"--policy.path={config.base_model}",
        f"--dataset.repo_id={config.dataset_repo_ids[suite]}",
        f"--dataset.root={root}",
        f"--env.type=libero",
        f"--env.max_parallel_tasks={config.env_max_parallel_tasks}",
        f"--eval_freq={config.eval_freq}",
        f"--env.task={suite}",
        f"--env.task_ids={task_ids_arg(config.train_task_ids)}",
        f"--output_dir={output_dir}",
        f"--job_name=clare_xvla_{suite}",
        f"--steps={config.train_steps}",
        f"--batch_size={config.batch_size}",
        f"--num_workers={config.num_workers}",
        f"--policy.device={config.device}",
        f"--policy.push_to_hub={cli_bool(config.policy_push_to_hub)}",
        f"--policy.optimizer_lr={config.optimizer_lr}",
        f"--policy.scheduler_warmup_steps={config.scheduler_warmup_steps}",
        f"--policy.scheduler_decay_steps={config.scheduler_decay_steps}",
        f"--policy.scheduler_decay_lr={config.scheduler_decay_lr}",
        f"--policy.action_mode={config.action_mode}",
        f"--policy.use_proprio={cli_bool(config.policy_use_proprio)}",
        f"--policy.tokenizer_name={config.tokenizer_name}",
        f"--policy.tokenizer_max_length={config.tokenizer_max_length}",
        f"--policy.freeze_vision_encoder={cli_bool(config.freeze_vision_encoder)}",
        f"--policy.freeze_language_encoder={cli_bool(config.freeze_language_encoder)}",
        f"--policy.train_policy_transformer={cli_bool(config.train_policy_transformer)}",
        f"--policy.train_soft_prompts={cli_bool(config.train_soft_prompts)}",
        f"--rename_map={rename_map}",
        f"--tokenizer_task_key={config.tokenizer_task_key}",
        f"--domain_id={config.domain_id}",
        f"--expand_threshold={config.clare_expand_threshold}",
        f"--detect_distribution_shift_steps={config.clare_detect_steps}",
        f"--detect_distribution_shift_batch_size={config.clare_detect_batch_size}",
        f"--detect_distribution_shift_num_workers={config.clare_detect_num_workers}",
        f"--train_discriminators_steps={config.clare_train_discriminator_steps}",
        f"--train_discriminators_batch_size={config.clare_train_discriminator_batch_size}",
        f"--train_discriminators_num_workers={config.clare_train_discriminator_num_workers}",
        f"--train_discriminators_log_freq=10",
        f"--train_discriminators_save_freq={max(config.clare_train_discriminator_steps, 1)}",
        f"--train_discriminators_eval_freq=0",
        f"--save_freq={max(config.train_steps, 1)}",
        f"--log_freq=10",
    ]
    if config.policy_dtype:
        args.append(f"--policy.dtype={config.policy_dtype}")
    if config.policy_num_image_views is not None:
        args.append(f"--policy.num_image_views={config.policy_num_image_views}")
    if config.policy_empty_cameras is not None:
        args.append(f"--policy.empty_cameras={config.policy_empty_cameras}")
    return args


def train_suite(config: Config, suite: str, previous_adapter: Path | None) -> Path:
    output_dir = config.run_output_root / f"train_{suite}"
    adapter_checkpoint = checkpoint_path(config, suite)
    if previous_adapter is not None and not config.dry_run and not previous_adapter.exists():
        raise FileNotFoundError(f"Previous CLARE adapter checkpoint does not exist: {previous_adapter}")

    common_args = build_common_train_args(config, suite, output_dir)
    if previous_adapter is None:
        clare_config_dir = generate_clare_config(config)
        peft_arg = f"--peft_cfg_path={clare_config_dir}"
    else:
        peft_arg = f"--peft_weight_path={previous_adapter}"

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        TRAIN_CHILD_FLAG,
        *common_args,
        peft_arg,
    ]

    print("\n" + "=" * 60)
    print(f"CLARE-X-VLA TRAINING on: {suite}")
    print(f"Starting base model: {config.base_model}")
    print(f"Previous adapter: {previous_adapter if previous_adapter else '<initial CLARE config>'}")
    print(f"Output adapter: {adapter_checkpoint}")
    print("=" * 60 + "\n")
    run_command(command, config)
    return adapter_checkpoint


def build_eval_args(config: Config, adapter_checkpoint: Path, suite: str, output_dir: Path) -> list[str]:
    root = require_dataset_root(config, suite)
    rename_map = json.dumps(config.rename_map, separators=(",", ":"))
    args = [
        f"--policy.path={config.base_model}",
        f"--peft_weight_path={adapter_checkpoint}",
        f"--dataset.repo_id={config.dataset_repo_ids[suite]}",
        f"--dataset.root={root}",
        f"--env.type=libero",
        f"--env.task={suite}",
        f"--env.task_ids={task_ids_arg(config.test_task_ids)}",
        f"--output_dir={output_dir}",
        f"--eval.batch_size={config.eval_batch_size}",
        f"--eval.n_episodes={config.n_eval_episodes}",
        f"--policy.device={config.device}",
        f"--policy.action_mode={config.action_mode}",
        f"--policy.use_proprio={cli_bool(config.policy_use_proprio)}",
        f"--policy.tokenizer_name={config.tokenizer_name}",
        f"--policy.tokenizer_max_length={config.tokenizer_max_length}",
        f"--rename_map={rename_map}",
        f"--tokenizer_task_key={config.tokenizer_task_key}",
        f"--domain_id={config.domain_id}",
        f"--proprio_dim={config.state_shape[-1] if config.state_shape else 0}",
    ]
    if config.policy_num_image_views is not None:
        args.append(f"--policy.num_image_views={config.policy_num_image_views}")
    if config.policy_empty_cameras is not None:
        args.append(f"--policy.empty_cameras={config.policy_empty_cameras}")
    return args


def parse_success_rate(stdout: str) -> float | None:
    success_rate: float | None = None
    for line in stdout.splitlines():
        if "success" not in line.lower():
            continue
        matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if not matches:
            continue
        value = float(matches[-1])
        success_rate = value / 100.0 if "%" in line and value > 1.0 else value
    return success_rate


def evaluate_suite(config: Config, adapter_checkpoint: Path, suite: str) -> float | None:
    if not config.dry_run and not adapter_checkpoint.exists():
        raise FileNotFoundError(f"CLARE adapter checkpoint does not exist: {adapter_checkpoint}")
    output_dir = config.run_output_root / "eval" / suite / datetime.now().strftime("%Y%m%d_%H%M%S")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        EVAL_CHILD_FLAG,
        *build_eval_args(config, adapter_checkpoint, suite, output_dir),
    ]
    completed = run_command(command, config, capture=True)
    if completed.stdout:
        print(completed.stdout)
    return parse_success_rate(completed.stdout or "")


def save_results(config: Config, results: dict[str, Any]) -> None:
    if config.dry_run:
        print(f"DRY_RUN enabled; not writing {config.results_file}")
        return
    config.results_file.parent.mkdir(parents=True, exist_ok=True)
    config.results_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {config.results_file}")


def save_eval_results(config: Config, results: dict[str, Any]) -> None:
    if config.dry_run:
        print(f"DRY_RUN enabled; not writing {config.eval_results_file}")
        return
    payload = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "base_model": config.base_model,
            "suites": config.suites,
            "train_task_ids": config.train_task_ids,
            "test_task_ids": config.test_task_ids,
        },
        "eval_rounds": results.get("eval_rounds", {}),
    }
    config.eval_results_file.parent.mkdir(parents=True, exist_ok=True)
    config.eval_results_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation results saved to {config.eval_results_file}")


def save_all_results(config: Config, results: dict[str, Any]) -> None:
    save_results(config, results)
    save_eval_results(config, results)


def configure_runtime_env(config: Config) -> None:
    os.environ["CUDNN_BENCHMARK"] = config.cudnn_benchmark
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.cublas_workspace_config
    os.environ["LIBERO_CONFIG_PATH"] = str(config.libero_config_path)
    os.environ["CLARE_ROOT"] = str(config.clare_root)


def run_with_logging(config: Config) -> int:
    if config.dry_run:
        print(f"DRY_RUN enabled; not writing run log {config.run_log_file}")
        return run_workflow(config)
    config.run_log_file.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with config.run_log_file.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write("\n\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"CLARE-X-VLA run started at {datetime.now().isoformat()}\n")
        log_file.write("=" * 80 + "\n")
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print(f"Full run log: {config.run_log_file}")
            return run_workflow(config)
        finally:
            print(f"Full run log saved to {config.run_log_file}")
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_workflow(config: Config) -> int:
    print_config(config)
    if not config.dry_run:
        config.output_root.mkdir(parents=True, exist_ok=True)
        config.run_output_root.mkdir(parents=True, exist_ok=True)
    if config.install_deps and not config.dry_run:
        install_dependencies(config)
    if not config.dry_run:
        validate_runtime_imports(config)
    if not config.dry_run and (config.convert_datasets or config.run_train or config.run_eval):
        ensure_libero_config(config)
    if config.copy_datasets:
        copy_datasets(config)
    if config.convert_datasets:
        convert_datasets(config)

    results: dict[str, Any] = {
        "started_at": datetime.now().isoformat(),
        "method": "CLARE-X-VLA",
        "config": safe_config_for_results(config),
        "train_runs": [],
        "eval_rounds": {},
    }

    if config.run_train:
        train_suites = config.suites if config.train_all_suites else config.suites[:1]
        previous_adapter: Path | None = None
        latest_adapter: Path | None = None
        for suite_index, suite in enumerate(train_suites):
            adapter_checkpoint = train_suite(config, suite, previous_adapter)
            latest_adapter = adapter_checkpoint
            train_run = {
                "suite": suite,
                "started_from_base_model": config.base_model,
                "previous_adapter": str(previous_adapter) if previous_adapter else None,
                "adapter_checkpoint": str(adapter_checkpoint),
                "checkpoint": str(adapter_checkpoint),
                "timestamp": datetime.now().isoformat(),
            }
            results["train_runs"].append(train_run)

            if config.run_eval:
                round_key = f"after_training_{suite}"
                seen_suites = train_suites[: suite_index + 1]
                results["eval_rounds"][round_key] = {
                    "trained_on": suite,
                    "checkpoint": str(adapter_checkpoint),
                    "timestamp": datetime.now().isoformat(),
                    "evaluations": {},
                }
                for eval_suite in seen_suites:
                    success_rate = evaluate_suite(config, adapter_checkpoint, eval_suite)
                    results["eval_rounds"][round_key]["evaluations"][eval_suite] = success_rate
                save_all_results(config, results)

            previous_adapter = adapter_checkpoint

        if latest_adapter is not None:
            print(f"Latest CLARE adapter: {latest_adapter}")

    elif config.run_eval:
        if not config.eval_policy_path:
            raise ConfigError("EVAL_POLICY_PATH must point to a CLARE adapter checkpoint when RUN_TRAIN=0")
        adapter_checkpoint = Path(config.eval_policy_path).expanduser()
        results["eval_rounds"]["standalone_eval"] = {
            "checkpoint": str(adapter_checkpoint),
            "timestamp": datetime.now().isoformat(),
            "evaluations": {},
        }
        for suite in config.suites:
            success_rate = evaluate_suite(config, adapter_checkpoint, suite)
            results["eval_rounds"]["standalone_eval"]["evaluations"][suite] = success_rate

    if config.run_train or config.run_eval:
        save_all_results(config, results)
    else:
        print("RUN_TRAIN and RUN_EVAL are disabled; imports and configuration parsed successfully.")

    print("\n========== FINAL RESULTS ==========")
    print(json.dumps(results, indent=2))
    return 0


def run_clare_train_child(train_args: list[str]) -> int:
    ensure_clare_peft_path(Path(env("CLARE_ROOT", str(Path(__file__).resolve().parent / "clare"))).expanduser())
    patch_clare_runtime()

    import logging
    from contextlib import nullcontext
    from dataclasses import dataclass, field
    from pathlib import Path as ChildPath
    from typing import Literal

    import torch
    from torch.amp.grad_scaler import GradScaler
    from torch.optim import Optimizer

    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.datasets.utils import cycle
    from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
    from lerobot.optim.schedulers import LRScheduler, LRSchedulerConfig
    from lerobot.policies.factory import make_policy
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.policies.utils import get_device_from_parameters
    from lerobot.utils.random_utils import set_seed
    from peft import PeftConfig, PeftModel, get_peft_model
    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
    from transformers import AutoTokenizer

    get_safe_torch_device, init_logging = import_lerobot_runtime_helpers()
    try:
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
    except ImportError:
        OBS_LANGUAGE_TOKENS = "observation.language.tokens"

    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
    OBS_STATE = "observation.state"

    class PeftWrapperPolicy(torch.nn.Module):
        policy: PreTrainedPolicy

        def __init__(self, policy: PreTrainedPolicy):
            super().__init__()
            self.policy = policy

    @dataclass
    class CLARETrainPipelineConfig(TrainPipelineConfig):
        peft_cfg_path: ChildPath | None = None
        peft_weight_path: ChildPath | None = None
        detect_distribution_shift_steps: int = 20
        detect_distribution_shift_batch_size: int = 16
        detect_distribution_shift_num_workers: int = 0
        detect_distribution_shift_log_freq: int = 10
        train_discriminators_steps: int = 200
        train_discriminators_batch_size: int = 16
        train_discriminators_num_workers: int = 0
        train_discriminators_log_freq: int = 10
        train_discriminators_save_freq: int = 200
        train_discriminators_eval_freq: int = 0
        train_discriminator_optimizer: OptimizerConfig = field(
            default_factory=lambda: AdamWConfig(
                lr=5e-4,
                weight_decay=0.01,
                grad_clip_norm=10.0,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        )
        train_discriminator_lr_scheduler: LRSchedulerConfig | None = None
        maximum_expand: int = 10000
        expand_threshold: float = 1.0
        at_least_expand: Literal["shallowest", "deepest"] = "shallowest"
        tokenizer_task_key: str = "task"
        domain_id: int = 3

        def __post_init__(self) -> None:
            if not (self.peft_cfg_path or self.peft_weight_path):
                raise ValueError("One of peft_cfg_path or peft_weight_path must be specified")

    update_dataclass_type_hints(
        CLARETrainPipelineConfig,
        {
            "peft_cfg_path": ChildPath | None,
            "peft_weight_path": ChildPath | None,
            "train_discriminator_optimizer": OptimizerConfig,
            "train_discriminator_lr_scheduler": LRSchedulerConfig | None,
            "at_least_expand": Literal["shallowest", "deepest"],
            "tokenizer_task_key": str,
            "domain_id": int,
        },
    )

    def set_peft_module_train(peft_modules: list[Any], train: bool = True) -> list[Any]:
        if not peft_modules:
            raise RuntimeError("No CLARE adapter modules were injected.")
        prefix = PEFT_TYPE_TO_PREFIX_MAPPING[peft_modules[0].peft_config.peft_type]
        for peft_module in peft_modules:
            for name, module in peft_module.named_modules():
                if prefix in name or name == "":
                    module.train(train)
                if "base_layer" in name:
                    module.train(False)
        return peft_modules

    def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
        return batch

    def infer_batch_size(batch: dict[str, Any]) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        return 1

    def normalize_task_texts(raw_tasks: Any, batch_size: int) -> list[str]:
        if raw_tasks is None:
            tasks = [""]
        elif isinstance(raw_tasks, str):
            tasks = [raw_tasks]
        elif isinstance(raw_tasks, torch.Tensor):
            tasks = [str(item) for item in raw_tasks.detach().cpu().tolist()]
        elif isinstance(raw_tasks, (list, tuple)):
            tasks = [str(item) for item in raw_tasks]
        else:
            tasks = [str(raw_tasks)]
        if len(tasks) == 1 and batch_size > 1:
            tasks = tasks * batch_size
        if len(tasks) != batch_size:
            tasks = (tasks + [tasks[-1] if tasks else ""] * batch_size)[:batch_size]
        return tasks

    def add_xvla_language_and_domain(
        batch: dict[str, Any],
        cfg: CLARETrainPipelineConfig,
        tokenizer: Any,
    ) -> dict[str, Any]:
        batch_size = infer_batch_size(batch)
        if OBS_LANGUAGE_TOKENS not in batch:
            tasks = normalize_task_texts(batch.get(cfg.tokenizer_task_key, batch.get("task")), batch_size)
            tokenizer.padding_side = getattr(cfg.policy, "tokenizer_padding_side", "right")
            encoded = tokenizer(
                tasks,
                padding=getattr(cfg.policy, "pad_language_to", "max_length"),
                truncation=True,
                max_length=int(getattr(cfg.policy, "tokenizer_max_length", 50)),
                return_tensors="pt",
            )
            batch[OBS_LANGUAGE_TOKENS] = encoded["input_ids"]
            if "attention_mask" in encoded:
                batch[OBS_LANGUAGE_ATTENTION_MASK] = encoded["attention_mask"]
        if "domain_id" not in batch:
            batch["domain_id"] = torch.full((batch_size,), int(cfg.domain_id), dtype=torch.long)
        return batch

    def prepare_batch(
        batch: dict[str, Any],
        cfg: CLARETrainPipelineConfig,
        device: torch.device,
        rename_map: dict[str, str],
        tokenizer: Any,
    ) -> dict[str, Any]:
        batch = apply_rename_map_to_batch(batch, rename_map)
        batch = add_xvla_language_and_domain(batch, cfg, tokenizer)
        return move_batch_to_device(batch, device)

    def detect_distribution_shift(
        cfg: CLARETrainPipelineConfig,
        policy: PreTrainedPolicy,
        peft_modules: list[Any],
        dataset: Any,
        device: torch.device,
        tokenizer: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if cfg.detect_distribution_shift_steps <= 0:
            raise RuntimeError("detect_distribution_shift_steps must be > 0 for later CLARE stages")
        for peft_module in peft_modules:
            peft_module.track_z_score(True)
        detect_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.detect_distribution_shift_num_workers,
            batch_size=cfg.detect_distribution_shift_batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        detect_iter = cycle(detect_loader)
        policy.eval()
        z_scores_sum: dict[str, list[float]] = {}
        losses_sum: dict[str, list[float]] = {}
        step = 0
        for _ in range(cfg.detect_distribution_shift_steps):
            batch = prepare_batch(next(detect_iter), cfg, device, getattr(cfg, "rename_map", {}) or {}, tokenizer)
            with torch.inference_mode():
                policy.forward(batch)
            for peft_module in peft_modules:
                key = f"{peft_module.layer_name}.{peft_module.layer_id}"
                z_scores_sum.setdefault(key, [0.0] * peft_module.num_discriminators)
                losses_sum.setdefault(key, [0.0] * peft_module.num_discriminators)
                for discriminator_id in range(peft_module.num_discriminators):
                    info = peft_module.info_dicts[f"discriminator_{discriminator_id}"]
                    z_scores_sum[key][discriminator_id] += float(info["z_score"].mean().item())
                    losses_sum[key][discriminator_id] += float(info["loss"].mean().item())
            step += 1
        z_scores_mean: dict[str, torch.Tensor] = {}
        losses_mean: dict[str, torch.Tensor] = {}
        for peft_module in peft_modules:
            peft_module.track_z_score(False)
            key = f"{peft_module.layer_name}.{peft_module.layer_id}"
            z_scores_mean[key] = torch.tensor(z_scores_sum[key], device="cpu") / step
            losses_mean[key] = torch.tensor(losses_sum[key], device="cpu") / step
            logging.info("Distribution shift %s z=%s loss=%s", key, z_scores_mean[key], losses_mean[key])
        return z_scores_mean, losses_mean

    def build_loader(cfg: CLARETrainPipelineConfig, dataset: Any, device: torch.device) -> Any:
        if hasattr(cfg.policy, "drop_n_last_frames"):
            sampler = EpisodeAwareSampler(
                dataset.episode_data_index,
                drop_n_last_frames=cfg.policy.drop_n_last_frames,
                shuffle=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

    def optimizer_params(optimizer: Optimizer) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for group in optimizer.param_groups:
            params.extend(group["params"])
        return params

    def named_parameters_for_optimizer(
        root_module: torch.nn.Module,
        parameters: list[torch.nn.Parameter],
        fallback_prefix: str,
    ) -> dict[str, torch.nn.Parameter]:
        target_ids = {id(parameter) for parameter in parameters}
        found_ids: set[int] = set()
        named_params: dict[str, torch.nn.Parameter] = {}
        for name, parameter in root_module.named_parameters():
            parameter_id = id(parameter)
            if parameter_id in target_ids and parameter_id not in found_ids and parameter.requires_grad:
                named_params[name] = parameter
                found_ids.add(parameter_id)
        for index, parameter in enumerate(parameters):
            parameter_id = id(parameter)
            if parameter_id in found_ids or not parameter.requires_grad:
                continue
            named_params[f"{fallback_prefix}.{index}"] = parameter
            found_ids.add(parameter_id)
        if not named_params:
            raise RuntimeError(f"No trainable named parameters were found for {fallback_prefix}.")
        return named_params

    def build_optimizer_compatible(
        optimizer_config: OptimizerConfig,
        named_params: dict[str, torch.nn.Parameter],
        label: str,
    ) -> Optimizer:
        try:
            optimizer = optimizer_config.build(named_params)
        except Exception as first_exc:
            try:
                optimizer = optimizer_config.build(list(named_params.values()))
            except Exception:
                raise first_exc
            logging.info(
                "Built %s optimizer with parameter list fallback for optimizer config %s",
                label,
                type(optimizer_config).__name__,
            )
        if isinstance(optimizer, dict):
            raise RuntimeError(
                f"{label} optimizer config returned multiple optimizers, which this CLARE loop does not support."
            )
        logging.info("Built %s optimizer over %s named parameters", label, len(named_params))
        return optimizer

    def update_policy(
        policy: PreTrainedPolicy,
        peft_modules: list[Any],
        batch: dict[str, Any],
        optimizer: Optimizer,
        grad_clip_norm: float,
        grad_scaler: GradScaler,
        lr_scheduler: LRScheduler | None,
        use_amp: bool,
    ) -> float:
        device = get_device_from_parameters(policy)
        set_peft_module_train(peft_modules, True)
        with torch.autocast(device_type=device.type) if use_amp else nullcontext():
            policy_loss, _ = policy.forward(batch)
            if peft_modules[0]._train_discriminator:
                losses = []
                for peft_module in peft_modules:
                    discriminator_id = peft_module._forwarded_discriminator_id
                    info = peft_module.info_dicts[f"discriminator_{discriminator_id}"]
                    losses.append(info["loss"].mean())
                loss = sum(losses)
            else:
                loss = policy_loss
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(optimizer_params(optimizer), grad_clip_norm, error_if_nonfinite=False)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if hasattr(policy, "update"):
            policy.update()
        return float(loss.item())

    def save_adapter(peft_policy: PeftModel, cfg: CLARETrainPipelineConfig, step: int) -> None:
        adapter_dir = ChildPath(cfg.output_dir) / "checkpoints" / "last" / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_policy.save_pretrained(str(adapter_dir))
        (adapter_dir / "clare_xvla_training_state.json").write_text(
            json.dumps({"step": step, "saved_at": datetime.now().isoformat()}, indent=2) + "\n",
            encoding="utf-8",
        )
        logging.info("Saved CLARE adapter to %s", adapter_dir)

    def train(cfg: CLARETrainPipelineConfig) -> None:
        cfg.validate()
        if cfg.seed is not None:
            set_seed(cfg.seed)
        device = get_safe_torch_device(cfg.policy.device, log=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        tokenizer_name = getattr(cfg.policy, "tokenizer_name", "facebook/bart-large")
        logging.info("Loading tokenizer for X-VLA language inputs: %s", tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        logging.info("Creating X-VLA policy")
        policy_meta = make_policy_compatible_metadata(dataset.meta, cfg.policy, getattr(cfg, "rename_map", {}) or {})
        policy = make_policy(cfg=cfg.policy, ds_meta=policy_meta)
        policy.eval()
        wrapper = PeftWrapperPolicy(policy=policy)

        if cfg.peft_weight_path:
            peft_policy = PeftModel.from_pretrained(
                wrapper,
                cfg.peft_weight_path,
                is_trainable=True,
                autocast_adapter_dtype=False,
            )
            peft_config = peft_policy.peft_config["default"]
        else:
            peft_cfg = PeftConfig.from_pretrained(cfg.peft_cfg_path)
            peft_cfg.inference_mode = False
            matched_targets = ensure_clare_targets_match_wrapped_model(peft_cfg, wrapper)
            logging.info("CLARE target patterns matched %s wrapped policy modules", matched_targets)
            peft_policy = get_peft_model(wrapper, peft_cfg)
            peft_config = peft_policy.peft_config["default"]

        peft_modules = list(peft_policy.base_model.adapter_layers)
        if not peft_modules:
            raise RuntimeError("CLARE injected zero adapter layers. Check CLARE target modules.")

        adapter_params: list[torch.nn.Parameter] = []
        discriminator_params: list[torch.nn.Parameter] = []
        new_task_id = peft_config.num_learned_task

        if new_task_id == 0:
            for peft_module in peft_modules:
                adapter_param, discriminator_param = peft_module.add_adapter_and_discriminator(new_task_id)
                adapter_params += adapter_param
                discriminator_params += discriminator_param
                peft_module._forwarded_adapter_id = peft_module.num_adapters - 1
                peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1
                peft_config.structure[f"{peft_module.layer_name}.{peft_module.layer_id}"] = [
                    peft_module.num_adapters,
                    peft_module.num_discriminators,
                ]
        else:
            z_scores_mean, losses_mean = detect_distribution_shift(cfg, policy, peft_modules, dataset, device, tokenizer)
            only_forward_ids: list[int] = []
            to_expand_or_not: list[bool] = []
            for peft_module in peft_modules:
                key = f"{peft_module.layer_name}.{peft_module.layer_id}"
                closest_discriminator_id = torch.argmin(losses_mean[key]).item()
                connected_adapter_id = peft_module.get_adapter_id_by_discriminator_id(closest_discriminator_id)
                only_forward_ids.append(connected_adapter_id)
                expand = bool(torch.all(z_scores_mean[key] > cfg.expand_threshold).item())
                if expand and sum(to_expand_or_not) >= cfg.maximum_expand:
                    expand = False
                to_expand_or_not.append(expand)
            if sum(to_expand_or_not) == 0:
                if cfg.at_least_expand == "shallowest":
                    to_expand_or_not[0] = True
                    only_forward_ids[0] = -1
                else:
                    to_expand_or_not[-1] = True
                    only_forward_ids[-1] = -1
            for peft_module, should_expand, only_forward_id in zip(
                peft_modules, to_expand_or_not, only_forward_ids, strict=False
            ):
                key = f"{peft_module.layer_name}.{peft_module.layer_id}"
                if should_expand:
                    adapter_param, discriminator_param = peft_module.add_adapter_and_discriminator(new_task_id)
                    adapter_params += adapter_param
                    discriminator_params += discriminator_param
                    peft_module._forwarded_adapter_id = peft_module.num_adapters - 1
                    peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1
                else:
                    discriminator_param = peft_module.add_discriminator(only_forward_id, new_task_id)
                    discriminator_params += discriminator_param
                    peft_module._forwarded_adapter_id = only_forward_id
                    peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1
                peft_module._active_task = new_task_id
                peft_config.structure[key] = [peft_module.num_adapters, peft_module.num_discriminators]

        peft_config.num_learned_task += 1
        if not adapter_params:
            raise RuntimeError("No adapter parameters were selected for training.")
        if not discriminator_params:
            raise RuntimeError("No discriminator parameters were selected for training.")

        adapter_named_params = named_parameters_for_optimizer(peft_policy, adapter_params, "clare_adapter")
        discriminator_named_params = named_parameters_for_optimizer(
            peft_policy,
            discriminator_params,
            "clare_discriminator",
        )

        adapter_optimizer = build_optimizer_compatible(cfg.optimizer, adapter_named_params, "adapter")
        adapter_scheduler = None
        if cfg.scheduler:
            adapter_scheduler = cfg.scheduler.build(adapter_optimizer, cfg.steps)
        discriminator_optimizer = build_optimizer_compatible(
            cfg.train_discriminator_optimizer,
            discriminator_named_params,
            "discriminator",
        )
        discriminator_scheduler = None
        if cfg.train_discriminator_lr_scheduler:
            discriminator_scheduler = cfg.train_discriminator_lr_scheduler.build(
                discriminator_optimizer, cfg.train_discriminators_steps
            )

        loader = build_loader(cfg, dataset, device)
        iterator = cycle(loader)
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
        step = 0

        logging.info("Training CLARE functional adapters for %s steps", cfg.steps)
        for peft_module in peft_modules:
            peft_module.train_discriminator(False)
            peft_module.update_stats(False)
        for _ in range(cfg.steps):
            batch = prepare_batch(next(iterator), cfg, device, getattr(cfg, "rename_map", {}) or {}, tokenizer)
            loss = update_policy(
                policy,
                peft_modules,
                batch,
                adapter_optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler,
                adapter_scheduler,
                cfg.policy.use_amp,
            )
            step += 1
            if cfg.log_freq > 0 and step % cfg.log_freq == 0:
                logging.info("adapter step=%s loss=%.6f", step, loss)

        logging.info("Training CLARE discriminators for %s steps", cfg.train_discriminators_steps)
        for peft_module in peft_modules:
            peft_module.train_discriminator(True)
            peft_module.update_stats(True)
        for _ in range(cfg.train_discriminators_steps):
            batch = prepare_batch(next(iterator), cfg, device, getattr(cfg, "rename_map", {}) or {}, tokenizer)
            loss = update_policy(
                policy,
                peft_modules,
                batch,
                discriminator_optimizer,
                cfg.train_discriminator_optimizer.grad_clip_norm,
                grad_scaler,
                discriminator_scheduler,
                cfg.policy.use_amp,
            )
            step += 1
            rel_step = step - cfg.steps
            if cfg.train_discriminators_log_freq > 0 and rel_step % cfg.train_discriminators_log_freq == 0:
                logging.info("discriminator step=%s loss=%.6f", rel_step, loss)

        save_adapter(peft_policy, cfg, step)

    sys.argv = [sys.argv[0], *train_args]
    init_logging()
    train.__annotations__["cfg"] = CLARETrainPipelineConfig
    train = parser.wrap()(train)
    train()
    return 0


def run_clare_eval_child(eval_args: list[str]) -> int:
    ensure_clare_peft_path(Path(env("CLARE_ROOT", str(Path(__file__).resolve().parent / "clare"))).expanduser())
    patch_clare_runtime()

    import json as child_json
    import logging
    import time
    from contextlib import nullcontext
    from dataclasses import dataclass, field
    from pathlib import Path as ChildPath

    import numpy as np
    import torch

    from lerobot.configs import parser
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.envs.factory import make_env
    try:
        from lerobot.envs.utils import check_env_attributes_and_types, preprocess_observation
    except ImportError:
        check_env_attributes_and_types = None

        def preprocess_observation(observation: dict[str, Any]) -> dict[str, Any]:
            for key, value in list(observation.items()):
                if isinstance(value, np.ndarray):
                    observation[key] = torch.from_numpy(value)
            return observation

    from lerobot.policies.factory import make_policy
    from lerobot.policies.pretrained import PreTrainedPolicy
    try:
        from lerobot.policies.utils import get_device_from_parameters
    except ImportError:
        def get_device_from_parameters(module: torch.nn.Module) -> torch.device:
            return next(module.parameters()).device

    from lerobot.utils.random_utils import set_seed
    from peft import PeftModel
    from transformers import AutoTokenizer

    get_safe_torch_device, init_logging = import_lerobot_runtime_helpers()
    try:
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
    except ImportError:
        OBS_LANGUAGE_TOKENS = "observation.language.tokens"

    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
    OBS_STATE = "observation.state"

    class PeftWrapperPolicy(torch.nn.Module):
        policy: PreTrainedPolicy

        def __init__(self, policy: PreTrainedPolicy):
            super().__init__()
            self.policy = policy

    @dataclass
    class CLAREEvalPipelineConfig(EvalPipelineConfig):
        peft_weight_path: ChildPath | None = None
        dataset: DatasetConfig | None = None
        rename_map: dict[str, str] = field(default_factory=dict)
        tokenizer_task_key: str = "task"
        domain_id: int = 3
        proprio_dim: int = 8

    update_dataclass_type_hints(
        CLAREEvalPipelineConfig,
        {
            "peft_weight_path": ChildPath | None,
            "dataset": DatasetConfig | None,
            "rename_map": dict[str, str],
            "tokenizer_task_key": str,
            "domain_id": int,
            "proprio_dim": int,
        },
    )

    def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
        return batch

    def infer_batch_size(batch: dict[str, Any]) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        return 1

    def default_task_text(cfg: CLAREEvalPipelineConfig) -> str:
        task = getattr(cfg.env, "task", "")
        return str(task if task is not None else "")

    def normalize_task_texts(raw_tasks: Any, batch_size: int, fallback: str) -> list[str]:
        if raw_tasks is None:
            tasks = [fallback]
        elif isinstance(raw_tasks, str):
            tasks = [raw_tasks]
        elif isinstance(raw_tasks, torch.Tensor):
            tasks = [str(item) for item in raw_tasks.detach().cpu().tolist()]
        elif isinstance(raw_tasks, (list, tuple)):
            tasks = [str(item) for item in raw_tasks]
        else:
            tasks = [str(raw_tasks)]
        if len(tasks) == 1 and batch_size > 1:
            tasks = tasks * batch_size
        if len(tasks) != batch_size:
            tasks = (tasks + [tasks[-1] if tasks else fallback] * batch_size)[:batch_size]
        return tasks

    def add_xvla_language_and_domain(
        batch: dict[str, Any],
        cfg: CLAREEvalPipelineConfig,
        tokenizer: Any,
    ) -> dict[str, Any]:
        batch_size = infer_batch_size(batch)
        if OBS_LANGUAGE_TOKENS not in batch:
            tasks = normalize_task_texts(
                batch.get(cfg.tokenizer_task_key, batch.get("task")),
                batch_size,
                default_task_text(cfg),
            )
            tokenizer.padding_side = getattr(cfg.policy, "tokenizer_padding_side", "right")
            encoded = tokenizer(
                tasks,
                padding=getattr(cfg.policy, "pad_language_to", "max_length"),
                truncation=True,
                max_length=int(getattr(cfg.policy, "tokenizer_max_length", 50)),
                return_tensors="pt",
            )
            batch[OBS_LANGUAGE_TOKENS] = encoded["input_ids"]
            if "attention_mask" in encoded:
                batch[OBS_LANGUAGE_ATTENTION_MASK] = encoded["attention_mask"]
        if "domain_id" not in batch:
            batch["domain_id"] = torch.full((batch_size,), int(cfg.domain_id), dtype=torch.long)
        return batch

    def as_state_tensor(value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, (list, tuple)):
            return torch.as_tensor(value)
        return None

    def ensure_xvla_proprio_state(batch: dict[str, Any], cfg: CLAREEvalPipelineConfig) -> dict[str, Any]:
        proprio_dim = int(getattr(cfg, "proprio_dim", 0) or 0)
        if proprio_dim <= 0:
            return batch
        batch_size = infer_batch_size(batch)
        state = as_state_tensor(batch.get(OBS_STATE))
        if state is None:
            batch[OBS_STATE] = torch.zeros((batch_size, proprio_dim), dtype=torch.float32)
            return batch

        if not torch.is_floating_point(state):
            state = state.float()
        if state.ndim == 0:
            state = state.reshape(1, 1)
        elif state.ndim == 1:
            if state.numel() == proprio_dim:
                state = state.reshape(1, proprio_dim)
            elif state.numel() == batch_size * proprio_dim:
                state = state.reshape(batch_size, proprio_dim)
            else:
                state = state.reshape(batch_size, -1) if state.numel() % max(batch_size, 1) == 0 else state.reshape(1, -1)

        current_dim = int(state.shape[-1]) if state.ndim > 0 else 0
        if current_dim == proprio_dim:
            batch[OBS_STATE] = state
            return batch

        target_shape = (*state.shape[:-1], proprio_dim)
        if current_dim == 0:
            batch[OBS_STATE] = torch.zeros(target_shape, dtype=state.dtype, device=state.device)
        elif current_dim < proprio_dim:
            pad_shape = (*state.shape[:-1], proprio_dim - current_dim)
            padding = torch.zeros(pad_shape, dtype=state.dtype, device=state.device)
            batch[OBS_STATE] = torch.cat([state, padding], dim=-1)
        else:
            batch[OBS_STATE] = state[..., :proprio_dim]
        return batch

    def prepare_eval_batch(
        batch: Any,
        cfg: CLAREEvalPipelineConfig,
        device: torch.device,
        tokenizer: Any,
    ) -> Any:
        if not isinstance(batch, dict):
            return batch
        batch = apply_rename_map_to_batch(batch, cfg.rename_map or {})
        batch = add_xvla_language_and_domain(batch, cfg, tokenizer)
        batch = ensure_xvla_proprio_state(batch, cfg)
        return move_batch_to_device(batch, device)

    def success_list_from_info(info: dict[str, Any], num_envs: int) -> list[bool]:
        if "final_info" in info:
            successes = []
            for final_info in info["final_info"]:
                if final_info is None:
                    successes.append(False)
                else:
                    successes.append(bool(final_info.get("is_success", False)))
            return successes
        if "is_success" in info:
            value = info["is_success"]
            if isinstance(value, torch.Tensor):
                return [bool(item) for item in value.detach().cpu().flatten().tolist()]
            if isinstance(value, np.ndarray):
                return [bool(item) for item in value.flatten().tolist()]
            if isinstance(value, (list, tuple)):
                return [bool(item) for item in value]
            return [bool(value)] * num_envs
        return [False] * num_envs

    def infer_action_dim_from_space(space: Any) -> int | None:
        shape = getattr(space, "shape", None)
        if shape:
            return int(shape[-1])
        return None

    def infer_env_action_dim(env_object: Any, seen: set[int] | None = None) -> int | None:
        if seen is None:
            seen = set()
        object_id = id(env_object)
        if object_id in seen:
            return None
        seen.add(object_id)

        for attr_name in ("single_action_space", "action_space"):
            dim = infer_action_dim_from_space(getattr(env_object, attr_name, None))
            if dim is not None:
                return dim

        action_dim = getattr(env_object, "action_dim", None)
        if isinstance(action_dim, int) and action_dim > 0:
            return action_dim

        for child in getattr(env_object, "envs", []) or []:
            dim = infer_env_action_dim(child, seen)
            if dim is not None:
                return dim

        for attr_name in ("unwrapped", "env", "_env"):
            child = getattr(env_object, attr_name, None)
            if child is not None:
                dim = infer_env_action_dim(child, seen)
                if dim is not None:
                    return dim
        return None

    def adapt_action_to_env(action: np.ndarray, env_action_dim: int | None) -> np.ndarray:
        if env_action_dim is None or action.shape[-1] == env_action_dim:
            return action
        if action.shape[-1] > env_action_dim:
            return action[..., :env_action_dim]

        pad_shape = (*action.shape[:-1], env_action_dim - action.shape[-1])
        padding = np.zeros(pad_shape, dtype=action.dtype)
        return np.concatenate([action, padding], axis=-1)

    def local_rollout(env: Any, policy: PreTrainedPolicy, seeds: list[int] | None = None) -> dict[str, torch.Tensor]:
        device = get_device_from_parameters(policy)
        env_action_dim = infer_env_action_dim(env)
        if env_action_dim is None:
            logging.warning("Could not infer env action dimension; using policy action shape directly.")
        policy.reset()
        observation, _info = env.reset(seed=seeds)
        if check_env_attributes_and_types is not None:
            check_env_attributes_and_types(env)

        done = np.array([False] * env.num_envs)
        max_steps = int(env.call("_max_episode_steps")[0])
        all_actions: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        all_successes: list[torch.Tensor] = []
        all_dones: list[torch.Tensor] = []

        for _step in range(max_steps):
            if np.all(done):
                break
            observation = preprocess_observation(observation)
            for key in observation:
                if isinstance(observation[key], torch.Tensor):
                    observation[key] = observation[key].to(device, non_blocking=device.type == "cuda")

            with torch.inference_mode():
                action = policy.select_action(observation)

            action_np = action.detach().to("cpu").numpy()
            action_np = adapt_action_to_env(action_np, env_action_dim)
            observation, reward, terminated, truncated, info = env.step(action_np)
            done = np.asarray(terminated) | np.asarray(truncated) | done
            successes = success_list_from_info(info, env.num_envs)

            all_actions.append(torch.from_numpy(action_np))
            all_rewards.append(torch.as_tensor(reward))
            all_successes.append(torch.tensor(successes))
            all_dones.append(torch.from_numpy(done.copy()))

        if not all_actions:
            raise RuntimeError("Evaluation rollout finished before any action was produced.")

        return {
            "action": torch.stack(all_actions, dim=1),
            "reward": torch.stack(all_rewards, dim=1),
            "success": torch.stack(all_successes, dim=1),
            "done": torch.stack(all_dones, dim=1),
        }

    def eval_policy(
        env: Any,
        policy: PreTrainedPolicy,
        n_episodes: int,
        max_episodes_rendered: int = 0,
        videos_dir: ChildPath | None = None,
        start_seed: int | None = None,
    ) -> dict[str, Any]:
        if max_episodes_rendered > 0:
            logging.warning("Video rendering is not supported by the local CLARE-X-VLA eval loop; skipping videos.")
        start = time.time()
        policy.eval()
        n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)
        sum_rewards: list[float] = []
        max_rewards: list[float] = []
        all_successes: list[bool] = []
        all_seeds: list[int | None] = []

        for batch_ix in range(n_batches):
            if start_seed is None:
                seeds = None
            else:
                seeds = list(range(start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)))
            rollout_data = local_rollout(env, policy, seeds=seeds)
            n_steps = rollout_data["done"].shape[1]
            done_indices = torch.argmax(rollout_data["done"].to(torch.int64), dim=1)
            mask = torch.arange(n_steps).unsqueeze(0) <= (done_indices + 1).unsqueeze(1)

            batch_sum_rewards = (rollout_data["reward"] * mask).sum(dim=1)
            batch_max_rewards = rollout_data["reward"].masked_fill(~mask, float("-inf")).max(dim=1).values
            batch_successes = (rollout_data["success"] & mask).any(dim=1)

            sum_rewards.extend(float(item) for item in batch_sum_rewards.tolist())
            max_rewards.extend(float(item) for item in batch_max_rewards.tolist())
            all_successes.extend(bool(item) for item in batch_successes.tolist())
            all_seeds.extend(seeds if seeds is not None else [None] * env.num_envs)

        elapsed = time.time() - start
        return {
            "per_episode": [
                {
                    "episode_ix": i,
                    "sum_reward": sum_reward,
                    "max_reward": max_reward,
                    "success": success,
                    "seed": seed,
                }
                for i, (sum_reward, max_reward, success, seed) in enumerate(
                    zip(
                        sum_rewards[:n_episodes],
                        max_rewards[:n_episodes],
                        all_successes[:n_episodes],
                        all_seeds[:n_episodes],
                        strict=True,
                    )
                )
            ],
            "aggregated": {
                "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
                "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
                "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
                "eval_s": elapsed,
                "eval_ep_s": elapsed / n_episodes,
            },
        }

    def collect_vector_envs(env_object: Any, prefix: str = "env") -> list[tuple[str, Any]]:
        if hasattr(env_object, "num_envs"):
            return [(prefix, env_object)]
        if not isinstance(env_object, dict):
            return []
        envs: list[tuple[str, Any]] = []
        for key, value in env_object.items():
            child_prefix = f"{prefix}.{key}"
            envs.extend(collect_vector_envs(value, child_prefix))
        return envs

    def merge_eval_infos(infos: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        per_episode: list[dict[str, Any]] = []
        per_env: dict[str, Any] = {}
        elapsed = 0.0
        for env_name, info in infos:
            per_env[env_name] = info.get("aggregated", {})
            elapsed += float(info.get("aggregated", {}).get("eval_s", 0.0))
            for episode in info.get("per_episode", []):
                merged_episode = dict(episode)
                merged_episode["env_name"] = env_name
                merged_episode["episode_ix"] = len(per_episode)
                per_episode.append(merged_episode)

        if not per_episode:
            raise RuntimeError("Evaluation produced no episodes.")

        sum_rewards = [float(ep["sum_reward"]) for ep in per_episode]
        max_rewards = [float(ep["max_reward"]) for ep in per_episode]
        successes = [bool(ep["success"]) for ep in per_episode]
        return {
            "per_episode": per_episode,
            "per_env": per_env,
            "aggregated": {
                "avg_sum_reward": float(np.nanmean(sum_rewards)),
                "avg_max_reward": float(np.nanmean(max_rewards)),
                "pc_success": float(np.nanmean(successes) * 100),
                "eval_s": elapsed,
                "eval_ep_s": elapsed / len(per_episode),
            },
        }

    def close_vector_envs(env_items: list[tuple[str, Any]]) -> None:
        closed: set[int] = set()
        for _env_name, env in env_items:
            env_id = id(env)
            if env_id in closed:
                continue
            closed.add(env_id)
            try:
                env.close()
            except Exception as exc:
                logging.warning("Failed to close eval env cleanly: %s", exc)

    def eval_main(cfg: CLAREEvalPipelineConfig) -> None:
        if cfg.peft_weight_path is None:
            raise ValueError("peft_weight_path is required for CLARE-X-VLA eval")
        device = get_safe_torch_device(cfg.policy.device, log=True)
        set_seed(cfg.seed)
        use_async_envs = bool(getattr(cfg.eval, "use_async_envs", False))
        max_episodes_rendered = int(getattr(cfg.eval, "max_episodes_rendered", 0) or 0)
        env_object = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=use_async_envs)
        env_items = collect_vector_envs(env_object)
        if not env_items:
            if isinstance(env_object, dict):
                raise RuntimeError(
                    f"make_env returned a dict, but no vector env was found inside. Keys: {list(env_object.keys())}"
                )
            raise RuntimeError(f"make_env returned unsupported object type: {type(env_object).__name__}")
        ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision)
        policy_meta = make_policy_compatible_metadata(ds_meta, cfg.policy, cfg.rename_map or {})
        policy = make_policy(cfg=cfg.policy, ds_meta=policy_meta)
        wrapper = PeftWrapperPolicy(policy=policy)
        peft_policy = PeftModel.from_pretrained(
            wrapper,
            cfg.peft_weight_path,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
        if not peft_policy.base_model.adapter_layers:
            raise RuntimeError("Loaded CLARE adapter has no adapter layers")
        tokenizer_name = getattr(cfg.policy, "tokenizer_name", None) or getattr(cfg.policy, "pretrained_model_name_or_path", None)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        original_select_action = policy.select_action
        original_forward = policy.forward

        def select_action_with_xvla_inputs(batch: Any, *args: Any, **kwargs: Any) -> Any:
            return original_select_action(prepare_eval_batch(batch, cfg, device, tokenizer), *args, **kwargs)

        def forward_with_xvla_inputs(batch: Any, *args: Any, **kwargs: Any) -> Any:
            return original_forward(prepare_eval_batch(batch, cfg, device, tokenizer), *args, **kwargs)

        policy.select_action = select_action_with_xvla_inputs
        policy.forward = forward_with_xvla_inputs
        policy.eval()
        infos: list[tuple[str, dict[str, Any]]] = []
        try:
            with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
                for env_name, env in env_items:
                    logging.info("Evaluating %s with %s parallel env(s)", env_name, env.num_envs)
                    info = eval_policy(
                        env,
                        policy,
                        cfg.eval.n_episodes,
                        max_episodes_rendered=max_episodes_rendered,
                        videos_dir=ChildPath(cfg.output_dir) / "videos",
                        start_seed=cfg.seed,
                    )
                    infos.append((env_name, info))
        finally:
            close_vector_envs(env_items)
        info = infos[0][1] if len(infos) == 1 else merge_eval_infos(infos)
        ChildPath(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        with (ChildPath(cfg.output_dir) / "eval_info.json").open("w", encoding="utf-8") as handle:
            child_json.dump(info, handle, indent=2)
            handle.write("\n")
        pc_success = float(info["aggregated"]["pc_success"])
        print(child_json.dumps(info["aggregated"], indent=2))
        print(f"success: {pc_success:.3f}%")
        logging.info("End of CLARE-X-VLA eval")

    sys.argv = [sys.argv[0], *eval_args]
    init_logging()
    eval_main.__annotations__["cfg"] = CLAREEvalPipelineConfig
    eval_main = parser.wrap()(eval_main)
    eval_main()
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == TRAIN_CHILD_FLAG:
        return run_clare_train_child(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == EVAL_CHILD_FLAG:
        return run_clare_eval_child(sys.argv[2:])
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    configure_runtime_env(config)
    return run_with_logging(config)


if __name__ == "__main__":
    raise SystemExit(main())
