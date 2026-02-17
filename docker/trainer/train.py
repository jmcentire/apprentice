"""Apprentice Trainer — runs inside the K8s training Job container.

Performs QLoRA fine-tuning via Unsloth, merges LoRA adapters, converts to
GGUF, and uploads the artifact + metrics to GCS.

All configuration is passed via environment variables set by the
KubernetesLoRABackend when creating the K8s Job.

Production features:
  - GCS operations with exponential backoff retry
  - /tmp cleanup before and after training
  - Validation of required tools (convert script, llama-quantize)
  - Structured error exit codes (1=general, 2=GCS, 3=training, 4=conversion)
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── Exit codes ──────────────────────────────────────────────────────────────
EXIT_GCS_ERROR = 2
EXIT_TRAINING_ERROR = 3
EXIT_CONVERSION_ERROR = 4


# ── Retry helper ────────────────────────────────────────────────────────────


def _retry(fn, retries=3, base_delay=2.0, operation="operation"):
    """Retry a callable with exponential backoff. Returns the result or re-raises."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(
                    f"[trainer] {operation} failed (attempt {attempt}/{retries}): {e}. "
                    f"Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(
                    f"[trainer] {operation} failed after {retries} attempts: {e}",
                    file=sys.stderr,
                )
    raise last_exc


# ── Cleanup helper ──────────────────────────────────────────────────────────


def _cleanup_work_dir(work_dir: Path) -> None:
    """Remove the work directory if it exists."""
    if work_dir.exists():
        try:
            shutil.rmtree(str(work_dir))
            print(f"[trainer] Cleaned up {work_dir}")
        except Exception as e:
            print(f"[trainer] Warning: cleanup of {work_dir} failed: {e}", file=sys.stderr)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Parse environment variables ──────────────────────────────────────
    required_vars = ["GCS_BUCKET", "GCS_PREFIX", "RUN_ID"]
    for var in required_vars:
        if var not in os.environ:
            print(f"[trainer] FATAL: Required environment variable {var} is not set", file=sys.stderr)
            sys.exit(1)

    gcs_bucket = os.environ["GCS_BUCKET"]
    gcs_prefix = os.environ["GCS_PREFIX"]
    run_id = os.environ["RUN_ID"]
    base_model = os.environ.get("BASE_MODEL", "unsloth/llama-3.1-8b-bnb-4bit")
    quantization_type = os.environ.get("QUANTIZATION_TYPE", "Q4_K_M")
    max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
    lora_rank = int(os.environ.get("LORA_RANK", "16"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-4"))
    num_epochs = int(os.environ.get("NUM_EPOCHS", "3"))

    work_dir = Path("/tmp/training")

    # Clean up any leftover state from a previous run (spot preemption retry)
    _cleanup_work_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    data_path = work_dir / "data.jsonl"
    model_dir = work_dir / "model"
    merged_dir = work_dir / "merged"
    gguf_path = work_dir / "model.gguf"
    metrics_path = work_dir / "metrics.json"

    gcs_data_blob = f"{gcs_prefix}/{run_id}/data.jsonl"
    gcs_gguf_blob = f"{gcs_prefix}/{run_id}/model.gguf"
    gcs_metrics_blob = f"{gcs_prefix}/{run_id}/metrics.json"

    print(f"[trainer] Starting run {run_id}")
    print(f"[trainer] Base model: {base_model}")
    print(f"[trainer] Quantization: {quantization_type}")
    print(f"[trainer] LoRA rank: {lora_rank}, LR: {learning_rate}, Epochs: {num_epochs}")

    # ── Validate required tools ──────────────────────────────────────────
    convert_script = Path("/opt/llama.cpp/convert_hf_to_gguf.py")
    llama_quantize = os.environ.get("LLAMA_CPP_PATH", "/usr/local/bin/llama-quantize")

    if not convert_script.exists():
        print(f"[trainer] FATAL: Convert script not found at {convert_script}", file=sys.stderr)
        sys.exit(EXIT_CONVERSION_ERROR)

    if not Path(llama_quantize).exists():
        print(f"[trainer] FATAL: llama-quantize not found at {llama_quantize}", file=sys.stderr)
        sys.exit(EXIT_CONVERSION_ERROR)

    train_start = time.time()

    try:
        # ── 1. Download training data from GCS ───────────────────────────
        print("[trainer] Downloading training data from GCS...")
        from google.cloud import storage

        gcs_client = _retry(
            lambda: storage.Client(),
            retries=3,
            operation="GCS client init",
        )
        bucket = gcs_client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_data_blob)

        _retry(
            lambda: blob.download_to_filename(str(data_path)),
            retries=3,
            base_delay=5.0,
            operation=f"GCS download gs://{gcs_bucket}/{gcs_data_blob}",
        )

        # Parse JSONL into dataset
        examples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        if not examples:
            print("[trainer] FATAL: Training data file is empty", file=sys.stderr)
            sys.exit(EXIT_GCS_ERROR)

        print(f"[trainer] Loaded {len(examples)} training examples")

        # ── 2. Load model and tokenizer ──────────────────────────────────
        print("[trainer] Loading model and tokenizer...")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )

        # ── 3. Apply LoRA adapters ───────────────────────────────────────
        print("[trainer] Applying LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        # ── 4. Prepare dataset ───────────────────────────────────────────
        from datasets import Dataset

        def format_chat(example: dict) -> dict:
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
            )
            return {"text": text}

        dataset = Dataset.from_list(examples)
        dataset = dataset.map(format_chat)

        # ── 5. Train with SFTTrainer ─────────────────────────────────────
        print("[trainer] Starting training...")
        from trl import SFTTrainer
        from transformers import TrainingArguments

        training_args = TrainingArguments(
            output_dir=str(model_dir),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="no",
            optim="adamw_8bit",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            packing=False,
        )

        train_result = trainer.train()
        train_loss = train_result.training_loss
        train_steps = train_result.global_step
        print(f"[trainer] Training complete. Loss: {train_loss:.4f}, Steps: {train_steps}")

        # ── 6. Merge LoRA adapters ───────────────────────────────────────
        print("[trainer] Merging LoRA adapters...")
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

        # ── 7. Convert to GGUF ───────────────────────────────────────────
        print(f"[trainer] Converting to GGUF ({quantization_type})...")

        # First convert HF model to f16 GGUF
        f16_gguf = work_dir / "model-f16.gguf"
        convert_result = subprocess.run(
            ["python", str(convert_script), str(merged_dir),
             "--outfile", str(f16_gguf), "--outtype", "f16"],
            check=False,
            capture_output=True,
            text=True,
        )
        if convert_result.returncode != 0:
            print(f"[trainer] FATAL: HF-to-GGUF conversion failed:\n{convert_result.stderr}", file=sys.stderr)
            sys.exit(EXIT_CONVERSION_ERROR)

        # Then quantize
        quant_result = subprocess.run(
            [llama_quantize, str(f16_gguf), str(gguf_path), quantization_type],
            check=False,
            capture_output=True,
            text=True,
        )
        if quant_result.returncode != 0:
            print(f"[trainer] FATAL: Quantization failed:\n{quant_result.stderr}", file=sys.stderr)
            sys.exit(EXIT_CONVERSION_ERROR)

        gguf_size = gguf_path.stat().st_size
        print(f"[trainer] GGUF created: {gguf_size / 1024 / 1024:.1f} MB")

        # Clean up intermediate files to free disk space before upload
        if f16_gguf.exists():
            f16_gguf.unlink()
        if merged_dir.exists():
            shutil.rmtree(str(merged_dir))
        print("[trainer] Cleaned up intermediate files")

        # ── 8. Write metrics ─────────────────────────────────────────────
        train_duration = time.time() - train_start
        metrics = {
            "final_loss": train_loss,
            "num_steps": train_steps,
            "num_epochs_completed": float(num_epochs),
            "training_duration_seconds": train_duration,
            "additional_metrics": {
                "gguf_size_bytes": gguf_size,
                "quantization_type": quantization_type,
                "lora_rank": lora_rank,
                "learning_rate": learning_rate,
                "num_examples": len(examples),
                "base_model": base_model,
            },
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # ── 9. Upload GGUF + metrics to GCS ──────────────────────────────
        print("[trainer] Uploading GGUF to GCS...")
        gguf_blob = bucket.blob(gcs_gguf_blob)
        _retry(
            lambda: gguf_blob.upload_from_filename(str(gguf_path)),
            retries=3,
            base_delay=10.0,
            operation=f"GCS upload GGUF ({gguf_size / 1024 / 1024:.1f} MB)",
        )

        print("[trainer] Uploading metrics to GCS...")
        metrics_blob = bucket.blob(gcs_metrics_blob)
        _retry(
            lambda: metrics_blob.upload_from_filename(str(metrics_path)),
            retries=3,
            base_delay=5.0,
            operation="GCS upload metrics",
        )

        print(f"[trainer] Done. Artifacts at gs://{gcs_bucket}/{gcs_prefix}/{run_id}/")
        print(f"[trainer] Total duration: {train_duration:.0f}s")

    finally:
        # Always clean up /tmp to avoid disk pressure on shared nodes
        _cleanup_work_dir(work_dir)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[trainer] FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
