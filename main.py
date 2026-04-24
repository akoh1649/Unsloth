"""
FastAPI server wrapping Unsloth fine-tuning for Railway deployment.

Exposes two endpoints:
  GET  /health  — liveness check
  POST /train   — kick off a supervised fine-tuning job and stream progress
                  to stdout (visible in Railway deployment logs).
"""

import logging
import sys
import traceback

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging — write to stdout so Railway surfaces every line in the log panel.
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Unsloth Fine-Tuning API",
    description="HTTP wrapper around Unsloth model fine-tuning for Railway.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    model_name: str = Field(
        ...,
        description="HuggingFace model ID, e.g. 'unsloth/Llama-3.2-1B'",
        examples=["unsloth/Llama-3.2-1B"],
    )
    dataset_name: str = Field(
        ...,
        description="HuggingFace dataset ID, e.g. 'imdb'",
        examples=["imdb"],
    )
    num_epochs: int = Field(
        default=1,
        ge=1,
        description="Number of training epochs.",
    )
    max_steps: int = Field(
        default=-1,
        description="Maximum training steps. -1 means run for the full epoch(s).",
    )


class TrainResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Health check")
def health():
    """Returns 200 when the server is up and ready."""
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse, summary="Fine-tune a model")
def train(request: TrainRequest):
    """
    Load *model_name* with Unsloth, attach LoRA adapters, load *dataset_name*
    from HuggingFace, and run supervised fine-tuning.  Training progress is
    logged to stdout and therefore visible in the Railway deployment logs.
    """
    logger.info(
        "Received training request: model=%s  dataset=%s  epochs=%d  max_steps=%d",
        request.model_name,
        request.dataset_name,
        request.num_epochs,
        request.max_steps,
    )

    try:
        result = _run_training(
            model_name=request.model_name,
            dataset_name=request.dataset_name,
            num_epochs=request.num_epochs,
            max_steps=request.max_steps,
        )
        return TrainResponse(status="success", message=result)
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        logger.debug(traceback.format_exc())
        return TrainResponse(status="error", message=str(exc))


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------
def _run_training(
    model_name: str,
    dataset_name: str,
    num_epochs: int,
    max_steps: int,
) -> str:
    """
    Orchestrates the full fine-tuning pipeline:
      1. Load model + tokenizer via Unsloth (4-bit quantised by default).
      2. Attach LoRA adapters with `get_peft_model`.
      3. Load the HuggingFace dataset.
      4. Run SFT with TRL's SFTTrainer (Unsloth-patched for speed).
      5. Return a summary string.
    """
    # ------------------------------------------------------------------
    # 1. Imports — deferred so the server starts fast even if the GPU
    #    stack is slow to initialise.
    # ------------------------------------------------------------------
    logger.info("Importing Unsloth and training dependencies …")
    # unsloth must be imported before transformers / trl
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ------------------------------------------------------------------
    # 2. Load model + tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,          # auto-detect: bf16 on Ampere+, fp16 otherwise
        load_in_4bit=True,   # QLoRA — keeps VRAM usage low
    )
    logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # 3. Attach LoRA adapters
    # ------------------------------------------------------------------
    logger.info("Attaching LoRA adapters …")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    logger.info("LoRA adapters attached.")

    # ------------------------------------------------------------------
    # 4. Load dataset
    # ------------------------------------------------------------------
    logger.info("Loading dataset: %s", dataset_name)
    dataset = load_dataset(dataset_name, split="train")
    logger.info("Dataset loaded — %d examples.", len(dataset))

    # ------------------------------------------------------------------
    # 5. Fine-tune
    # ------------------------------------------------------------------
    logger.info(
        "Starting fine-tuning: epochs=%d  max_steps=%d", num_epochs, max_steps
    )

    training_args_kwargs = dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        report_to="none",   # disable W&B / MLflow in Railway
    )
    if max_steps != -1:
        training_args_kwargs["max_steps"] = max_steps

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(**training_args_kwargs),
    )

    trainer_stats = trainer.train()

    summary = (
        f"Training complete. "
        f"Steps: {trainer_stats.global_step}, "
        f"Loss: {trainer_stats.training_loss:.4f}, "
        f"Runtime: {trainer_stats.metrics.get('train_runtime', 0):.1f}s"
    )
    logger.info(summary)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
