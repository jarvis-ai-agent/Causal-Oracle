"""
Causal Oracle — Modal Inference Endpoint
Serves TimesFM 2.5 + Chronos t5-small as a persistent GPU/CPU web endpoint.
Models are loaded once on container start and kept warm between requests.
"""
import modal
import numpy as np
from typing import Optional

# ── Modal app definition ──────────────────────────────────────────────────────
app = modal.App("causal-oracle-inference")

# Image with all ML dependencies
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "numpy",
        "huggingface_hub==0.36.2",
        "transformers>=4.40.0",
        "accelerate",
        "safetensors",
        "chronos-forecasting==2.2.2",
        "einops",
        "fastapi[standard]",
    )
    # Install TimesFM 2.x from latest source (no PyPI release for 2.x)
    .run_commands(
        "pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git'",
    )
    # Bundle patch_timesfm.py into the image so it's importable at runtime
    .add_local_python_source("patch_timesfm")

)

# Model weights volume for caching (avoids re-downloading on each cold start)
model_volume = modal.Volume.from_name("causal-oracle-models", create_if_missing=True)


# ── Global model state (loaded once per container) ───────────────────────────
_timesfm_model = None
_chronos_pipeline = None


def _load_models():
    global _timesfm_model, _chronos_pipeline
    import os, torch
    from chronos import BaseChronosPipeline

    os.environ["HF_HOME"] = "/model-cache/hf"

    if _timesfm_model is None:
        # Monkey-patch TimesFM BEFORE importing it to strip unexpected HF Hub kwargs
        import sys
        # Ensure timesfm is not already partially imported
        for mod in list(sys.modules.keys()):
            if "timesfm" in mod:
                del sys.modules[mod]
        import patch_timesfm  # noqa — must run before timesfm import
        import timesfm
        print("[Modal] Loading TimesFM 2.5...")
        _timesfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
        )
        print("[Modal] TimesFM loaded ✓")

    if _chronos_pipeline is None:
        print("[Modal] Loading Chronos t5-small...")
        _chronos_pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",
            dtype="float32",
        )
        print("[Modal] Chronos loaded ✓")


@app.function(
    image=inference_image,
    cpu=4,
    memory=8192,
    timeout=300,
    scaledown_window=300,
    volumes={"/model-cache": model_volume},

)
@modal.fastapi_endpoint(method="POST")
def forecast_endpoint(payload: dict) -> dict:
    """
    HTTP inference endpoint — models loaded once per container, reused across requests.
    POST body: {"context": [...], "horizon": int, "model": "timesfm"|"chronos"|"both", "num_samples": int}
    """
    import torch, timesfm as tfm

    _load_models()

    context   = payload.get("context", [])
    horizon   = int(payload.get("horizon", 5))
    model     = payload.get("model", "both")
    n_samples = int(payload.get("num_samples", 50))

    if not context:
        return {"error": "context is required"}

    ctx = np.array(context, dtype=np.float64)
    result = {
        "timesfm_point": None,
        "timesfm_quantiles": None,
        "chronos_point": None,
        "chronos_quantiles": None,
    }

    # TimesFM
    if model in ("timesfm", "both"):
        try:
            _timesfm_model.compile(tfm.ForecastConfig(
                max_context=min(512, len(ctx)),
                max_horizon=horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            ))
            pf_arr, qf_arr = _timesfm_model.forecast(horizon=horizon, inputs=[ctx])
            pf = np.array(pf_arr[0][:horizon])
            qf = np.array(qf_arr[0][:horizon])
            if qf.ndim == 1:
                qf = qf[:, np.newaxis]
            result["timesfm_point"]     = pf.tolist()
            result["timesfm_quantiles"] = qf.tolist()
        except Exception as e:
            print(f"[Modal] TimesFM error: {e}")

    # Chronos
    if model in ("chronos", "both"):
        try:
            ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
            samples = _chronos_pipeline.predict(
                inputs=ctx_tensor,
                prediction_length=horizon,
                num_samples=n_samples,
                limit_prediction_length=False,
            )[0].numpy()
            ch_point = samples.mean(axis=0)
            q_levels = np.linspace(0.1, 0.9, 10)
            ch_quant = np.array([
                np.quantile(samples[:, h], q_levels) for h in range(horizon)
            ])
            result["chronos_point"]     = ch_point.tolist()
            result["chronos_quantiles"] = ch_quant.tolist()
        except Exception as e:
            print(f"[Modal] Chronos error: {e}")

    return result


@app.local_entrypoint()
def test():
    """Quick smoke test — run with: modal run modal_inference.py"""
    ctx = list(np.random.randn(252) * 0.01)
    result = forecast_endpoint.remote({"context": ctx, "horizon": 3, "model": "both", "num_samples": 10})
    print("TimesFM point:", result["timesfm_point"])
    print("Chronos point:", result["chronos_point"])
    print("Test passed ✓")
