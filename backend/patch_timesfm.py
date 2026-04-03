"""
Monkey-patch TimesFM's _from_pretrained at import time to strip unexpected
HuggingFace Hub kwargs (proxies, resume_download, etc.) that newer versions
of huggingface_hub pass but TimesFM 2.x __init__ doesn't accept.

This patch wraps the _from_pretrained classmethod to filter kwargs before
they reach __init__, making it compatible with huggingface_hub >=0.31.
"""
import pathlib, inspect, functools

try:
    import timesfm
    from timesfm.timesfm_2p5 import timesfm_2p5_torch as _tfm_mod

    _cls = _tfm_mod.TimesFM_2p5_200M_torch
    _orig_from_pretrained = _cls._from_pretrained.__func__

    @classmethod  # type: ignore
    @functools.wraps(_orig_from_pretrained)
    def _patched_from_pretrained(cls, *, model_id, revision, cache_dir,
                                  force_download, local_files_only, token,
                                  config=None, **model_kwargs):
        # Strip all kwargs that TimesFM.__init__ doesn't accept
        _accepted = {"torch_compile"}
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in _accepted}
        return _orig_from_pretrained(
            cls,
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            config=config,
            **model_kwargs,
        )

    _cls._from_pretrained = _patched_from_pretrained
    print("TimesFM monkey-patched ✓")

except Exception as e:
    print(f"TimesFM patch failed: {e}")
    raise

