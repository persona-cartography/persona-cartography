"""LoRA vectors in factored (B, A) form for memory-efficient vector space operations.

Instead of materializing full ∆W = B @ A matrices (which can be billions of
elements for large models), this module keeps LoRA weights in their low-rank
factored form and performs all operations — dot products, addition, scaling,
PCA — directly on the small factor matrices.

Memory: O(n_modules × r × (d_in + d_out)) instead of O(n_modules × d_in × d_out).
For rank 16, that's roughly a 250x reduction.

Rank and lossy operations
------------------
All arithmetic in the vector space is exact, but **operations that combine
vectors grow the internal rank**:

- ``a + b`` produces a vector with rank ``r_a + r_b`` (B columns and A rows
  are concatenated).
- ``a - b`` similarly produces rank ``r_a + r_b``.
- Scalar multiplication and negation leave the rank unchanged.

This means that after repeated additions the internal rank can far exceed the
original adapter rank. The representation is still exact — no information is
lost — but storage and compute grow with rank. Note that ``max_rank``
reflects factor shape, not true matrix rank: ``a + a`` has double the
``max_rank`` of ``2 * a`` even though ΔW is identical. Methods like
``to_peft_model()`` and ``to_file()`` use ``max_rank`` for the adapter's
``r``, so consider calling ``rank_reduce()`` after arithmetic to avoid
unnecessarily inflated adapter rank.

**Loss only occurs in** ``rank_reduce(new_rank)`` — truncated SVD to a lower
rank. Information in the discarded singular vectors is gone. Repeated
add → rank_reduce cycles accumulate approximation error. If you need to
preserve fidelity, keep the full-rank vector and only reduce at the end.

Model I/O
---------
- ``write_to_existing_peft_model()`` writes factors into an existing PeftModel.
  By default (``resize_peft_rank=True``) the model's adapter is resized to
  match the vector's rank. Pass ``resize_peft_rank=False`` to truncate via SVD
  instead (lossy). If you want a specific rank, call ``rank_reduce()`` first.
- ``to_peft_model()`` creates a fresh PeftModel from a base model and injects
  the vector's weights. All config (rank, target_modules, etc.) is inferred.
- ``to_file()`` saves the vector as a PEFT-compatible adapter directory
  (``adapter_config.json`` + ``adapter_model.safetensors``), loadable by
  ``from_file()`` or ``PeftModel.from_pretrained()``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig as PeftLoraConfig, PeftModel, get_peft_model
from safetensors.torch import safe_open, save_file
from torch import Tensor, nn

from src.utils.linalg import reduce_lora_rank_efficient
from src.utils.model_layer_info import LayerIdxExtractor, extract_layer_idx
from src.utils.peft_manipulations import _iter_all_lora_modules, _matches_target_modules


class LoRaVector:
    """A point in LoRA weight space, stored in factored (B, A) form.

    Each instance holds a dict mapping module names to (B, A) factor pairs,
    where B is (out_features, r) and A is (r, in_features). The rank r can
    vary per module and grows after addition/subtraction.

    Args:
        factors: Mapping of module_name -> (B, A) tensor pairs.
    """

    def __init__(self, factors: dict[str, tuple[Tensor, Tensor]]) -> None:
        if not factors:
            raise ValueError("factors dict must not be empty")
        self._factors = dict(factors)

    @classmethod
    def from_peft(
        cls,
        model: PeftModel,
        adapter_name: str,
        *,
        include_scaling: bool = True,
        target_modules: list[str] | dict[int, list[str]] | None = None,
        layers: list[int] | None = None,
        layer_idx_extractor: LayerIdxExtractor | None = None,
    ) -> LoRaVector:
        """Extract a LoRAVector from a PeftModel's adapter.

        Args:
            model: A PEFT model with LoRA adapters.
            adapter_name: Which adapter to extract.
            include_scaling: If True, absorb the PEFT scaling factor
                (lora_alpha / r) into B.
            target_modules: Filter which modules to include (see
                BaseLoRaModifier for semantics).
            layers: If provided, only include modules in these layer indices.
            layer_idx_extractor: Custom layer index extraction function.

        Returns:
            A LoRAVector with the adapter's factors.
        """
        if adapter_name not in model.peft_config:
            available = list(model.peft_config.keys())
            raise ValueError(
                f"Adapter '{adapter_name}' not found. Available: {available}"
            )

        extractor = layer_idx_extractor or extract_layer_idx
        layers_set = set(layers) if layers is not None else None

        factors: dict[str, tuple[Tensor, Tensor]] = {}
        for name, module in _iter_all_lora_modules(model, adapter_name):
            # Apply filtering
            if isinstance(target_modules, dict):
                layer_idx = extractor(name)
                if layer_idx is None or layer_idx not in target_modules:
                    continue
                if not _matches_target_modules(name, target_modules[layer_idx]):
                    continue
            else:
                if target_modules is not None:
                    if not _matches_target_modules(name, target_modules):
                        continue
                if layers_set is not None:
                    layer_idx = extractor(name)
                    if layer_idx is None or layer_idx not in layers_set:
                        continue

            A = module.lora_A[adapter_name].weight.data.clone()
            B = module.lora_B[adapter_name].weight.data.clone()

            if include_scaling:
                scaling = module.scaling[adapter_name]
                B = scaling * B

            factors[name] = (B, A)

        if not factors:
            raise ValueError(
                "No LoRA modules matched the given filters. "
                "Check target_modules and layers arguments."
            )

        return cls(factors)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        target_modules: list[str] | None = None,
        layers: list[int] | None = None,
        layer_idx_extractor: LayerIdxExtractor | None = None,
    ) -> LoRaVector:
        """Load a LoRAVector from a saved adapter directory.

        Reads adapter_config.json for config and adapter_model.safetensors
        for weights. Absorbs the scaling factor (lora_alpha / r) into B.

        Args:
            path: Path to adapter directory containing adapter_config.json
                and adapter_model.safetensors.
            target_modules: Filter which modules to include by name suffix.
            layers: If provided, only include modules in these layer indices.
            layer_idx_extractor: Custom layer index extraction function.

        Returns:
            A LoRAVector with the adapter's factors.
        """
        path = Path(path)
        config_path = path / "adapter_config.json"
        weights_path = path / "adapter_model.safetensors"

        if not config_path.exists():
            raise FileNotFoundError(f"No adapter_config.json found at {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No adapter_model.safetensors found at {weights_path}"
            )

        with open(config_path) as f:
            config = json.load(f)

        r = config["r"]
        lora_alpha = config.get("lora_alpha", r)
        scaling = lora_alpha / r

        extractor = layer_idx_extractor or extract_layer_idx
        layers_set = set(layers) if layers is not None else None

        factors: dict[str, tuple[Tensor, Tensor]] = {}
        with safe_open(str(weights_path), framework="pt") as f:
            # Discover all module names from the weight keys
            all_keys = list(f.keys())
            module_names = sorted(
                {
                    k.removesuffix(".lora_A.weight").removesuffix(".lora_B.weight")
                    for k in all_keys
                }
            )

            for module_name in module_names:
                a_key = f"{module_name}.lora_A.weight"
                b_key = f"{module_name}.lora_B.weight"
                if a_key not in all_keys or b_key not in all_keys:
                    continue

                # Apply filtering
                if target_modules is not None:
                    if not _matches_target_modules(module_name, target_modules):
                        continue
                if layers_set is not None:
                    layer_idx = extractor(module_name)
                    if layer_idx is None or layer_idx not in layers_set:
                        continue

                A = f.get_tensor(a_key)
                B = f.get_tensor(b_key)
                B = scaling * B

                factors[module_name] = (B, A)

        if not factors:
            raise ValueError(
                "No LoRA modules matched the given filters. "
                "Check target_modules and layers arguments."
            )

        return cls(factors)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        *,
        subfolder: str | None = None,
        target_modules: list[str] | None = None,
        layers: list[int] | None = None,
        layer_idx_extractor: LayerIdxExtractor | None = None,
        revision: str | None = None,
        repo_type: str | None = None,
    ) -> LoRaVector:
        """Load a LoRAVector from a HuggingFace Hub repository.

        Downloads the adapter files and delegates to :meth:`from_file`.

        Args:
            repo_id: HuggingFace repo ID (e.g. "user/model-lora-adapters").
            subfolder: Subfolder within the repo containing the adapter
                (e.g. "sarcasm" for a multi-adapter repo).
            target_modules: Filter which modules to include by name suffix.
            layers: If provided, only include modules in these layer indices.
            layer_idx_extractor: Custom layer index extraction function.
            revision: Git revision (branch, tag, or commit hash) to download.
            repo_type: Type of HuggingFace repo ("model", "dataset", "space").
                Defaults to "model" via ``snapshot_download``.

        Returns:
            A LoRAVector with the adapter's factors.
        """
        allow_patterns = [f"{subfolder}/*"] if subfolder else None
        local_path = Path(
            snapshot_download(
                repo_id,
                allow_patterns=allow_patterns,
                revision=revision,
                repo_type=repo_type,
            )
        )
        adapter_path = local_path / subfolder if subfolder else local_path
        return cls.from_file(
            adapter_path,
            target_modules=target_modules,
            layers=layers,
            layer_idx_extractor=layer_idx_extractor,
        )

    def zero_like(self) -> LoRaVector:
        """Return a zero vector with the same module structure.

        Each module gets zero-valued (B, A) factors with rank 1 and the
        same (out_features, in_features) dimensions as this vector.
        """
        factors = {}
        for name, (B, A) in self._factors.items():
            factors[name] = (
                torch.zeros(B.shape[0], 1, dtype=B.dtype, device=B.device),
                torch.zeros(1, A.shape[1], dtype=A.dtype, device=A.device),
            )
        return LoRaVector(factors)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def factors(self) -> dict[str, tuple[Tensor, Tensor]]:
        """The underlying (B, A) factor pairs, keyed by module name."""
        return self._factors

    @property
    def module_names(self) -> list[str]:
        """Sorted list of module names."""
        return sorted(self._factors.keys())

    @property
    def max_rank(self) -> int:
        """Maximum rank across all modules (factor shape, not true matrix rank).

        After addition, ``max_rank`` reflects the concatenated factor size,
        not the true rank of ΔW. For example, ``a + a`` has
        ``max_rank == 2 * a.max_rank`` even though the resulting ΔW is
        identical to ``2 * a`` (which preserves the original rank). Call
        ``rank_reduce()`` to compress redundant dimensions.
        """
        return max(B.shape[1] for B, _ in self._factors.values())

    @property
    def total_params(self) -> int:
        """Total number of parameters stored (sum of r*(in+out) per module)."""
        total = 0
        for B, A in self._factors.values():
            total += B.numel() + A.numel()
        return total

    @property
    def vector_dim(self) -> int:
        """Dimensionality of the flattened ΔW vector (sum of out*in per module)."""
        return sum(B.shape[0] * A.shape[1] for B, A in self._factors.values())

    # ------------------------------------------------------------------
    # Arithmetic (exact, no SVD)
    # ------------------------------------------------------------------

    def _validate_compatible(self, other: LoRaVector) -> None:
        """Check that two vectors have the same modules and compatible shapes."""
        if self._factors.keys() != other._factors.keys():
            missing = self._factors.keys() ^ other._factors.keys()
            raise ValueError(
                f"LoRAVectors have different module names. "
                f"Symmetric difference: {sorted(missing)}"
            )
        for name in self._factors:
            B1, A1 = self._factors[name]
            B2, A2 = other._factors[name]
            # Check ∆W shape compatibility (out_features and in_features)
            if B1.shape[0] != B2.shape[0] or A1.shape[1] != A2.shape[1]:
                raise ValueError(
                    f"Incompatible ∆W shapes for module '{name}': "
                    f"({B1.shape[0]}, {A1.shape[1]}) vs ({B2.shape[0]}, {A2.shape[1]})"
                )

    def __add__(self, other: LoRaVector) -> LoRaVector:
        self._validate_compatible(other)
        factors = {}
        for name in self._factors:
            B1, A1 = self._factors[name]
            B2, A2 = other._factors[name]
            factors[name] = (
                torch.cat([B1, B2], dim=1),
                torch.cat([A1, A2], dim=0),
            )
        return LoRaVector(factors)

    def __sub__(self, other: LoRaVector) -> LoRaVector:
        self._validate_compatible(other)
        factors = {}
        for name in self._factors:
            B1, A1 = self._factors[name]
            B2, A2 = other._factors[name]
            factors[name] = (
                torch.cat([B1, B2], dim=1),
                torch.cat([A1, -A2], dim=0),
            )
        return LoRaVector(factors)

    def __neg__(self) -> LoRaVector:
        factors = {}
        for name, (B, A) in self._factors.items():
            factors[name] = (-B, A)
        return LoRaVector(factors)

    def __mul__(self, scalar: float) -> LoRaVector:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        factors = {}
        for name, (B, A) in self._factors.items():
            factors[name] = (scalar * B, A)
        return LoRaVector(factors)

    def __rmul__(self, scalar: float) -> LoRaVector:
        return self.__mul__(scalar)

    # ------------------------------------------------------------------
    # Metrics (never materialize full ∆W)
    # ------------------------------------------------------------------

    def dot(self, other: LoRaVector) -> float:
        """Compute the dot product ⟨self, other⟩ in ∆W space.

        Uses the identity: ⟨B₁A₁, B₂A₂⟩ = Σᵢ tr(A₂ᵢ A₁ᵢᵀ · B₁ᵢᵀ B₂ᵢ).
        Cost: O(r₁ r₂ (in + out)) per module instead of O(r in out).
        """
        self._validate_compatible(other)
        total = 0.0
        for name in self._factors:
            B1, A1 = self._factors[name]
            B2, A2 = other._factors[name]
            # B1^T B2 is (r1, r2), A2 A1^T is (r2, r1)
            BtB = B1.T @ B2  # (r1, r2)
            AAt = A2 @ A1.T  # (r2, r1)
            # tr(AAt @ BtB) = tr((r2, r1) @ (r1, r2)) = sum of elementwise product
            total += (AAt * BtB.T).sum().item()
        return total

    @property
    def norm(self) -> float:
        """Frobenius norm of the ∆W vector."""
        return math.sqrt(self.dot(self))

    def cosine_similarity(self, other: LoRaVector) -> float:
        """Cosine similarity between two LoRAVectors in ∆W space."""
        d = self.dot(other)
        n1 = self.norm
        n2 = other.norm
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return d / (n1 * n2)

    # ------------------------------------------------------------------
    # Rank management
    # ------------------------------------------------------------------

    def rank_reduce(self, new_rank: int) -> LoRaVector:
        """Return a new LoRAVector with each module's rank reduced via SVD.

        Uses the efficient QR-based rank reduction from src.utils.linalg
        (no full ∆W materialization).

        Args:
            new_rank: Target rank for all modules.

        Returns:
            New LoRAVector with reduced rank.
        """
        if new_rank < 1:
            raise ValueError(f"new_rank must be >= 1, got {new_rank=}")

        factors = {}
        for name, (B, A) in self._factors.items():
            current_rank = A.shape[0]
            if new_rank > current_rank:
                raise ValueError(f"new_rank must be <= {current_rank=} got {new_rank=}")
            if current_rank <= new_rank:
                factors[name] = (B.clone(), A.clone())
            else:
                new_A, new_B = reduce_lora_rank_efficient(A, B, new_rank=new_rank)
                factors[name] = (new_B, new_A)
        return LoRaVector(factors)

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def write_to_existing_peft_model(
        self,
        model: PeftModel,
        adapter_name: str,
        *,
        resize_peft_rank: bool = True,
    ) -> None:
        """Write this vector's factors into an existing PeftModel's adapter.

        By default (``resize_peft_rank=True``), the model's adapter is resized
        to match the vector's rank — both expanding and shrinking. This avoids
        lossy truncation and keeps ``module.r`` consistent with actual weight
        shapes. Pass ``resize_peft_rank=False`` to instead truncate via SVD
        when the vector's rank exceeds the model's (lossy).

        To write at a specific rank, call ``rank_reduce()`` first::

            vec.rank_reduce(4).write_to_existing_peft_model(model, "default")

        Sets ``module.scaling`` to 1.0 (scaling is already absorbed into B)
        and syncs ``model.peft_config`` so that ``save_pretrained()`` writes
        the correct rank and lora_alpha to ``adapter_config.json``.

        Args:
            model: PeftModel to write into (modified in-place).
            adapter_name: Which adapter to overwrite.
            resize_peft_rank: If True (default), resize the adapter's
                nn.Linear modules and module.r to match the vector's rank.
                If False, truncate via SVD when the vector's rank exceeds
                the model's.
        """
        if adapter_name not in model.peft_config:
            available = list(model.peft_config.keys())
            raise ValueError(
                f"Adapter '{adapter_name}' not found. Available: {available}"
            )

        modules = dict(model.named_modules())
        written_ranks = []

        for name, (B, A) in self._factors.items():
            if name not in modules:
                raise ValueError(f"Module '{name}' not found in model")

            module = modules[name]
            model_rank = module.r[adapter_name]
            vec_rank = A.shape[0]
            device = module.lora_A[adapter_name].weight.device
            dtype = module.lora_A[adapter_name].weight.dtype

            if vec_rank != model_rank:
                if resize_peft_rank:
                    in_features = A.shape[1]
                    out_features = B.shape[0]
                    module.lora_A[adapter_name] = nn.Linear(
                        in_features, vec_rank, bias=False
                    ).to(device=device, dtype=dtype)
                    module.lora_B[adapter_name] = nn.Linear(
                        vec_rank, out_features, bias=False
                    ).to(device=device, dtype=dtype)
                    module.r[adapter_name] = vec_rank
                elif vec_rank > model_rank:
                    A, B = reduce_lora_rank_efficient(A, B, new_rank=model_rank)

            module.lora_A[adapter_name].weight = nn.Parameter(
                A.to(device=device, dtype=dtype)
            )
            module.lora_B[adapter_name].weight = nn.Parameter(
                B.to(device=device, dtype=dtype)
            )
            module.scaling[adapter_name] = 1.0
            written_ranks.append(module.r[adapter_name])

        # Sync peft_config so save_pretrained writes correct metadata.
        # Set lora_alpha = r so scaling = lora_alpha/r = 1.0, matching the
        # module.scaling = 1.0 we set above. This prevents double-scaling
        # when the adapter is later loaded via PeftModel.from_pretrained.
        new_r = max(written_ranks)
        model.peft_config[adapter_name].r = new_r
        model.peft_config[adapter_name].lora_alpha = new_r

    def to_peft_model(
        self,
        base_model: nn.Module,
        adapter_name: str = "default",
    ) -> PeftModel:
        """Create a PeftModel from a base model and inject this vector's weights.

        WARNING: this updates base_model in place, be careful.

        All adapter configuration is inferred from the vector itself — no
        LoraConfig needed. Typical usage::

            base = AutoModelForCausalLM.from_pretrained("meta-llama/...")
            combined = vec_a + vec_b
            peft_model = combined.to_peft_model(base)

        The following LoraConfig fields are set automatically:

        - ``r``: set to ``self.max_rank`` (the vector's factor shapes
          determine the rank).
        - ``lora_alpha``: set to ``self.max_rank`` (= r), giving scaling
          = 1.0. This is required because the vector's B factors already
          have the original adapter's scaling absorbed. Any other value
          would cause double-scaling when saving and reloading.
        - ``target_modules``: inferred from the vector's module names by
          extracting suffixes (e.g. ``model.layers.0.self_attn.q_proj``
          → ``q_proj``). PEFT pattern-matches these against the base model
          to wrap the correct modules with LoRA layers.
        - ``bias``: ``"none"`` — the vector doesn't store bias values.
        - ``init_lora_weights``: ``False`` — skips PEFT's default
          initialization since we overwrite immediately.

        Args:
            base_model: The base model (e.g. from
                ``AutoModelForCausalLM.from_pretrained``).
            adapter_name: Name for the adapter in the PeftModel.

        Returns:
            A PeftModel with this vector's weights injected.
        """
        target_modules = sorted({name.split(".")[-1] for name in self.module_names})
        config = PeftLoraConfig(
            r=self.max_rank,
            lora_alpha=self.max_rank,
            target_modules=target_modules,
            bias="none",
            init_lora_weights=False,
        )
        peft_model = get_peft_model(base_model, config, adapter_name=adapter_name)
        self.write_to_existing_peft_model(peft_model, adapter_name)
        return peft_model

    def to_file(self, path: str | Path) -> None:
        """Save this vector as a PEFT-compatible adapter directory.

        The inverse of ``from_file``. Writes two files:

        - ``adapter_config.json`` — adapter metadata (rank, lora_alpha,
          target_modules, etc.).
        - ``adapter_model.safetensors`` — the (B, A) weight tensors.

        The saved directory can be loaded by ``LoRaVector.from_file(path)``
        or by ``PeftModel.from_pretrained(base_model, path)``.

        Scaling is already absorbed into B, so ``lora_alpha`` is set equal
        to ``r`` in the config (giving scaling = 1.0). This ensures that
        ``from_file`` (which computes ``scaling = lora_alpha / r`` and
        multiplies into B) is a no-op on reload — no double-scaling.

        Args:
            path: Directory to write into. Created if it doesn't exist.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        target_modules = sorted({name.split(".")[-1] for name in self.module_names})

        # task_type is hardcoded — this codebase only operates on causal LMs.
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": self.max_rank,
            "lora_alpha": self.max_rank,
            "target_modules": target_modules,
            "bias": "none",
        }
        with open(path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)

        tensors = {}
        for name, (B, A) in self._factors.items():
            tensors[f"{name}.lora_A.weight"] = A.contiguous()
            tensors[f"{name}.lora_B.weight"] = B.contiguous()
        save_file(tensors, str(path / "adapter_model.safetensors"))

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LoRaVector:
        """Return a new LoRAVector with tensors moved/cast.

        Args:
            device: Target device.
            dtype: Target dtype.

        Returns:
            New LoRAVector with moved/cast tensors.
        """
        factors = {}
        for name, (B, A) in self._factors.items():
            factors[name] = (
                B.to(device=device, dtype=dtype),
                A.to(device=device, dtype=dtype),
            )
        return LoRaVector(factors)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._factors)
        ranks = [A.shape[0] for _, (_, A) in sorted(self._factors.items())]
        min_r, max_r = min(ranks), max(ranks)
        rank_str = str(min_r) if min_r == max_r else f"{min_r}-{max_r}"
        return (
            f"LoRAVector(modules={n}, rank={rank_str}, "
            f"vector_dim={self.vector_dim:,}, "
            f"params={self.total_params:,})"
        )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _lincomb(coeffs: Sequence[float], vectors: list[LoRaVector]) -> LoRaVector:
    """Weighted sum: Σ coeffs[i] * vectors[i]."""
    result = coeffs[0] * vectors[0]
    for i in range(1, len(vectors)):
        if coeffs[i] != 0.0:
            result = result + (coeffs[i] * vectors[i])
    return result


def gram_matrix(vectors: list[LoRaVector]) -> Tensor:
    """Compute the n x n Gram matrix of pairwise dot products.

    Args:
        vectors: List of LoRAVector instances (must all be compatible).

    Returns:
        Symmetric tensor of shape (n, n) where G[i,j] = vectors[i].dot(vectors[j]).
    """
    n = len(vectors)
    G = torch.zeros(n, n)
    for i in range(n):
        G[i, i] = vectors[i].dot(vectors[i])
        for j in range(i):
            val = vectors[i].dot(vectors[j])
            G[i, j] = val
            G[j, i] = val
    return G


def cosine_similarity_matrix(
    vectors: list[LoRaVector],
    *,
    _gram: Tensor | None = None,
) -> Tensor:
    """Compute the n x n pairwise cosine similarity matrix.

    Args:
        vectors: List of LoRAVector instances (must all be compatible).
        _gram: Precomputed Gram matrix (internal use by LoRAVectorCollection).

    Returns:
        Symmetric tensor of shape (n, n) where C[i,j] is the cosine
        similarity between vectors[i] and vectors[j].
    """
    G = _gram if _gram is not None else gram_matrix(vectors)
    return _cosine_from_gram(G)


def _cosine_from_gram(G: Tensor) -> Tensor:
    """Convert a Gram matrix to a cosine similarity matrix."""
    norms = torch.sqrt(torch.diag(G))
    norms = norms.clamp(min=1e-10)
    return G / norms.unsqueeze(1) / norms.unsqueeze(0)


# ---------------------------------------------------------------------------
# LoRAVectorCollection
# ---------------------------------------------------------------------------


@dataclass
class PCAResult:
    """Result from LoRAVectorCollection.pca, bundling metadata."""

    space: LoRaVectorSpace
    input_coords: (
        Tensor  # (n_vectors, n_dims) — coordinates of input vectors in PCA space
    )
    eigenvalues: Tensor  # all eigenvalues (for scree plots)
    explained_variance: Tensor  # fraction explained per retained component
    pc_scales: Tensor  # sqrt(eigenvalue) per retained component — std dev along each PC
    input_names: list[str]  # input vector names (row labels for input_coords)
    pc_names: list[str]  # component names, e.g. ["PC0", "PC1", ...] (column labels)
    n_dims: int  # number of principal components calculated
    normalized: (
        bool  # whether coords/space use std-dev units (True) or raw dot-product units
    )


class LoRaVectorCollection:
    """A named set of LoRAVectors with cached bulk analysis operations.

    Lazily computes the Gram matrix and reuses it for cosine similarity
    and PCA, avoiding redundant pairwise dot products.

    Args:
        vectors: Either a dict mapping names to LoRAVectors, or a list
            (in which case names default to "0", "1", "2", ...).
    """

    def __init__(self, vectors: dict[str, LoRaVector] | list[LoRaVector]) -> None:
        if isinstance(vectors, dict):
            if not vectors:
                raise ValueError("vectors dict must not be empty")
            self._names = list(vectors.keys())
            self._vectors = list(vectors.values())
        else:
            if not vectors:
                raise ValueError("vectors list must not be empty")
            self._names = [str(i) for i in range(len(vectors))]
            self._vectors = list(vectors)

        self._gram_cache: Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Names in insertion/index order."""
        return list(self._names)

    @property
    def vectors(self) -> list[LoRaVector]:
        """Vectors in matching order."""
        return list(self._vectors)

    def __len__(self) -> int:
        return len(self._vectors)

    def __getitem__(self, key: str | int) -> LoRaVector:
        if isinstance(key, int):
            return self._vectors[key]
        try:
            idx = self._names.index(key)
        except ValueError:
            raise KeyError(f"No vector named '{key}'") from None
        return self._vectors[idx]

    # ------------------------------------------------------------------
    # Analysis (Gram matrix computed once, cached)
    # ------------------------------------------------------------------

    def gram_matrix(self) -> Tensor:
        """Compute (or return cached) n x n Gram matrix of pairwise dot products."""
        if self._gram_cache is None:
            self._gram_cache = gram_matrix(self._vectors)
        return self._gram_cache

    def cosine_similarity_matrix(self) -> Tensor:
        """Compute n x n pairwise cosine similarity matrix (reuses cached Gram)."""
        return _cosine_from_gram(self.gram_matrix())

    def _centered_eigen(self) -> tuple[Tensor, Tensor]:
        """Center the cached Gram matrix and eigendecompose (descending order).

        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted by descending eigenvalue.
        """
        G = self.gram_matrix()
        n = len(self._vectors)
        ones_n = torch.ones(n, n) / n
        G_centered = G - ones_n @ G - G @ ones_n + ones_n @ G @ ones_n
        eigenvalues, eigenvectors = torch.linalg.eigh(G_centered)
        return eigenvalues.flip(0), eigenvectors.flip(1)

    def _mean(self) -> LoRaVector:
        """Mean of all vectors in the collection."""
        coeffs = [1.0 / len(self._vectors)] * len(self._vectors)
        return _lincomb(coeffs, self._vectors)

    def pca(self, n_dims: int | None = None, *, normalize: bool = True) -> PCAResult:
        """Perform kernel PCA on the collection (reuses cached Gram matrix).

        Args:
            n_dims: Number of principal components to retain.
                Defaults to len(vectors).
            normalize: If True (default), coordinates and space use
                standard-deviation units (delta=1 in shift means 1 std dev).
                If False, use raw dot-product units.

        Returns:
            PCAResult containing the space, coordinates, eigenvalues,
            and explained variance.
        """
        n = len(self._vectors)
        if n_dims is None:
            n_dims = n
        elif n_dims < 1:
            raise ValueError(f"n_dims must be >= 1, got {n_dims}")
        elif n_dims > n:
            raise ValueError(f"n_dims ({n_dims}) cannot exceed number of vectors ({n})")

        eigenvalues, eigenvectors = self._centered_eigen()
        eigenvalues_safe = eigenvalues.clamp(min=0)

        # Explained variance
        total_var = eigenvalues_safe.sum()
        explained = (
            eigenvalues_safe[:n_dims] / total_var
            if total_var > 0
            else torch.zeros(n_dims)
        )

        # Orthonormal basis: each PC is a normalized linear combination of input vectors
        basis: list[LoRaVector] = []
        for k in range(n_dims):
            if eigenvalues_safe[k] <= 0:
                break
            basis_vec = _lincomb(eigenvectors[:, k].tolist(), self._vectors)
            vec_norm = basis_vec.norm
            if vec_norm > 0:
                basis_vec = (1.0 / vec_norm) * basis_vec
            basis.append(basis_vec)

        n_actual = len(basis)
        explained = explained[:n_actual]
        pc_scales = eigenvalues_safe[:n_actual].sqrt()
        pc_names = [f"PC{k}" for k in range(n_actual)]

        # Raw coords: (v_i - mean) · basis_k = eigvec[i,k] * sqrt(λ_k)
        raw_coords = eigenvectors[:, :n_actual] * pc_scales

        if normalize:
            # Normalize so each PC has unit variance.
            # Variance of raw_coords[:, k] = λ_k / n, so std = sqrt(λ_k / n).
            std_scales = pc_scales / math.sqrt(n)
            input_coords = raw_coords / std_scales
            space = LoRaVectorSpace(basis, center=self._mean(), scale=std_scales)
        else:
            input_coords = raw_coords
            space = LoRaVectorSpace(basis, center=self._mean())

        return PCAResult(
            space=space,
            input_coords=input_coords,
            eigenvalues=eigenvalues,
            explained_variance=explained,
            pc_scales=pc_scales,
            input_names=list(self._names),
            pc_names=pc_names,
            n_dims=len(pc_names),
            normalized=normalize,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"LoRAVectorCollection(n={len(self._vectors)}, names={self._names})"


# ---------------------------------------------------------------------------
# LoRAVectorSpace
# ---------------------------------------------------------------------------


class LoRaVectorSpace:
    """An orthonormal basis in LoRA weight space.

    Provides projection, reconstruction, and shifting operations.
    Typically created via :meth:`LoRAVectorCollection.pca`.

    An optional affine transformation (center + scale) maps between the
    full LoRA space and the coordinate space:
    - ``center``: origin of the subspace (e.g. mean of training vectors for PCA).
    - ``scale``: per-dimension scaling (e.g. sqrt(eigenvalues) for PCA).

    Args:
        basis: Either a dict mapping names to basis vectors, or a list
            (in which case names default to "0", "1", "2", ...).
        center: Optional origin vector. When set, ``project`` subtracts it
            before dotting, and ``reconstruct`` adds it back.
        scale: Optional per-dimension scale tensor of shape ``(n_dims,)``.
            When set, ``project`` divides by it and ``reconstruct`` multiplies.
    """

    def __init__(
        self,
        basis: dict[str, LoRaVector] | list[LoRaVector],
        *,
        center: LoRaVector | None = None,
        scale: Tensor | None = None,
    ) -> None:
        if isinstance(basis, dict):
            if not basis:
                raise ValueError("basis must not be empty")
            self._names = list(basis.keys())
            self._basis = list(basis.values())
        else:
            if not basis:
                raise ValueError("basis must not be empty")
            self._names = [str(i) for i in range(len(basis))]
            self._basis = list(basis)
        self._center = center
        self._scale = scale

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Names of the basis vectors."""
        return list(self._names)

    @property
    def basis(self) -> list[LoRaVector]:
        """The orthonormal basis vectors."""
        return self._basis

    @property
    def n_dims(self) -> int:
        """Number of basis dimensions."""
        return len(self._basis)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def project(self, vec: LoRaVector) -> Tensor:
        """Project a LoRAVector onto this basis.

        Subtracts ``center`` (if set) before dotting, then divides by
        ``scale`` (if set).

        Args:
            vec: The vector to project.

        Returns:
            Coordinates tensor of shape (n_dims,).
        """
        v = vec if self._center is None else (vec + (-1.0 * self._center))
        coords = torch.tensor([v.dot(b) for b in self._basis])
        if self._scale is not None:
            coords = coords / self._scale
        return coords

    def project_all(self, vectors: list[LoRaVector]) -> Tensor:
        """Project multiple LoRAVectors onto this basis.

        Args:
            vectors: List of vectors to project.

        Returns:
            Coordinates tensor of shape (n_vectors, n_dims).
        """
        return torch.stack([self.project(v) for v in vectors])

    def reconstruct(self, coords: Tensor) -> LoRaVector:
        """Reconstruct a LoRAVector from coordinates in this basis.

        Multiplies by ``scale`` (if set) before summing, then adds
        ``center`` (if set).

        Args:
            coords: Coordinates tensor of shape (n_dims,).

        Returns:
            The reconstructed LoRAVector.
        """
        if coords.shape != (self.n_dims,):
            raise ValueError(
                f"Expected coords of shape ({self.n_dims},), got {coords.shape}"
            )

        c = coords * self._scale if self._scale is not None else coords
        result = c[0].item() * self._basis[0]
        for i in range(1, self.n_dims):
            result = result + (c[i].item() * self._basis[i])
        if self._center is not None:
            result = self._center + result
        return result

    def shift(self, vec: LoRaVector, deltas: dict[int, float]) -> LoRaVector:
        """Shift a vector along specific basis dimensions.

        Adds ``delta * scale * basis_k`` for each specified dimension.
        This is lossless — components orthogonal to the basis are preserved.

        Args:
            vec: The vector to shift.
            deltas: Mapping of dimension index to shift amount (in coordinate
                space). E.g. ``{0: +0.5, 4: -0.3}`` shifts along PC1 and PC5.

        Returns:
            The shifted LoRAVector.
        """
        for dim in deltas:
            if dim < 0 or dim >= self.n_dims:
                raise ValueError(f"Dimension {dim} out of range [0, {self.n_dims})")

        result = vec
        for dim, delta in deltas.items():
            s = self._scale[dim].item() if self._scale is not None else 1.0
            result = result + ((delta * s) * self._basis[dim])
        return result
