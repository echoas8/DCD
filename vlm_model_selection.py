from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class VLMModelProfile:
    name: str
    quality_score: float
    speed_score: float
    cost_score: float
    supports_ocr: bool = False
    supports_grounding: bool = False
    supports_video: bool = False


DEFAULT_VLM_MODELS: tuple[VLMModelProfile, ...] = (
    VLMModelProfile(
        name="gpt-4o-mini",
        quality_score=0.82,
        speed_score=0.94,
        cost_score=0.95,
        supports_ocr=True,
        supports_grounding=False,
        supports_video=False,
    ),
    VLMModelProfile(
        name="gpt-4o",
        quality_score=0.96,
        speed_score=0.78,
        cost_score=0.62,
        supports_ocr=True,
        supports_grounding=True,
        supports_video=True,
    ),
    VLMModelProfile(
        name="gpt-4.1",
        quality_score=0.93,
        speed_score=0.72,
        cost_score=0.67,
        supports_ocr=True,
        supports_grounding=True,
        supports_video=False,
    ),
)


_PRIORITY_WEIGHTS: Mapping[str, tuple[float, float, float]] = {
    "balanced": (0.45, 0.35, 0.20),
    "quality": (0.70, 0.20, 0.10),
    "speed": (0.20, 0.70, 0.10),
    "cost": (0.20, 0.20, 0.60),
}


def select_vlm_model(
    *,
    require_ocr: bool = False,
    require_grounding: bool = False,
    require_video: bool = False,
    priority: str = "balanced",
    available_models: Iterable[VLMModelProfile] | None = None,
) -> str:
    if priority not in _PRIORITY_WEIGHTS:
        raise ValueError(
            f"Unsupported priority '{priority}'. Expected one of {tuple(_PRIORITY_WEIGHTS)}."
        )

    candidates = tuple(available_models or DEFAULT_VLM_MODELS)
    filtered = tuple(
        model
        for model in candidates
        if (not require_ocr or model.supports_ocr)
        and (not require_grounding or model.supports_grounding)
        and (not require_video or model.supports_video)
    )
    if not filtered:
        raise ValueError("No available VLM model satisfies the requested capabilities.")

    quality_weight, speed_weight, cost_weight = _PRIORITY_WEIGHTS[priority]

    def score(model: VLMModelProfile) -> float:
        return (
            model.quality_score * quality_weight
            + model.speed_score * speed_weight
            + model.cost_score * cost_weight
        )

    return max(filtered, key=score).name
