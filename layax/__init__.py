"""layax -- late-interaction option heads and a Learn-then-Test abstention gate for Laya.

Two model-layer changes to Laya (https://github.com/NandhaKishorM/laya, Apache 2.0),
each aimed at a weakness its own README documents:

1. **Late interaction** removes the shared option token budget, so high-cardinality
   label sets stop being truncated to a few tokens per label (Banking77: upstream's best
   arm scores 0.453 on the test rows layax scores 0.908 on).
2. **An abstention gate**: Learn-then-Test on the temperature-scaled max-softmax. A
   learned competence head is also here, off by default: it lost to max-softmax in
   every run (README, "What did not work").

Neither technique is new in itself. Applying them to Laya's RLCD-trained typed decision
heads is the part worth measuring, and nothing here ships a benchmarked claim -- run
``layax.pipeline.run`` and read your own numbers.
"""

__version__ = "0.1.0"

from .competence import (  # noqa: F401
    CompetenceHead,
    CompetenceModel,
    aurc,
    evaluate_selective,
    fit_abstention_threshold,
    risk_coverage_curve,
)
from .config import CompConfig, LIConfig, RunConfig  # noqa: F401
from .li_head import (  # noqa: F401
    LateInteractionDecisionModel,
    OptionCache,
    download_laya,
)
from .runtime import LayaxAgent, load_base  # noqa: F401

__all__ = [
    "LIConfig", "CompConfig", "RunConfig",
    "LateInteractionDecisionModel", "OptionCache", "download_laya",
    "LayaxAgent", "load_base",
    "CompetenceHead", "CompetenceModel", "aurc", "risk_coverage_curve",
    "fit_abstention_threshold", "evaluate_selective",
    "__version__",
]
