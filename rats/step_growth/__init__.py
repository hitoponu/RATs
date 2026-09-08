"""Step-growth arm: oracle step judgement, step-level skill credit and
step-level skill extraction for RATS play.

Everything in this package is inert unless ``RATS_STEP_GROWTH=1``. The
lifelong loop talks to it only through :class:`StepGrowthController`.
"""

from rats.step_growth.config import StepGrowthConfig, load_config, step_growth_enabled

__all__ = ["StepGrowthConfig", "load_config", "step_growth_enabled"]
