"""Three-mode BCI pipeline package.

Subcommands live in pipeline.cli:
    collect    record AutoMove session(s) per participant under data/subNN/
    train      train the base 4-class MI + blink model from data/sub*/ CSVs
    realtime   personalize + run real-time inference, streaming to Unity

Shared label taxonomy and marker codes are defined in pipeline.labels.
"""

__all__ = ["labels"]
