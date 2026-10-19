"""Stable errors shared by the CLI and experiment owners."""


class EvolutionError(Exception):
    """A bounded, user-actionable domain failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
