class OmniBoundError(Exception):
    """Base exception for OmniBound application."""
    pass


class UnreachableURLError(OmniBoundError):
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to fetch document from URL '{url}': {reason}")


class LLMTimeoutError(OmniBoundError):
    def __init__(self, operation: str):
        super().__init__(f"LLM request timed out during operation: {operation}")
