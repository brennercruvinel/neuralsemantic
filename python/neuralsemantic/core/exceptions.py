"""Exception definitions for Neural Semantic Compiler."""


class NSCException(Exception):
    """Base exception for Neural Semantic Compiler."""
    pass


class CompressionError(NSCException):
    """Error during compression operation."""
    pass


class PatternConflictError(NSCException):
    """Pattern conflict detected."""
    pass


class QualityError(NSCException):
    """Quality threshold not met."""
    pass


class DatabaseError(NSCException):
    """Database operation error."""
    pass


class VectorStoreError(NSCException):
    """Vector store operation error."""
    pass


class ConfigurationError(NSCException):
    """Configuration error."""
    pass


class PatternLearningError(NSCException):
    """Pattern learning error."""
    pass