"""
Exception classes for the report_generator component.
"""


class ConfigValidationError(Exception):
    """Error raised when ReportConfig validation fails."""
    pass


class DataSourceError(Exception):
    """Error raised when a data source is unavailable."""
    pass


class EmptyReportError(Exception):
    """Error raised when no task types are found for report generation."""
    pass


class FormattingError(Exception):
    """Error raised when report formatting fails."""
    pass
