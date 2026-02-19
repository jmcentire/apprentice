"""
WOS PII Patterns — extends PIITokenizer with Wander Operations System patterns.

Handles tokenization of: Wander booking IDs, Stripe payment tokens,
PMS reservation IDs (Guesty/Hostaway/OwnerRez/Streamline), and US addresses.
"""

import re
from typing import Final

from apprentice.pii_tokenizer import PIITokenizer, PIITokenizerConfig
from apprentice.plugin_registry import PluginRegistrySet


# ===========================================================================
# Built-in entity types from PIITokenizer
# ===========================================================================

BUILTIN_ENTITY_TYPES: Final[list[str]] = ["email", "phone", "ssn", "credit_card"]


# ===========================================================================
# WOS-specific PII patterns (module-level constant, never mutated)
# ===========================================================================

WOS_CUSTOM_PATTERNS: Final[dict[str, str]] = {
    # Wander booking IDs: W-AB12CD34
    "wander_booking_id": r"\bW-[A-Z0-9]{8}\b",

    # Stripe payment intent IDs: pi_ + 24+ alphanumeric chars
    "stripe_payment_intent": r"\bpi_[a-zA-Z0-9]{24,}\b",

    # Stripe charge IDs: ch_ + 24+ alphanumeric chars
    "stripe_charge": r"\bch_[a-zA-Z0-9]{24,}\b",

    # Stripe refund IDs: re_ + 24+ alphanumeric chars
    "stripe_refund": r"\bre_[a-zA-Z0-9]{24,}\b",

    # Stripe customer IDs: cus_ + 14+ alphanumeric chars
    "stripe_customer": r"\bcus_[a-zA-Z0-9]{14,}\b",

    # Guesty reservation IDs: 24 hex characters (MongoDB ObjectId)
    "guesty_reservation_id": r"\b[0-9a-f]{24}\b",

    # Hostaway numeric reservation IDs: 6-10 digit numbers
    "hostaway_reservation_id": r"\b\d{6,10}\b",

    # OwnerRez reservation IDs: OR- + 6+ alphanumeric chars
    "ownerrez_reservation_id": r"\bOR-[A-Za-z0-9]{6,}\b",

    # Streamline reservation IDs: SL + 8+ digits
    "streamline_reservation_id": r"\bSL\d{8,}\b",

    # US street addresses: number + street name + suffix
    "us_street_address": (
        r"\b\d{1,6}\s+[A-Za-z0-9.\s]+?"
        r"(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Ln|Lane|Rd|Road|Way|Ct|Court|Pl|Place|Cir|Circle)\b"
    ),
}

# Validate all patterns at import time (fail-fast)
for _name, _pattern in WOS_CUSTOM_PATTERNS.items():
    try:
        re.compile(_pattern)
    except re.error as e:
        raise re.error(f"WOS pattern '{_name}' failed to compile: {e}")


# ===========================================================================
# Factory Function
# ===========================================================================

def create_wos_pii_tokenizer(
    enabled_entity_types: list[str] | None = None,
    additional_patterns: dict[str, str] | None = None,
    token_format: str = "<PII:{type}:{hash}>",
) -> PIITokenizer:
    """Create a PIITokenizer pre-configured with all WOS-specific patterns.

    Args:
        enabled_entity_types: Entity types to enable. Defaults to all builtins + WOS.
        additional_patterns: Extra patterns merged on top of WOS defaults.
        token_format: Token format string (must contain {type} and {hash}).

    Returns:
        Configured PIITokenizer ready for use as middleware.
    """
    if "{type}" not in token_format or "{hash}" not in token_format:
        raise ValueError("token_format must contain both {type} and {hash} placeholders")

    if additional_patterns is not None:
        for name, pattern in additional_patterns.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise re.error(f"Additional pattern '{name}' failed to compile: {e}")

    if enabled_entity_types is not None:
        for et in enabled_entity_types:
            if not isinstance(et, str) or not et:
                raise ValueError("All enabled_entity_types must be non-empty strings")

    if enabled_entity_types is None:
        enabled_entity_types = BUILTIN_ENTITY_TYPES + list(WOS_CUSTOM_PATTERNS.keys())

    merged_patterns = dict(WOS_CUSTOM_PATTERNS)
    if additional_patterns is not None:
        merged_patterns.update(additional_patterns)

    config = PIITokenizerConfig(
        enabled_entity_types=enabled_entity_types,
        custom_patterns=merged_patterns,
        token_format=token_format,
    )

    return PIITokenizer(config)


# ===========================================================================
# Plugin Registration
# ===========================================================================

def register_wos_pii_tokenizer(registry: PluginRegistrySet) -> None:
    """Register the WOS PII tokenizer in the 'middleware' domain.

    Raises DuplicatePluginError if already registered.
    """
    middleware_registry = registry.get_registry("middleware")
    middleware_registry.register("wos_pii_tokenizer", create_wos_pii_tokenizer)


# ===========================================================================
# Validation / Health Check
# ===========================================================================

def validate_wos_patterns() -> dict[str, bool]:
    """Compile each WOS pattern and return name -> success mapping."""
    results = {}
    for name, pattern in WOS_CUSTOM_PATTERNS.items():
        try:
            re.compile(pattern)
            results[name] = True
        except re.error:
            results[name] = False
    return results


__all__ = [
    "WOS_CUSTOM_PATTERNS",
    "BUILTIN_ENTITY_TYPES",
    "create_wos_pii_tokenizer",
    "register_wos_pii_tokenizer",
    "validate_wos_patterns",
]
