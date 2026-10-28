"""ISBN-10 and ISBN-13 validation, checksum calculation, and formatting utilities."""

from __future__ import annotations

import re

_ISBN_CLEAN_RE = re.compile(r"[- \t]")


def clean_isbn(isbn: str) -> str:
    """Strip hyphens, spaces, and whitespace, converting lowercase 'x' to uppercase."""
    cleaned = _ISBN_CLEAN_RE.sub("", isbn.strip())
    return cleaned.upper()


def calculate_isbn13_check_digit(digits12: str) -> str:
    """Calculate the 13th check digit for a 12-digit ISBN prefix using Modulo 10."""
    if len(digits12) != 12 or not digits12.isdigit():
        raise ValueError(f"Expected 12 decimal digits, got '{digits12}'")
    total = sum(int(ch) * (1 if idx % 2 == 0 else 3) for idx, ch in enumerate(digits12))
    check = (10 - (total % 10)) % 10
    return str(check)


def calculate_isbn10_check_digit(digits9: str) -> str:
    """Calculate the 10th check digit for a 9-digit ISBN prefix using Modulo 11."""
    if len(digits9) != 9 or not digits9.isdigit():
        raise ValueError(f"Expected 9 decimal digits, got '{digits9}'")
    total = sum(int(ch) * (10 - idx) for idx, ch in enumerate(digits9))
    rem = (11 - (total % 11)) % 11
    if rem == 10:
        return "X"
    return str(rem)


def validate_isbn13(isbn: str) -> bool:
    """Validate 13-digit ISBN with Modulo 10 checksum."""
    cleaned = clean_isbn(isbn)
    if len(cleaned) != 13 or not cleaned.isdigit():
        return False
    if not (cleaned.startswith("978") or cleaned.startswith("979")):
        return False
    expected_check = calculate_isbn13_check_digit(cleaned[:12])
    return cleaned[12] == expected_check


def validate_isbn10(isbn: str) -> bool:
    """Validate 10-digit ISBN with Modulo 11 checksum."""
    cleaned = clean_isbn(isbn)
    if len(cleaned) != 10:
        return False
    if not cleaned[:9].isdigit():
        return False
    if cleaned[9] not in "0123456789X":
        return False
    expected_check = calculate_isbn10_check_digit(cleaned[:9])
    return cleaned[9] == expected_check


def validate_isbn(isbn: str) -> bool:
    """Validate either an ISBN-10 or ISBN-13 string."""
    cleaned = clean_isbn(isbn)
    if len(cleaned) == 13:
        return validate_isbn13(cleaned)
    if len(cleaned) == 10:
        return validate_isbn10(cleaned)
    return False


def format_isbn(isbn: str) -> str:
    """Format an ISBN-13 or ISBN-10 into standard hyphenated representation."""
    cleaned = clean_isbn(isbn)
    if not validate_isbn(cleaned):
        return isbn.strip()

    if len(cleaned) == 13:
        # Standard Chinese / International 5-element hyphenation:
        # E.g. 978-7-XXX-XXXXX-X for China (Group 7)
        if cleaned.startswith("9787") or cleaned.startswith("9797"):
            return f"{cleaned[:3]}-{cleaned[3]}-{cleaned[4:7]}-{cleaned[7:12]}-{cleaned[12]}"
        return f"{cleaned[:3]}-{cleaned[3:5]}-{cleaned[5:9]}-{cleaned[9:12]}-{cleaned[12]}"

    if len(cleaned) == 10:
        # ISBN-10: Group-Publisher-Title-Check
        return f"{cleaned[:1]}-{cleaned[1:4]}-{cleaned[4:9]}-{cleaned[9]}"

    return cleaned
