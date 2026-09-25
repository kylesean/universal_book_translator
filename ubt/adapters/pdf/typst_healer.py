"""Compiler-in-the-loop diagnostic-driven self-healing engine for Typst.

This module encapsulates the compilation, error parsing, and precision healing
ladder for Typst documents. Instead of heuristic pre-emptive regex guessing on
prose, it relies on the Typst compiler's structured stderr diagnostics (file,
line, column, error kind, and variable name) to apply surgical, localized repairs
and fail-closed content-preserving degradations.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
import time
from bisect import bisect_right
from collections.abc import Callable, Sequence
from pathlib import Path

from ubt.adapters.pdf.typst_constants import (
    _MAX_MATH_VERIFY_ROUNDS,
    _TYPST_HEAL_BUDGET_S,
    _TYPST_VERSION_RE,
)
from ubt.adapters.pdf.typst_diagnostics import (
    TypstErrorKind,
    error_line_numbers,
    parse_typst_stderr,
)
from ubt.adapters.pdf.typst_fragments import (
    _INLINE_CALL_COLLISION_RE,
    _TYPST_KEYWORDS_AND_STOPWORDS,
    _decouple_inline_box_calls,
    _stage_image_assets,
)
from ubt.core.env import subprocess_env
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)

# _TYPST_VERSION_RE / _TYPST_HEAL_BUDGET_S / _MAX_MATH_VERIFY_ROUNDS come
# from typst_constants — the single definition site shared with typst_reconstructor.


def _quote_unknown_var_on_line(line: str, var: str) -> tuple[str, bool]:
    """Quote an unknown variable within math mode ($...$) on a single line."""
    if var == "dx":
        repaired = re.sub(r"\bdx\b", "d x", line)
        return repaired, repaired != line
    if var.lower() in _TYPST_KEYWORDS_AND_STOPWORDS:
        return line, False
    if var.startswith("_") or '"' in var or "'" in var or "\\" in var:
        return line, False
    if var.lower() in {
        "assets",
        "image",
        "figure",
        "caption",
        "width",
        "height",
        "png",
        "jpg",
        "jpeg",
        "svg",
    }:
        return line, False

    if "$" not in line:
        return line, False

    pattern = re.compile(rf'(?<!["a-zA-Z0-9]){re.escape(var)}(?!["a-zA-Z0-9])')

    def _replace_in_math(m: re.Match[str]) -> str:
        math_content = m.group(1)
        parts = re.split(r'(?<!\\)"', math_content)
        if len(parts) % 2 == 1:
            for i in range(0, len(parts), 2):
                parts[i] = pattern.sub(f'"{var}"', parts[i])
            return "$" + '"'.join(parts) + "$"
        return m.group(0)

    new_line, count = re.subn(r"\$([^$]+)\$", _replace_in_math, line)
    return new_line, count > 0


def _heal_persistent_comment_error(
    lines: list[str], err_idx: int, audit: list[str] | None = None
) -> bool:
    """Break retry loop deadlock when Typst reports error on an already commented line.

    This happens when an unclosed delimiter ($ or ()/{}/[]) or unclosed string (")
    was opened on an earlier line, making the Typst lexer treat subsequent lines as
    part of an unclosed construct where // line comments are ignored.

    ``audit`` collects a record for every line this nullifies: both the culprit and
    the error line are translated content that disappears from the delivered PDF, so
    without a handle on it the loss is invisible outside the log stream.
    """
    logger.warning(
        "Typst persistent error on commented line %d; searching for unclosed construct above",
        err_idx + 1,
    )
    culprit_idx: int | None = None

    # Look back up to 100 lines for an unclosed string or delimiter
    for k in range(err_idx - 1, max(-1, err_idx - 100), -1):
        line = lines[k]

        # Check for unclosed string literal (odd number of unescaped quotes)
        unescaped_quotes = len(re.findall(r'(?<!\\)(?:\\\\)*"', line))
        if unescaped_quotes % 2 != 0:
            culprit_idx = k
            break

        # Check for unclosed raw code backtick block (odd number of unescaped `)
        unescaped_backticks = len(re.findall(r"(?<!\\)(?:\\\\)*`", line))
        if unescaped_backticks % 2 != 0:
            culprit_idx = k
            break

        # Check for unclosed math block (odd number of unescaped $)
        unescaped_dollars = len(re.findall(r"(?<!\\)(?:\\\\)*\$", line))
        if unescaped_dollars % 2 != 0:
            culprit_idx = k
            break

        # Check for unclosed parentheses, brackets, or braces
        if (
            line.count("(") > line.count(")")
            or line.count("[") > line.count("]")
            or line.count("{") > line.count("}")
        ):
            culprit_idx = k
            break

    # If no delimiter imbalance found, pick nearest syntax-bearing line above err_idx
    if culprit_idx is None:
        for k in range(err_idx - 1, max(-1, err_idx - 20), -1):
            cand = lines[k].strip()
            if (
                cand
                and not lines[k].lstrip().startswith("//")
                and any(c in cand for c in ('"', "$", "\\", "(", "[", "{", "`"))
            ):
                culprit_idx = k
                break

    modified = False
    if culprit_idx is not None and culprit_idx != err_idx:
        logger.warning(
            "Nullifying culprit line %d for persistent error at line %d",
            culprit_idx + 1,
            err_idx + 1,
        )
        if audit is not None:
            audit.append(f"line {culprit_idx + 1}: {lines[culprit_idx][:80]}")
        safe_line = (
            lines[culprit_idx]
            .replace('"', "'")
            .replace("$", " ")
            .replace("`", "'")
            .replace("\n", " ")
        )
        lines[culprit_idx] = f"// [UBT_SYNTAX_FALLBACK_REMOVED] #v(0pt) // {safe_line}"
        modified = True

    # Also neutralize quotes, math dollars, and backticks on the error line itself
    safe_err = (
        lines[err_idx].replace('"', "'").replace("$", " ").replace("`", "'").replace("\n", " ")
    )
    if safe_err != lines[err_idx] or not lines[err_idx].startswith(
        "// [UBT_SYNTAX_FALLBACK_REMOVED]"
    ):
        if audit is not None and not lines[err_idx].startswith("// [UBT_SYNTAX_FALLBACK_REMOVED]"):
            audit.append(f"line {err_idx + 1}: {lines[err_idx][:80]}")
        lines[err_idx] = f"// [UBT_SYNTAX_FALLBACK_REMOVED] #v(0pt) // {safe_err}"
        modified = True

    return modified


def _degrade_failing_line(line: str) -> tuple[str, bool]:
    """Content-preserving last resort for a line Typst refuses to compile.

    Inline math is demoted to a visible code span (``$x$`` -> `` `$x$` ``) so the
    translated prose survives and only the unrenderable notation changes form.
    Commenting the whole line out is the final fallback, taken only when there
    is no inline math left to keep.

    Returns ``(new_line, removed)``; ``removed`` is True only for the comment
    fallback, so the audit ledger records content loss and never a form change.
    """

    def _to_code_span(match: re.Match[str]) -> str:
        body = match.group(1).replace("`", "'")
        return f"`${body}$`"

    if "$" in line:
        degraded = re.sub(r"(?<!`)\$([^$\n]+)\$(?!`)", _to_code_span, line)
        if degraded != line:
            return degraded, False
    safe = line.replace("`", "'").replace('"', "'")
    return f"// [UBT_SYNTAX_FALLBACK] {safe}", True


class TypstDiagnosticHealer:
    """Diagnostic-driven compiler loop with precision self-healing and degradation."""

    def __init__(
        self,
        typst_binary: str = "typst",
        max_attempts: int = 3,
        budget_s: float = _TYPST_HEAL_BUDGET_S,
    ) -> None:
        self.typst_binary = shutil.which(typst_binary) or typst_binary
        self.max_attempts = max_attempts
        self.budget_s = budget_s
        self.last_syntax_fallbacks: list[str] = []
        self._compiler_version: str | None = None
        self._compiler_version_probed: bool = False

    def is_compiler_available(self) -> bool:
        """Check if Typst CLI compiler is available on PATH."""
        return shutil.which(self.typst_binary) is not None

    def compiler_version(self) -> str | None:
        """Report the Typst compiler version (e.g. '0.15.1')."""
        if self._compiler_version_probed:
            return self._compiler_version
        self._compiler_version_probed = True
        if not self.is_compiler_available():
            self._compiler_version = None
            return None
        try:
            proc = subprocess.run(
                [self.typst_binary, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=5,
                env=subprocess_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Typst version check failed: %s", exc)
            self._compiler_version = None
            return None
        if proc.returncode != 0:
            self._compiler_version = None
            return None
        match = _TYPST_VERSION_RE.search(proc.stdout or proc.stderr)
        self._compiler_version = match.group(1) if match else None
        return self._compiler_version

    def run_typst(
        self,
        cmd: list[str],
        timeout: float = 120.0,
    ) -> subprocess.CompletedProcess[str]:
        """Run a Typst command, converting timeout/OS errors to DocumentParseError."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout,
                env=subprocess_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise DocumentParseError(f"Typst compilation timed out after {int(timeout)}s") from exc
        except OSError as exc:
            raise DocumentParseError(
                f"Typst compiler '{self.typst_binary}' could not be executed: {exc}"
            ) from exc

    def heal_and_compile(
        self,
        typ_source: str,
        output_pdf_path: Path | str,
        *,
        run_typst_override: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> Path:
        """Compile Typst markup to PDF with compiler-in-the-loop diagnostic self-healing."""
        out_pdf = Path(output_pdf_path)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        self.last_syntax_fallbacks = []

        typ_file = out_pdf.with_suffix(".typ")
        typ_source = _stage_image_assets(typ_source, typ_file)
        typ_source = _decouple_inline_box_calls(typ_source)
        typ_file.write_text(typ_source, encoding="utf-8")

        if not self.is_compiler_available() and run_typst_override is None:
            raise DocumentParseError(
                f"Typst compiler binary '{self.typst_binary}' not found on system PATH.\n"
                f"Source markup has been saved to '{typ_file}'.\n"
                "Install Typst via: 'cargo install --locked typst-cli' or download from https://github.com/typst/typst"
            )

        cmd = [
            self.typst_binary,
            "compile",
            "--root",
            str(typ_file.parent),
            str(typ_file),
            str(out_pdf),
        ]
        runner = run_typst_override or self.run_typst
        result = runner(cmd)

        if result.returncode == 0:
            return out_pdf

        heal_started = time.monotonic()
        for attempt in range(self.max_attempts):
            if result.returncode == 0:
                break
            if time.monotonic() - heal_started > self.budget_s:
                logger.warning(
                    "Typst self-healing budget (%.0fs) exhausted after %d attempt(s); "
                    "falling back to emergency sweep",
                    self.budget_s,
                    attempt,
                )
                break

            stderr = result.stderr
            current_source = typ_file.read_text(encoding="utf-8")
            fixed_source = current_source
            modified = False

            diagnostics = parse_typst_stderr(stderr)

            # Branch A: Handle unknown variables on the specific error lines the
            # compiler named. No global sweep: rewriting the variable across
            # every math block in the file would also rewrite unrelated formulas
            # the compiler never complained about.
            var_errs = [
                (diagnostic.variable, diagnostic.line)
                for diagnostic in diagnostics
                if diagnostic.kind is TypstErrorKind.UNKNOWN_VARIABLE and diagnostic.variable
            ]
            if var_errs:
                lines = fixed_source.splitlines()
                for var, line_num in set(var_errs):
                    if 1 <= line_num <= len(lines):
                        idx = line_num - 1
                        orig = lines[idx]
                        if orig.startswith("//") or orig.lstrip().startswith("#"):
                            continue
                        repaired, mod = _quote_unknown_var_on_line(orig, var)
                        if mod:
                            lines[idx] = repaired
                            modified = True
                if modified:
                    fixed_source = "\n".join(lines)

            # Branch B: Line-level delimiter/syntax errors from Typst compiler
            err_lines = error_line_numbers(stderr)
            if err_lines and not modified:
                lines = fixed_source.splitlines()
                for line_num in set(err_lines):
                    if 1 <= line_num <= len(lines):
                        idx = line_num - 1
                        orig_line = lines[idx]
                        if orig_line.startswith("//"):
                            if _heal_persistent_comment_error(
                                lines, idx, self.last_syntax_fallbacks
                            ):
                                modified = True
                            continue

                        if attempt < 2:
                            # Gentle repair: balance odd $, decouple #box calls, or wrap bare scripts
                            if orig_line.count("$") % 2 != 0:
                                repaired = orig_line.rstrip() + " $"
                                if repaired != orig_line:
                                    lines[idx] = repaired
                                    modified = True
                            elif _INLINE_CALL_COLLISION_RE.search(orig_line):
                                repaired = _decouple_inline_box_calls(orig_line)
                                if repaired != orig_line:
                                    lines[idx] = repaired
                                    modified = True
                            elif "_{" in orig_line or "^{" in orig_line:
                                repaired = re.sub(
                                    r"([a-zA-Z0-9\u0370-\u03ff\u2070-\u209f])_\{([^{}]+)\}",
                                    r'$\1_"\2"$',
                                    orig_line,
                                )
                                repaired = re.sub(
                                    r"([a-zA-Z0-9\u0370-\u03ff\u2070-\u209f])\^\{([^{}]+)\}",
                                    r"$\1^(\2)$",
                                    repaired,
                                )
                                if repaired != orig_line:
                                    lines[idx] = repaired
                                    modified = True
                        else:
                            # Fail-closed fallback: demote inline math to visible code span;
                            # comment line out only when nothing renderable is left to preserve.
                            degraded, removed = _degrade_failing_line(orig_line)
                            logger.warning(
                                "Typst compilation failure at line %d (attempt %d); fallback (%s)",
                                line_num,
                                attempt,
                                "line removed" if removed else "inline math degraded",
                            )
                            lines[idx] = degraded
                            if removed:
                                self.last_syntax_fallbacks.append(
                                    f"line {line_num}: {orig_line[:80]}"
                                )
                            modified = True
                if modified:
                    fixed_source = "\n".join(lines)

            if modified and fixed_source != current_source:
                typ_file.write_text(fixed_source, encoding="utf-8")
                result = runner(cmd)
            else:
                # If unhandled syntax errors persist without modifications, force degradation
                if err_lines:
                    lines = fixed_source.splitlines()
                    for line_num in set(err_lines):
                        if 1 <= line_num <= len(lines):
                            idx = line_num - 1
                            if not lines[idx].startswith("//"):
                                original_line = lines[idx]
                                degraded, removed = _degrade_failing_line(original_line)
                                logger.warning(
                                    "Typst persistent error at line %d; fallback (%s)",
                                    line_num,
                                    "line removed" if removed else "inline math degraded",
                                )
                                lines[idx] = degraded
                                if removed:
                                    self.last_syntax_fallbacks.append(
                                        f"line {line_num}: {original_line[:80]}"
                                    )
                                modified = True
                            elif _heal_persistent_comment_error(
                                lines, idx, self.last_syntax_fallbacks
                            ):
                                modified = True
                    if modified:
                        fixed_source = "\n".join(lines)
                        typ_file.write_text(fixed_source, encoding="utf-8")
                        result = runner(cmd)
                    else:
                        break
                else:
                    break

        if result.returncode != 0:
            # Emergency fail-closed sweep: comment out all residual error lines reported by Typst
            err_lines = error_line_numbers(result.stderr)
            if err_lines:
                lines = typ_file.read_text(encoding="utf-8").splitlines()
                modified = False
                for line_num in set(err_lines):
                    if 1 <= line_num <= len(lines):
                        idx = line_num - 1
                        if not lines[idx].startswith("//"):
                            original_line = lines[idx]
                            degraded, removed = _degrade_failing_line(original_line)
                            logger.warning(
                                "Typst emergency fallback on line %d (%s)",
                                line_num,
                                "line removed" if removed else "inline math degraded",
                            )
                            lines[idx] = degraded
                            if removed:
                                self.last_syntax_fallbacks.append(
                                    f"line {line_num}: {original_line[:80]}"
                                )
                            modified = True
                        elif _heal_persistent_comment_error(lines, idx, self.last_syntax_fallbacks):
                            modified = True
                if modified:
                    typ_file.write_text("\n".join(lines), encoding="utf-8")
                    result = runner(cmd)

        if result.returncode != 0:
            raise DocumentParseError(
                f"Typst compilation failed (exit code {result.returncode}):\n{result.stderr}"
            )

        if self.last_syntax_fallbacks:
            logger.warning(
                "Typst syntax fallback removed %d translated line(s) from the delivered PDF: %s%s",
                len(self.last_syntax_fallbacks),
                "; ".join(self.last_syntax_fallbacks[:5]),
                " …" if len(self.last_syntax_fallbacks) > 5 else "",
            )
        return out_pdf

    async def heal_and_compile_async(
        self,
        typ_source: str,
        output_pdf_path: Path | str,
        *,
        run_typst_override: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> Path:
        """Non-blocking compilation of Typst source executed in a threadpool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.heal_and_compile(
                typ_source,
                output_pdf_path,
                run_typst_override=run_typst_override,
            ),
        )

    def probe_single_math(self, math_line: str) -> bool:
        """Compile one math line alone; True iff it compiles."""
        try:
            clean_math = re.sub(r"#footnote\[.*?\]", "", math_line)
            with tempfile.TemporaryDirectory(prefix="ubt-math-probe-") as tmp:
                probe = Path(tmp) / "probe.typ"
                probe.write_text(
                    '#set page(width: auto, height: 842pt, margin: 1cm)\n#set math.equation(numbering: "(1)")\n\n'
                    + clean_math
                    + "\n",
                    encoding="utf-8",
                )
                proc = subprocess.run(
                    [self.typst_binary, "compile", str(probe), str(Path(tmp) / "probe.pdf")],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=120,
                    env=subprocess_env(),
                )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Math verify probe could not run (%s); keeping line", exc)
            return True
        return proc.returncode == 0

    def probe_math_failures(self, probe_lines: list[str], header_len: int) -> set[int]:
        """Compile probe lines; return failing math-line indices (probe-relative)."""
        try:
            with tempfile.TemporaryDirectory(prefix="ubt-math-probe-") as tmp:
                probe = Path(tmp) / "probe.typ"
                probe.write_text("\n".join(probe_lines) + "\n", encoding="utf-8")
                proc = subprocess.run(
                    [self.typst_binary, "compile", str(probe), str(Path(tmp) / "probe.pdf")],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=120,
                    env=subprocess_env(),
                )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Math verify probe could not run (%s); skipping", exc)
            return set()
        if proc.returncode == 0:
            return set()
        failing: set[int] = set()
        # Map each compiler error's *physical* line back to the probe entry that
        # owns it. An emitted formula legitimately spans several lines (author
        # ``\\`` breaks, split long equations), and the old one-line-per-entry
        # arithmetic mis-attributed everything after the first multi-line entry:
        # healthy formulas got degraded — their text swapped for the source
        # graphic — while the actual culprit survived.
        starts: list[int] = []
        pos = 0
        for entry in probe_lines[header_len:]:
            starts.append(pos)
            pos += entry.count("\n") + 1
        starts.append(pos)
        for line_no in error_line_numbers(proc.stderr):
            phys = (line_no - 1) - header_len
            if phys < 0:
                continue
            entry_index = bisect_right(starts, phys) - 1
            if 0 <= entry_index < len(starts) - 1:
                failing.add(entry_index)
        return failing

    def verify_math_lines(
        self,
        lines: list[str],
        blocks: Sequence[IRBlock],
        degrade_line_fn: Callable[[list[str], int, dict[str, IRBlock]], None],
    ) -> int:
        """Batch compile-probe every math line; degrade failures.

        The caller only invokes this when Typst itself will typeset the math —
        the mathjax/image backends replace those lines wholesale, so probing
        them first would be discarded work.
        """
        math_idx = [
            i
            for i, ln in enumerate(lines)
            if ln.lstrip().startswith(("$", "#math.equation", "#block[#show math.equation"))
        ]
        if not math_idx:
            return 0
        if not self.is_compiler_available():
            return 0
        by_id = {b.id: b for b in blocks}
        degraded = 0
        # The batch rounds above are bounded by _MAX_MATH_VERIFY_ROUNDS, but the
        # two per-line tails that follow call probe_single_math (120s each) once
        # per surviving line, so on a book of N stubborn formulas they cost up to
        # 2·N·120s with no ceiling. Cap the whole verify pass by the same budget
        # the heal loop honours; past it we leave the rest untouched (fail-open,
        # matching probe_single_math's own give-up path).
        verify_deadline = time.monotonic() + self.budget_s
        for _ in range(_MAX_MATH_VERIFY_ROUNDS):
            probe_header = [
                "#set page(width: auto, height: auto)",
                '#set math.equation(numbering: "(1)")',
                "",
            ]
            probe_lines = probe_header + [lines[i] for i in math_idx]
            failing = self.probe_math_failures(probe_lines, len(probe_header))
            failing = {mi for mi in failing if mi < len(math_idx)}
            if not failing:
                return degraded
            for mi in sorted(failing, reverse=True):
                degrade_line_fn(lines, math_idx[mi], by_id)
                degraded += 1
            math_idx = [
                i
                for i, ln in enumerate(lines)
                if ln.lstrip().startswith(("$", "#math.equation", "#block[#show math.equation"))
            ]
            if not math_idx:
                return degraded

        for li in list(math_idx):
            if time.monotonic() > verify_deadline:
                logger.warning(
                    "Math verify: budget of %.0fs exhausted; leaving %d display line(s) unprobed",
                    self.budget_s,
                    len([x for x in math_idx if x >= li]),
                )
                break
            if self.probe_single_math(lines[li]):
                continue
            degrade_line_fn(lines, li, by_id)
            degraded += 1

        prose_math_idx = [
            i
            for i, ln in enumerate(lines)
            if "$" in ln
            and not ln.lstrip().startswith("$")
            and not ln.lstrip().startswith("//")
            and not ln.lstrip().startswith("#")
        ]
        for li in prose_math_idx:
            if time.monotonic() > verify_deadline:
                logger.warning(
                    "Math verify: budget of %.0fs exhausted; leaving inline math on "
                    "line %d and later unprobed",
                    self.budget_s,
                    li,
                )
                break
            if not self.probe_single_math(lines[li]):
                degrade_line_fn(lines, li, by_id)
                degraded += 1

        if degraded:
            logger.warning("Math verify: degraded %d lines to verbatim", degraded)
        return degraded
