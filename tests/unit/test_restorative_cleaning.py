"""Unit tests for Restorative Cleaning and OCR Denoising."""

from ubt.core.cleaners.lnds_pruner import strip_textbook_ocr_artifacts
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


def test_strip_ocr_mangled_cengage_legal_boilerplate() -> None:
    """Validate that OCR-corrupted publisher disclaimers (Ce ngage, Leaming, conten!) are stripped."""
    corrupted_sample = (
        "Cognitive fie bodice A AVW ON, OED VX AAA \\ VAI f ROBERT J. KARIN STERNBERG = STERNBERG ~ e CENGAGE Learning "
        "Copyright 2017 Ce ngage Leaming, All Rights Resa yed Maya ot be copie ned, or duplicated, in whole "
        "Editorial review has deemed that any suppressed conten! nit de jot materially al a ihe rall learning experi "
        "5 Cangas eal ne some third party c ERPE suppressed from the eBook and/or eChapter(s). "
        "the ight to remove additional cı antent at auy Gi e if subsequent rights res require it."
    )

    cleaned = strip_textbook_ocr_artifacts(corrupted_sample)

    # 1. Corrupted legal boilerplate must be completely gone
    assert "Copyright 2017" not in cleaned
    assert "Ce ngage" not in cleaned
    assert "Leaming" not in cleaned
    assert "Editorial review has deemed" not in cleaned
    assert "suppressed" not in cleaned
    assert "duplicated" not in cleaned

    # 2. Stray backslash artifacts removed (contract: backslash runs
    # between whitespace are OCR noise; mid-prose '=' / '~' single symbols are
    # deliberately PRESERVED so 'x = 0.5' style content is never corrupted)
    assert "\\" not in cleaned
    assert "VAI" in cleaned
    assert "STERNBERG" in cleaned

    # 3. Authentic author and title information remains
    assert "Cognitive" in cleaned
    assert "STERNBERG" in cleaned
    assert "CENGAGE Learning" in cleaned


def test_router_prompt_injects_restorative_denoising_protocol() -> None:
    """Validate that the ModelRouter prompt incorporates the restorative denoising protocol."""
    provider = MockModelProvider(default_response="测试翻译")
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")

    source = "Chapter 1. Introduction to Cognitive Psychology."
    system_prompt, user_prompt = router.build_draft_prompt(
        source_text=source,
        target_lang="zh",
        genre_profile="textbook",
    )

    assert "Restorative Denoising" in system_prompt
    assert "DO NOT transcribe or echo nonsensical OCR symbol strings" in system_prompt
    assert "Chapter 1" in user_prompt


def test_repair_prompt_injects_denoising_protocol() -> None:
    """Validate that the repair prompt instructs agents to suppress leaked OCR glyph noise."""
    provider = MockModelProvider(default_response="测试翻译")
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")

    system_prompt, user_prompt = router.build_repair_prompt(
        source_text="Some noisy source text",
        draft_text="Some draft text with \\ artifacts",
        error_flags=["OCR residue detected"],
        target_lang="zh",
    )

    assert "Fluency & Denoising" in system_prompt
    assert "Suppress any raw OCR glyph noise" in system_prompt


def test_strip_textbook_ocr_artifacts_does_not_mutilate_prose_words() -> None:
    """Verify that words following backslashes or punctuation are never deleted."""
    text = "Please examine the \\ note carefully and visit C:\\Users\\Data."
    cleaned = strip_textbook_ocr_artifacts(text)
    assert "note" in cleaned
    assert "Users" in cleaned
    assert "Data" in cleaned


def test_draft_prompt_forbids_invented_latex() -> None:
    r"""Rule 4 must not teach models to LaTeX-ify plain labels (chapter-1
    b0004: repair turned "(2D)" into $2\mathrm{D}$, literal garbage in the
    overlay). The few-shot example itself must be renderer-compilable."""
    provider = MockModelProvider(default_response="测试翻译")
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")

    system_prompt, _ = router.build_draft_prompt(
        source_text="The two-dimensional (2D) structure.",
        target_lang="zh",
    )

    assert "NEVER invent LaTeX" in system_prompt
    assert "2D/3D" in system_prompt
    assert "$V_{tm}$" in system_prompt
    assert "$V_{\\text{tm}}$" not in system_prompt


def test_repair_prompt_forbids_invented_latex() -> None:
    provider = MockModelProvider(default_response="测试翻译")
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")

    system_prompt, _ = router.build_repair_prompt(
        source_text="The two-dimensional (2D) structure.",
        draft_text="二维（2D）结构。",
        error_flags=["Undelimited math: source carries math symbols"],
        target_lang="zh",
    )

    assert "never invent LaTeX commands" in system_prompt
    assert "reproduce those exactly" in system_prompt
