"""Unit tests verifying the bidirectional multilingual translation architecture.

Covers:
1. BoilerplateCatalog declarative multilingual rules (EN, FR, DE, ES, JA, ZH, RU).
2. LanguagePairPolicy matrix, dynamic length expansion/contraction ratio bounds, and script identity gates.
3. FastPassFilter pair-aware 0-token evaluations for forward and reverse language pairs.
4. ModelRouter bidirectional prompt assembly for arbitrary (source, target) combinations.
5. LNDSPageCleaner and pruner multilingual dispatch.
"""

import pytest

from ubt.core.cleaners.boilerplate_catalog import BoilerplateCatalog
from ubt.core.cleaners.lnds_pruner import LNDSPageCleaner, strip_textbook_ocr_artifacts
from ubt.core.ir.models import FlowID, IRBlock
from ubt.core.language_profile import (
    get_pair_policy,
)
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter

# ---------------------------------------------------------------------------
# 1. BoilerplateCatalog & LNDS Pruner Multilingual Tests
# ---------------------------------------------------------------------------


def test_boilerplate_catalog_all_languages_supported() -> None:
    supported = BoilerplateCatalog.get_supported_languages()
    for code in ("en", "fr", "de", "es", "ja", "zh", "ru", "ko"):
        assert code in supported


# One row per language: the pruner's job is to drop the colophon lines and keep
# the chapter heading and the first real sentence. Written as seven functions,
# each 12 lines, this block was 90 lines of identical shape.
@pytest.mark.parametrize(
    "source_lang, text, keep, drop",
    [
        (
            "fr",
            "Chapitre Premier : La Révolution Industrielle.\n"
            "Tous droits réservés. Aucune partie de cet ouvrage ne peut être reproduite sans autorisation.\n"
            "Imprimé en France en 2020.\n"
            "Le développement économique a profondément modifié les structures sociales.",
            ["Chapitre Premier", "Le développement économique"],
            ["Tous droits réservés", "Imprimé en France"],
        ),
        (
            "de",
            "Kapitel 1: Grundlagen der Psychologie.\n"
            "Alle Rechte vorbehalten. Kein Teil dieses Werkes darf ohne Genehmigung vervielfältigt werden.\n"
            "Gedruckt in Deutschland.\n"
            "Die kognitiven Prozesse steuern das menschliche Verhalten im Alltag.",
            ["Kapitel 1", "Die kognitiven Prozesse"],
            ["Alle Rechte vorbehalten", "Gedruckt in Deutschland"],
        ),
        (
            "es",
            "Capítulo 1: Introducción a la Filosofía.\n"
            "Todos los derechos reservados. Queda prohibida la reproducción total o parcial de esta obra.\n"
            "Impreso en España.\n"
            "El pensamiento reflexivo constituye la base del conocimiento humano.",
            ["Capítulo 1", "El pensamiento reflexivo"],
            ["Todos los derechos reservados", "Impreso en España"],
        ),
        (
            "ja",
            "第一章 認知心理学の歩み\n"
            "本書の無断転載・複製を禁じます。\n"
            "人間の知覚と記憶のメカニズムについて考察する。",
            ["第一章 認知心理学の歩み", "人間の知覚と記憶のメカニズム"],
            ["無断転載"],
        ),
        (
            "zh",
            "第一章 认知心理学绪论\n"
            "版权所有，侵权必究。未经出版者预先书面许可，不得以任何形式复制。\n"
            "内部交流资料，严禁外传。\n"
            "认知心理学主要研究人类如何获取、储存和加工信息。",
            ["第一章 认知心理学绪论", "认知心理学主要研究人类"],
            ["版权所有，侵权必究", "内部交流资料"],
        ),
        (
            "ru",
            "Глава 1. Введение в когнитивную психологию.\n"
            "Все права защищены. Никакая часть данной книги не может быть воспроизведена в любой форме.\n"
            "Отпечатано в России.\n"
            "Когнитивные процессы формируют наше восприятие окружающего мира.",
            ["Глава 1", "Когнитивные процессы"],
            ["Все права защищены", "Отпечатано в России"],
        ),
        (
            "ko",
            "제1장 인지심리학의 기초\n"
            "이 책의 일부 또는 전부를 무단으로 복제할 수 없습니다.\n"
            "파본은 구입처에서 교환해 드립니다.\n"
            "인지 과정은 인간의 학습과 기억을 지배한다.",
            ["제1장 인지심리학의 기초", "인지 과정은 인간의 학습과 기억을 지배한다"],
            ["무단으로", "교환해 드립니다"],
        ),
    ],
    ids=["fr", "de", "es", "ja", "zh", "ru", "ko"],
)
def test_boilerplate_catalog_prunes_colophon_per_language(
    source_lang: str, text: str, keep: list[str], drop: list[str]
) -> None:
    cleaned = strip_textbook_ocr_artifacts(text, source_lang=source_lang)
    for phrase in keep:
        assert phrase in cleaned, f"{source_lang}: pruner removed real prose {phrase!r}"
    for phrase in drop:
        assert phrase not in cleaned, f"{source_lang}: boilerplate {phrase!r} survived"


def test_boilerplate_catalog_strips_gutenberg_banners_without_eating_prose() -> None:
    """Project Gutenberg injects banners + license headings; real prose survives.

    The banners are matched only with their triple-asterisk delimiters, so a
    sentence that merely mentions Project Gutenberg (or the transcription-credit
    shape "Produced by ...") must not be stripped — that looser rule was the
    tempting wrong fix and is asserted against here.
    """
    text = (
        "*** START OF THIS PROJECT GUTENBERG EBOOK FRANKENSTEIN ***\n"
        "Produced by the Online Distributed Proofreading Team.\n"
        "It was on a dreary night in November that I beheld the creature.\n"
        "THE FULL PROJECT GUTENBERG LICENSE\n"
        "Project Gutenberg License information: this ebook is free.\n"
        "*** END OF THIS PROJECT GUTENBERG EBOOK FRANKENSTEIN ***\n"
    )
    cleaned = strip_textbook_ocr_artifacts(text, source_lang="en")
    assert "PROJECT GUTENBERG EBOOK" not in cleaned
    assert "FULL PROJECT GUTENBERG LICENSE" not in cleaned
    assert "Project Gutenberg License information" not in cleaned
    # The real sentence, and the transcription-credit line, must both survive.
    assert "dreary night in November" in cleaned
    assert "Produced by the Online Distributed Proofreading Team" in cleaned

    # A mid-prose mention of the project is untouched by the banner rule.
    plain = "We checked it against the Project Gutenberg archive."
    assert "Project Gutenberg archive" in strip_textbook_ocr_artifacts(plain, source_lang="en")


def test_lnds_cleaner_dispatches_configured_source_lang() -> None:
    cleaner_de = LNDSPageCleaner(source_lang="de")
    block = IRBlock(
        id="de_01",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="Alle Rechte vorbehalten.\nEinführung in die Informatik.",
    )
    cleaned_blocks = cleaner_de.clean_chapter_blocks([block])
    assert len(cleaned_blocks) == 1
    assert "Alle Rechte vorbehalten" not in cleaned_blocks[0].source_text
    assert "Einführung in die Informatik" in cleaned_blocks[0].source_text


# ---------------------------------------------------------------------------
# 2. LanguagePairPolicy Matrix & Resolution Tests
# ---------------------------------------------------------------------------


def test_language_pair_policy_forward_pairs() -> None:
    # en -> zh (compression)
    policy_en_zh = get_pair_policy("en", "zh")
    assert policy_en_zh.source_code == "en"
    assert policy_en_zh.target_code == "zh"
    assert policy_en_zh.source_name == "English"
    assert policy_en_zh.target_name == "Chinese"
    assert policy_en_zh.min_length_ratio == 0.2
    assert policy_en_zh.max_length_ratio == 1.5
    assert policy_en_zh.min_target_ratio == 0.25
    assert policy_en_zh.code == "zh"

    # en -> de (Latin same-script: target gate disabled to prevent false positives)
    policy_en_de = get_pair_policy("en", "de")
    assert policy_en_de.min_target_ratio == 0.0
    assert policy_en_de.min_length_ratio == 0.7
    assert policy_en_de.max_length_ratio == 2.0

    # en -> ru (Cyrillic target)
    policy_en_ru = get_pair_policy("en", "ru")
    assert policy_en_ru.min_target_ratio == 0.25
    assert policy_en_ru.min_length_ratio == 0.6
    assert policy_en_ru.max_length_ratio == 2.0


def test_language_pair_policy_reverse_pairs() -> None:
    # zh -> en (significant expansion!)
    policy_zh_en = get_pair_policy("zh", "en")
    assert policy_zh_en.source_code == "zh"
    assert policy_zh_en.target_code == "en"
    assert policy_zh_en.source_name == "Chinese"
    assert policy_zh_en.target_name == "English"
    assert policy_zh_en.min_length_ratio == 1.0
    assert policy_zh_en.max_length_ratio == 5.0
    assert policy_zh_en.min_target_ratio == 0.0  # English target

    # ja -> en
    policy_ja_en = get_pair_policy("ja", "en")
    assert policy_ja_en.min_length_ratio == 0.8
    assert policy_ja_en.max_length_ratio == 4.5

    # de -> en
    policy_de_en = get_pair_policy("de", "en")
    assert policy_de_en.min_length_ratio == 0.6
    assert policy_de_en.max_length_ratio == 1.6


def test_language_pair_policy_fallback_for_unknown() -> None:
    # Unknown source language falls back to target profile ratio bounds
    policy_custom = get_pair_policy("xx", "zh")
    assert policy_custom.source_code == "xx"
    assert policy_custom.target_code == "zh"
    assert policy_custom.target_name == "Chinese"
    assert policy_custom.min_length_ratio == 0.2


# ---------------------------------------------------------------------------
# 3. FastPassFilter Dynamic Pair Validation Tests
# ---------------------------------------------------------------------------


def test_fast_pass_en_to_zh_evaluations() -> None:
    fp = FastPassFilter(source_lang="en", target_lang="zh")

    # Valid translation
    src = "Cognitive psychology is the scientific study of mind and mental function."
    good_tgt = "认知心理学是对心灵和心理功能的科学研究。"
    dec = fp.evaluate(src, good_tgt)
    assert dec.passed, dec.reason

    # Untranslated English residue. A full verbatim echo is classified by the
    # dedicated gate (precise reason, same for every script pair); partial
    # residue still has to fall to the density measurement.
    bad_echo = "Cognitive psychology is the scientific study of mind and mental function."
    dec_echo = fp.evaluate(src, bad_echo)
    assert not dec_echo.passed
    assert "identical to source" in dec_echo.reason
    dec_residue = fp.evaluate(src, "认知心理学 is the scientific study of mind.")
    assert not dec_residue.passed
    assert "script density" in dec_residue.reason


def test_fast_pass_zh_to_en_expansion_evaluation() -> None:
    fp = FastPassFilter(source_lang="zh", target_lang="en")

    src = "认知心理学是研究心理过程的科学。"
    # English target expands significantly
    good_tgt = "Cognitive psychology is the scientific study of mental processes."
    dec = fp.evaluate(src, good_tgt)
    assert dec.passed, dec.reason

    # Target too short (violates min_length_ratio=1.0 for zh->en)
    too_short = "Cognition."
    dec_short = fp.evaluate(src, too_short)
    assert not dec_short.passed
    assert "truncated" in dec_short.reason


def test_fast_pass_en_to_ru_evaluation() -> None:
    fp = FastPassFilter(source_lang="en", target_lang="ru")

    src = "Attention and consciousness are central topics in cognitive research."
    good_ru = "Внимание и сознание являются центральными темами когнитивных исследований."
    dec = fp.evaluate(src, good_ru)
    assert dec.passed, dec.reason

    # Untranslated English in Russian pipeline fails Cyrillic script ratio
    bad_ru = "Attention and consciousness are central topics in research."
    dec_bad = fp.evaluate(src, bad_ru)
    assert not dec_bad.passed
    assert "script density" in dec_bad.reason


def test_fast_pass_korean_evaluations() -> None:
    fp_en_ko = FastPassFilter(source_lang="en", target_lang="ko")

    # Valid English -> Korean translation
    src = "Memory consolidation is crucial for long-term knowledge retention."
    good_ko = "기억 공고화는 장기적인 지식 보존에 매우 중요합니다."
    dec = fp_en_ko.evaluate(src, good_ko)
    assert dec.passed, dec.reason

    # Untranslated English residue fails Hangul script density
    bad_ko = "Memory consolidation is crucial for knowledge retention."
    dec_bad = fp_en_ko.evaluate(src, bad_ko)
    assert not dec_bad.passed
    assert "script density" in dec_bad.reason

    # Korean -> English expansion
    fp_ko_en = FastPassFilter(source_lang="ko", target_lang="en")
    src_ko = "기억 공고화는 장기 지식 보존에 필수적이다."
    good_en = "Memory consolidation is essential for long-term knowledge retention."
    dec_en = fp_ko_en.evaluate(src_ko, good_en)
    assert dec_en.passed, dec_en.reason


def test_fast_pass_spanish_evaluations() -> None:
    fp_en_es = FastPassFilter(source_lang="en", target_lang="es")

    # English -> Spanish
    src = "Artificial intelligence is transforming scientific discoveries across disciplines."
    good_es = "La inteligencia artificial está transformando los descubrimientos científicos en diversas disciplinas."
    dec = fp_en_es.evaluate(src, good_es)
    assert dec.passed, dec.reason

    # Spanish -> English
    fp_es_en = FastPassFilter(source_lang="es", target_lang="en")
    src_es = (
        "El aprendizaje profundo permite modelar patrones complejos en grandes volúmenes de datos."
    )
    good_en = "Deep learning makes it possible to model complex patterns in large volumes of data."
    dec_es_en = fp_es_en.evaluate(src_es, good_en)
    assert dec_es_en.passed, dec_es_en.reason

    # Romance same-script pair (es -> fr): min_target_ratio is 0.0, valid length ratio passes
    fp_es_fr = FastPassFilter(source_lang="es", target_lang="fr")
    assert fp_es_fr.policy.min_target_ratio == 0.0
    good_fr = "L'apprentissage profond permet de modéliser des modèles complexes dans de grands volumes de données."
    dec_es_fr = fp_es_fr.evaluate(src_es, good_fr)
    assert dec_es_fr.passed, dec_es_fr.reason


# ---------------------------------------------------------------------------
# 4. ModelRouter Bidirectional Prompt Assembly Tests
# ---------------------------------------------------------------------------


def test_model_router_korean_and_spanish_bidirectional() -> None:
    provider = MockModelProvider(default_response="Output")
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")

    # English -> Korean
    sys_ko, _ = router.build_draft_prompt("Hello", target_lang="ko", source_lang="en")
    assert "translating from English into Korean" in sys_ko
    assert "fluent Korean" in sys_ko

    # Korean -> Chinese
    sys_ko_zh, _ = router.build_draft_prompt("안녕하세요", target_lang="zh", source_lang="ko")
    assert "translating from Korean into Chinese" in sys_ko_zh
    assert "fluent Chinese" in sys_ko_zh

    # Spanish -> English
    sys_es_en, _ = router.build_draft_prompt("Hola", target_lang="en", source_lang="es")
    assert "translating from Spanish into English" in sys_es_en

    # Chinese -> Spanish
    sys_zh_es, _ = router.build_draft_prompt("你好", target_lang="es", source_lang="zh")
    assert "translating from Chinese into Spanish" in sys_zh_es


def test_model_router_build_draft_prompt_bidirectional() -> None:
    provider = MockModelProvider(default_response="Traducción")
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")

    # 1. French -> German
    sys_prompt, user_prompt = router.build_draft_prompt(
        source_text="Les systèmes distribués sont complexes.",
        target_lang="de",
        source_lang="fr",
        genre_profile="academic",
    )
    assert "translating from French into German" in sys_prompt
    assert "fluent German" in sys_prompt
    assert "Les systèmes distribués sont complexes." in user_prompt

    # 2. Chinese -> English
    sys_prompt_zh_en, _ = router.build_draft_prompt(
        source_text="深度学习模型在翻译任务中表现优异。",
        target_lang="en",
        source_lang="zh",
        genre_profile="textbook",
    )
    assert "translating from Chinese into English" in sys_prompt_zh_en
    assert "fluent English" in sys_prompt_zh_en


def test_model_router_build_repair_prompt_bidirectional() -> None:
    provider = MockModelProvider(default_response="Repaired")
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")

    sys_prompt, user_prompt = router.build_repair_prompt(
        source_text="原著テキスト",
        draft_text="Rough draft",
        error_flags=["length_anomaly"],
        target_lang="en",
        source_lang="ja",
    )
    assert "refining a translation from Japanese into English" in sys_prompt
    assert "<final_translation>" in sys_prompt
    assert "原著テキスト" in user_prompt


@pytest.mark.asyncio
async def test_model_router_draft_and_repair_pass_bidirectional_languages() -> None:
    provider = MockModelProvider(
        default_response="<final_translation>Ceci est un test.</final_translation>"
    )
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")

    block = IRBlock(
        id="multi_01",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="This is a test sentence.",
    )

    # Draft en -> fr
    draft_res = await router.draft(
        block=block,
        source_lang="en",
        target_lang="fr",
    )
    assert draft_res
    assert len(provider.call_history) == 1
    assert "English into French" in provider.call_history[0]["system_prompt"]

    # Repair en -> fr
    repair_res = await router.repair(
        block=block,
        draft_text="Mauvais texte",
        error_flags=["unnatural_flow"],
        source_lang="en",
        target_lang="fr",
    )
    assert repair_res
    assert len(provider.call_history) == 2
    assert "English into French" in provider.call_history[1]["system_prompt"]
