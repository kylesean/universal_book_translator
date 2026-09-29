"""Tests for 0-token character mining and person-name rendering lock."""

from pathlib import Path
from typing import Any

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.memory.abbreviation_backfill import build_backfill_prompt
from ubt.core.memory.character_miner import mine_characters
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


def test_mines_honorific_names() -> None:
    text = (
        "Mr. Darcy walked in. Mrs. Bennet smiled. Lady Catherine de Bourgh frowned. "
        "Mr. Darcy bowed again."
    )
    entries = mine_characters(text, min_freq=99)
    sources = {e["source"] for e in entries}
    assert "Mr. Darcy" in sources
    assert "Mrs. Bennet" in sources
    assert "Lady Catherine de Bourgh" in sources
    darcy = next(e for e in entries if e["source"] == "Mr. Darcy")
    assert darcy["kind"] == "person"
    assert darcy["aliases"] == ["Darcy"]
    assert darcy["translation"] == ""
    # name core of a multi-part name skips particles
    lady = next(e for e in entries if e["source"].startswith("Lady"))
    assert lady["aliases"] == ["Bourgh"]


def test_mines_frequent_bare_names() -> None:
    text = " ".join(
        [
            "she said, Elizabeth agreed at once, and Elizabeth smiled.",
            "told Elizabeth the truth, because Elizabeth listened.",
            "asked Elizabeth twice more.",
            "Gutenberg appears often, Gutenberg again, Gutenberg thrice, Gutenberg four, Gutenberg five.",
            "Chapter marks, Chapter again, Chapter more.",
        ]
    )
    entries = mine_characters(text, min_freq=4)
    names = [e["source"] for e in entries]
    assert "Elizabeth" in names
    assert "Gutenberg" not in names
    assert "Chapter" not in names


def test_no_overlap_between_honorific_and_bare_mining() -> None:
    text = (
        "Mr. Bingley arrived. "
        "said Bingley, and Bingley laughed, and Bingley sang, and Bingley left, "
        "and Bingley returned, and Bingley stayed, and Bingley danced, and Bingley bowed, "
        "and Bingley smiled, and Bingley waved, and Bingley nodded, and Bingley wrote, "
        "and Bingley read, and Bingley ran, and Bingley spoke, and Bingley waited."
    )
    entries = mine_characters(text, min_freq=5)
    bingley_entries = [e for e in entries if "ingley" in e["source"].lower()]
    assert len(bingley_entries) == 1
    assert bingley_entries[0]["source"] == "Mr. Bingley"


def test_sentence_start_only_words_rejected() -> None:
    # "Then" appears 20 times but always sentence-initial
    text = "Then he left. " * 20
    assert mine_characters(text, min_freq=5) == []


def test_backfill_prompt_marks_person_entries() -> None:
    entries = [
        {"source": "Mr. Darcy", "translation": "", "aliases": ["Darcy"], "kind": "person"},
        {"source": "Working Memory", "translation": "", "aliases": ["WM"], "kind": "term"},
    ]
    system, user = build_backfill_prompt(entries, "zh")
    assert "1. Darcy — Mr. Darcy (person name)" in user
    assert "2. WM — Working Memory" in user
    assert "WITHOUT the courtesy title" in user
    assert "KEY = <rendering>" in user


@pytest.mark.asyncio
async def test_pipeline_locks_character_rendering_across_chapters(tmp_path: Path) -> None:
    """Mined person names get ONE decided rendering, injected into every
    draft prompt that mentions the name — drift becomes impossible."""
    md = tmp_path / "characters.md"
    md.write_text(
        "# CHAPTER I.\n\n"
        "Mr. Bingley arrived at Netherfield and called on the family.\n"
        "Everyone talked about Mr. Bingley for the rest of the day.\n\n"
        "# CHAPTER II.\n\n"
        "Mr. Bingley returned in the morning with his friend.\n"
        "The whole neighbourhood discussed Mr. Bingley again.\n",
        encoding="utf-8",
    )
    provider = MockModelProvider(
        default_response="这是默认翻译。",
        # key must match ONLY the backfill prompt ("### Items" never appears in
        # draft prompts), simulating the LLM deciding the canonical rendering
        custom_responses={"### Items": "Bingley = 宾利"},
    )
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(db_dir=tmp_path / "db", rate_limit_rpm=600),
        router=router,
        qe_runner=MockQERunner(default_score=0.9),
    )

    async for _ in orchestrator.run(
        input_path=md, output_path=tmp_path / "out.md", target_lang="zh", job_id="job_char_lock"
    ):
        pass

    backfill_calls = [c for c in provider.call_history if "### Items" in c["prompt"]]
    assert len(backfill_calls) == 1
    assert "Bingley — Mr. Bingley (person name)" in backfill_calls[0]["prompt"]

    # every draft prompt mentioning Bingley carries the locked rendering
    bingley_drafts = [
        c
        for c in provider.call_history
        if "Source Paragraph to Translate" in c["prompt"] and "Bingley" in c["prompt"]
    ]
    assert bingley_drafts, "expected draft prompts mentioning Bingley"
    assert all("宾利" in c["prompt"] for c in bingley_drafts)
    # and no draft prompt carries any competing rendering
    assert not any("彬格莱" in c["prompt"] for c in provider.call_history)


def test_mines_all_surface_variants_of_same_core() -> None:
    """Mr. Bingley and Miss Bingley must both be mined."""
    text = "Mr. Bingley arrived. Miss Bingley followed. Mr. Bingley smiled."
    entries = mine_characters(text, min_freq=99)
    sources = {e["source"] for e in entries}
    assert "Mr. Bingley" in sources
    assert "Miss Bingley" in sources
    # both share the name-core alias so the backfill key dedupe unifies them
    assert all(e["aliases"] == ["Bingley"] for e in entries)


@pytest.mark.asyncio
async def test_backfill_shares_rendering_across_surface_variants() -> None:
    from ubt.core.memory.abbreviation_backfill import backfill_abbreviation_translations

    entries = [
        {"source": "Mr. Bingley", "translation": "", "aliases": ["Bingley"], "kind": "person"},
        {"source": "Miss Bingley", "translation": "", "aliases": ["Bingley"], "kind": "person"},
    ]
    seen_items: list[str] = []

    async def fake_complete(system: str, user: str) -> str:
        seen_items.append(user)
        return "Bingley = 宾利"

    result, filled = await backfill_abbreviation_translations(entries, fake_complete)
    assert filled == 2
    assert all(e["translation"] == "宾利" for e in result)
    # dedupe: only ONE prompt item for the shared key
    assert "1. Bingley — Mr. Bingley (person name)" in seen_items[0]
    assert "Miss Bingley" not in seen_items[0].split("### Items")[1]


def test_glossary_validator_flags_alias_surface_drift() -> None:
    from ubt.core.validators.consistency import GlossaryConsistencyValidator

    glossary = [
        {"source": "Mr. Bingley", "translation": "宾利", "aliases": ["Bingley"], "kind": "person"}
    ]
    v = GlossaryConsistencyValidator(glossary)

    ok = v.validate("Miss Bingley smiled at the party.", "宾利小姐在聚会上微笑。")
    assert ok.is_valid  # alias surface carrying the decided rendering

    drift = v.validate("Miss Bingley smiled at the party.", "彬格莱小姐在聚会上微笑。")
    assert not drift.is_valid  # classic-translation drift must be flagged


def test_mines_german_honorifics_and_suppresses_bare_nouns() -> None:
    """German honorifics are mined, but bare capitalized nouns are suppressed."""
    text = (
        "Herr Müller kam an. Frau Schmidt lachte. Dr. Weber sprach gestern. "
        "Das Buch ist gut, und das Buch liegt hier, und das Buch ist neu. "
        "Die Stadt ist groß, die Stadt ist alt, die Stadt gefällt mir."
    )
    entries = mine_characters(text, source_lang="de", min_freq=2)
    sources = {e["source"] for e in entries}
    assert "Herr Müller" in sources
    assert "Frau Schmidt" in sources
    assert "Dr. Weber" in sources
    # Common German nouns must NEVER be mined as people
    assert "Buch" not in sources
    assert "Stadt" not in sources


def test_mines_french_honorifics() -> None:
    """French honorifics (M., Mme, Docteur) are properly mined."""
    text = "M. Dupont est venu. Mme Bovary regardait. Docteur Rieux est intervenu."
    entries = mine_characters(text, source_lang="fr", min_freq=99)
    sources = {e["source"] for e in entries}
    assert "M. Dupont" in sources
    assert "Mme Bovary" in sources
    assert "Docteur Rieux" in sources


def test_mines_spanish_honorifics() -> None:
    """Spanish honorifics (Don, Señor, Dr.) are properly mined."""
    text = "Don Quijote cabalgaba. Señor Pérez habló. Dr. García sonrió."
    entries = mine_characters(text, source_lang="es", min_freq=99)
    sources = {e["source"] for e in entries}
    assert "Don Quijote" in sources
    assert "Señor Pérez" in sources
    assert "Dr. García" in sources


def test_cjk_source_safely_degrades() -> None:
    """CJK text without any honorific still mines nothing (no Latin-regex spillover)."""
    text = "诸葛亮对刘备说：天下大势，分久必合，合久必分。周瑜和鲁肃在江东。"
    entries = mine_characters(text, source_lang="zh", min_freq=1)
    assert entries == []


def test_mines_chinese_honorific_names() -> None:
    """Chinese postposed honorifics (先生/女士/老师/教授) anchor name cores."""
    text = "王小明先生走进会议室。张老师说今天开工。李教授负责评审。欧阳文女士负责财务。"
    entries = mine_characters(text, source_lang="zh", min_freq=99)
    by_source = {e["source"]: e for e in entries}
    assert "王小明" in by_source
    assert "张" in by_source
    assert "李" in by_source
    assert "欧阳文" in by_source
    assert all(e["kind"] == "person" for e in entries)
    assert all(e["translation"] == "" for e in entries)
    # honorific is NOT part of the mined source (glossary protects the bare name)
    assert all(not e["source"].endswith(("先生", "女士", "老师", "教授")) for e in entries)


def test_chinese_strips_glued_function_words() -> None:
    """Conjunctions/particles gluing onto names are shaved: 和林先生 -> 林."""
    text = "王先生和林先生说好了。他告诉赵老师下周开会。我见到钱博士很高兴。"
    sources = {e["source"] for e in mine_characters(text, source_lang="zh", min_freq=99)}
    assert {"王", "林", "赵", "钱"} <= sources
    assert "和林" not in sources
    assert "告诉赵" not in sources


def test_chinese_rejects_generic_honorific_matches() -> None:
    """这位/一位/旁边的/N的先生 style matches must never become bible entries."""
    text = (
        "这位先生没有说话。我的一位先生提到过。旁边的先生在看。"
        "他的先生说走了。同先生们讨论。老先生也来了。"
    )
    assert mine_characters(text, source_lang="zh", min_freq=99) == []


def test_mines_japanese_honorific_names() -> None:
    """Japanese postposed honorifics (さん/様/ちゃん/くん) anchor name cores."""
    text = "田中さんが来た。佐藤様から电话があった。さくらちゃんは笑った。健太くんも来た。"
    entries = mine_characters(text, source_lang="ja", min_freq=99)
    sources = {e["source"] for e in entries}
    assert {"田中", "佐藤", "さくら", "健太"} <= sources
    assert all(e["kind"] == "person" for e in entries)
    # kana-only given names survive intact (no stripping on kana-only matches)
    sakura = next(e for e in entries if e["source"] == "さくら")
    assert sakura["aliases"] == ["さくら"]


def test_japanese_strips_particles_and_generic_compounds() -> None:
    """は田中さん -> 田中; 皆様/お疲れ様/お客様/同様 never become people."""
    text = "は田中さんが言った。すると田中さんも颔いた。"
    sources = {e["source"] for e in mine_characters(text, source_lang="ja", min_freq=99)}
    assert sources == {"田中"}  # both spellings dedupe to the bare name

    noise = "皆様のおかげです。お疲れ様でした。お客様が来ました。同様に処理する。諸君もがんばれ。"
    assert mine_characters(noise, source_lang="ja", min_freq=99) == []


def test_mines_portuguese_honorifics() -> None:
    """Portuguese reuses the Spanish family with PT honorifics."""
    text = "Dom Quixote cavalgava. Senhor Silva falou. Dr. Costa sorriu. Sr. Santos chegou."
    entries = mine_characters(text, source_lang="pt", min_freq=99)
    sources = {e["source"] for e in entries}
    assert "Dom Quixote" in sources
    assert "Senhor Silva" in sources
    assert "Dr. Costa" in sources
    assert "Sr. Santos" in sources
    silva = next(e for e in entries if e["source"] == "Senhor Silva")
    assert silva["aliases"] == ["Silva"]


def test_korean_source_still_defers() -> None:
    """ko remains on the CJK placeholder config."""
    assert mine_characters("김선생님이 말했다.", source_lang="ko") == []


def test_chinese_stream_matches_whole_text() -> None:
    """Streaming ZH mining equals whole-text mining."""
    from ubt.core.memory.character_miner import mine_characters_stream

    blocks = [
        "王小明先生走进会议室，",
        "张老师随即开始发言。李教授负责评审。",
        "欧阳文女士记录了会议纪要，",
        "王小明先生最后做了总结。",
    ]
    joined = "\n".join(blocks)
    assert mine_characters_stream(blocks, source_lang="zh") == mine_characters(
        joined, source_lang="zh"
    )


def test_japanese_stream_boundary_honorific() -> None:
    """A Japanese name split across two blocks is still mined."""
    from ubt.core.memory.character_miner import mine_characters_stream

    blocks = ["会議の最後に、", "田中さんがまとめました。佐藤様も同席した。"]
    entries = mine_characters_stream(blocks, source_lang="ja", min_freq=99)
    sources = {e["source"] for e in entries}
    assert {"田中", "佐藤"} <= sources


def test_character_stream_matches_whole_text() -> None:
    """Streaming block-wise mining equals whole-text mining."""
    from ubt.core.memory.character_miner import mine_characters_stream

    blocks = [
        "Elizabeth Bennet was reading. Mr. Darcy arrived soon after.",
        "Elizabeth said hello to the garden. Colonel Fitzwilliam laughed.",
        "Elizabeth and Mr. Darcy spoke quietly near the old wall.",
        "The Elizabeth river flowed. Elizabeth nodded at last.",
        "Elizabeth smiled. Dr. Grant observed from the window.",
    ]
    joined = "\n".join(blocks)
    assert mine_characters_stream(blocks, min_freq=3) == mine_characters(joined, min_freq=3)


def test_character_stream_boundary_honorific() -> None:
    """An honorific name split across two blocks is still mined."""
    from ubt.core.memory.character_miner import mine_characters_stream

    blocks = ["The estate was quiet when Mr.", "Darcy appeared at dawn."]
    entries = mine_characters_stream(blocks, min_freq=99)
    assert any("Darcy" in e["aliases"] for e in entries)


def test_english_honorific_names_keep_their_accents() -> None:
    """An ASCII-only name class truncated "Mr. José" to the fragment "Mr. Jos".

    The fragment passed the stop-word gate, got a backfilled transliteration, and
    persons rank first in the global sheet — so a bogus entry rode every block
    prompt of the book. DE/FR/ES already include their accented classes.
    """
    entries = mine_characters(
        "Mr. José arrived and shook hands. Mr. José left soon.", source_lang="en"
    )
    assert [e["source"] for e in entries] == ["Mr. José"]
    assert entries[0]["aliases"] == ["José"]


def test_chinese_compound_surnames_are_not_clipped_to_a_tail_fragment() -> None:
    """The 3-char window cut 诸葛孔明先生 down to 葛孔明.

    CJK term enforcement counts substrings, so that invented fragment matched
    inside every real occurrence of the name and its rendering was applied
    book-wide. Four characters covers the closed class of compound surnames and
    transliterated names.
    """
    entries = mine_characters(
        "诸葛孔明先生摇着扇子。欧阳小明先生走进房间。托尔斯泰先生说。",
        source_lang="zh",
        min_freq=99,
    )
    assert {e["source"] for e in entries} == {"诸葛孔明", "欧阳小明", "托尔斯泰"}


def test_stream_frequency_survives_a_flush_boundary() -> None:
    """A sighting inside the carried-over tail used to be counted twice.

    Regions are scanned with the previous chunk's last ``tail_chars`` glued on
    (so a name split across the boundary is still found), and production chunks
    at 262,144 chars — any book over ~256 KB crosses a boundary. The entry is
    deduped but the attestation count is not, so the inflated frequency outranks
    real terms in the capped prompt sheet. The single-flush fixtures elsewhere
    cannot see this.
    """
    from ubt.core.memory.character_miner import mine_characters, mine_characters_stream

    filler = "Quiet prose fills the page here. " * 40
    first = filler + " Mr. Darcy called early."
    second = filler

    def freq(entries: list[dict[str, Any]], name: str = "Mr. Darcy") -> list[int]:
        return [int(e["frequency"]) for e in entries if e["source"] == name]

    stream = mine_characters_stream([first, second], chunk_chars=1000, tail_chars=200, min_freq=1)
    assert freq(stream) == freq(mine_characters(first + "\n" + second, min_freq=1)) == [1]


def test_general_relative_phrase_is_not_mined_as_person() -> None:
    """ "General Relativity" produced kind=person entry "Relativity"."""
    text = (
        "General Relativity is a theory of gravitation. "
        "General Relativity predicts the precession of Mercury. "
        "General Relativity has been tested repeatedly."
    )
    entries = mine_characters(text, min_freq=99)
    assert not [e for e in entries if e["source"].startswith("General")]
    _, prompt = build_backfill_prompt(entries, "zh")
    assert "Relativity" not in prompt


def test_honorific_plus_surname_still_mined_as_person() -> None:
    """The courtesy-title → surname path keeps extracting valid individuals."""
    entries = mine_characters("Dr. Smith arrived. Dr. Smith left again.", min_freq=99)
    smith = next(e for e in entries if e["source"] == "Dr. Smith")
    assert smith["kind"] == "person"
    assert smith["aliases"] == ["Smith"]
    _, prompt = build_backfill_prompt(entries, "zh")
    assert "Smith — Dr. Smith (person name)" in prompt


def test_honorific_core_never_spans_a_following_honorific() -> None:
    """Greedy core capture manufactured '先生和' from adjacent 先生 honorifics."""
    from ubt.core.memory.character_miner import mine_characters

    mined = mine_characters("和先生和先生和先生", source_lang="zh", min_freq=1)
    assert all("先生" not in str(entry.get("source", "")) for entry in mined), mined


def test_honorific_core_still_mines_a_normal_name() -> None:
    from ubt.core.memory.character_miner import mine_characters

    mined = mine_characters("王先生来了。王先生走了。", source_lang="zh", min_freq=2)
    assert any(str(entry.get("source", "")) == "王" for entry in mined), mined


# Natural paragraph breaks over a mix of honorific and frequent bare names.
_MINER_BLOCKS = (
    "Elizabeth Bennet was reading quietly.",
    "Mr. Darcy arrived soon after and bowed.",
    "Elizabeth said hello to the garden.",
    "Colonel Fitzwilliam laughed out loud.",
    "Elizabeth and Mr. Darcy spoke near the old stone wall.",
    "The Elizabeth river flowed past the village.",
    "Elizabeth nodded at last.",
    "Elizabeth smiled warmly.",
    "Dr. Grant observed from the window.",
)


@pytest.mark.fast
def test_mining_is_deterministic_for_repeated_identical_input() -> None:
    """Mining is a pure function of (text, options): no dedupe-order nondeterminism."""
    from ubt.core.memory.character_miner import mine_characters_stream

    joined = "\n".join(_MINER_BLOCKS)
    assert mine_characters(joined, min_freq=3) == mine_characters(joined, min_freq=3)
    assert mine_characters_stream(list(_MINER_BLOCKS), min_freq=3) == mine_characters_stream(
        list(_MINER_BLOCKS), min_freq=3
    )


@pytest.mark.fast
@pytest.mark.parametrize("chunk_chars", [50, 100, 300, 1000, 262_144])
def test_stream_mining_equals_whole_text_for_every_chunk_size(chunk_chars: int) -> None:
    """The stream is a bounded-memory view of one whole-text scan.

    The chunk size — including sizes far below the default ``tail_chars`` — must
    not change the mined set or any attestation count. The reference is the same
    text joined exactly the way the stream joins its blocks (``"\\n".join``).
    """
    from ubt.core.memory.character_miner import mine_characters_stream

    blocks = list(_MINER_BLOCKS)
    joined = "\n".join(blocks)
    assert mine_characters_stream(blocks, min_freq=2, chunk_chars=chunk_chars) == mine_characters(
        joined, min_freq=2
    )


@pytest.mark.fast
@pytest.mark.parametrize("blocks", [[], [""], ["", ""], ["\n"], ["   \n  "]])
def test_empty_input_mines_no_entries(blocks: list[str]) -> None:
    from ubt.core.memory.character_miner import mine_characters_stream

    assert mine_characters_stream(blocks, min_freq=1) == []


@pytest.mark.fast
def test_empty_string_mines_no_entries() -> None:
    assert mine_characters("", min_freq=1) == []


@pytest.mark.fast
def test_overlong_block_equals_its_partitioned_equivalent() -> None:
    """A document far larger than one chunk mines the same names however it is cut."""
    from ubt.core.memory.character_miner import mine_characters_stream

    big = (" ".join(_MINER_BLOCKS) + "\n") * 400
    partitioned = [big[i : i + 4096] for i in range(0, len(big), 4096)]
    assert mine_characters_stream([big], min_freq=3) == mine_characters(big, min_freq=3)
    assert mine_characters_stream(partitioned, min_freq=3, chunk_chars=2048) == mine_characters(
        "\n".join(partitioned), min_freq=3
    )


@pytest.mark.fast
def test_cjk_stream_source_safely_degrades() -> None:
    """Honorific-free CJK mines nothing in stream mode either (no Latin-regex spill)."""
    from ubt.core.memory.character_miner import mine_characters_stream

    assert (
        mine_characters_stream(
            ["诸葛亮对刘备说：天下大势。", "周瑜和鲁肃在江东。"], source_lang="zh", min_freq=1
        )
        == []
    )
