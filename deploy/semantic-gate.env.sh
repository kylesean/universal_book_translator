# Semantic judgement: the configuration this project uses
#
# Tracked here (not under docs/, which is git-ignored) because these values are
# the measured result of the semantic-gate work, not a preference.
#
# The judge is the only gate that can see meaning. Everything here was measured —
# see docs/guides/SEMANTIC_GATE.md for the numbers and the models rejected.
#
# Put these in your shell profile, or prefix a run with them.

# --- the judge -------------------------------------------------------------
export UBT_QE_ENGINE=tiered
export UBT_QE_JUDGE_ENABLED=true
# Without a sample rate NO clean pass reaches the judge, even with it "enabled":
# a compressed translation scores 0.92, and the default sample is 0.0.
export UBT_QE_JUDGE_PASS_SAMPLE=0.15
# The local gateway's hosted model, not the local 4B. Measured on five real
# blocks: right on all five (0.72/0.78 caught, 0.92/0.97/0.87 passed), 1.6-3.5 s
# each. The 4B scored a translation that drops "publicly" at 100/100.
export UBT_QE_JUDGE_BASE_URL=http://127.0.0.1:3425/v1
export UBT_QE_JUDGE_MODEL=workbuddy/deepseek-v4.1-flash
export UBT_QE_JUDGE_API_MODE=openai-chat
# The gateway does not check keys, but it must be sent SOMETHING: an empty value
# produces the illegal header "Bearer " and the request fails before reaching it.
export UBT_QE_JUDGE_API_KEY=local-only

# Sentence-level comparison is what makes a one-word loss visible. A whole-block
# score dilutes it: the same defect measured 0.85 block-level and 0.78
# sentence-level. Leave this on (it is the default).
export UBT_QE_JUDGE_SENTENCES=true

# --- the translator --------------------------------------------------------
# The repair model must NOT be a 4B translation model: it has to notice that a
# noun is missing and write it back. The local gateway's general-purpose model
# handles repair; the translation-specialised model handles drafting.
export UBT_PROVIDER=index-translate
export UBT_DRAFT_MODEL=index-translate-9b
export UBT_REPAIR_MODEL=mirai-s-27b

# --- the vision table reader (optional) ------------------------------------
# Needs a running PaddleOCR-VL service. Off unless both lines are set.
# export UBT_TABLE_READER=1
# export UBT_OCR_ENDPOINT=http://127.0.0.1:8185/v1
# export UBT_OCR_MODEL=PaddlePaddle/PaddleOCR-VL-1.6
