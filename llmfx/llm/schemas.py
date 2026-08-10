"""LLM の構造化出力スキーマ.

構造化出力(output_config.format)は数値の範囲制約などをサポートしないため、
JSON Schema には型と enum だけを載せ、値域の検証は Python 側で行う。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    """全フィールド必須・追加プロパティ禁止(構造化出力の要件)。"""

    model_config = ConfigDict(extra="forbid")


class GateDecision(_Strict):
    """ルールが出したエントリーを通すか見送るかの判断."""

    approve: bool
    confidence: float
    """0.0〜1.0。0.5 未満は「消極的な承認」として扱う。"""
    primary_reason: str
    risk_flags: list[str]
    """見送りを検討させた要因(例: 重要指標発表直前、値幅が異常、流動性が薄い時間帯)。"""


class EntryNote(_Strict):
    """エントリー時の所感。何を根拠に入ったのかを後から検証できる形で残す."""

    thesis: str
    """このトレードで賭けている仮説を 1〜2 文で。"""
    structure_read: str
    """ダウ構造の読み(どの高値/安値をどう解釈したか)。"""
    invalidation: str
    """この仮説が間違いだったと判断できる条件。"""
    watch_points: list[str]
    conviction: int
    """1〜5。5 が最も確信が高い。"""


class ExitNote(_Strict):
    """決済後の振り返り。勝ち負けの原因を構造・執行・運の 3 つに切り分ける."""

    outcome: Literal["win", "loss", "breakeven"]
    what_happened: str
    thesis_was_correct: bool
    """仮説自体は正しかったか(正しくても負けることはある)。"""
    primary_cause: Literal[
        "thesis_wrong",
        "structure_misread",
        "stop_too_tight",
        "target_too_far",
        "entry_too_late",
        "execution_cost",
        "market_regime",
        "variance",
        "thesis_correct",
    ]
    cause_detail: str
    execution_errors: list[str]
    lessons: list[str]
    suggested_adjustments: list[str]
    """次に試す具体的な変更案(パラメータ名と方向を含める)。"""


class ParameterSuggestion(_Strict):
    parameter: str
    current_value: str
    suggested_value: str
    rationale: str
    expected_effect: str
    risk_of_change: str


class ReviewReport(_Strict):
    """蓄積したトレードと所感から生成する改善レポート."""

    summary: str
    biggest_problem: str
    strengths: list[str]
    weaknesses: list[str]
    root_causes: list[str]
    parameter_suggestions: list[ParameterSuggestion]
    risk_warnings: list[str]
    next_experiments: list[str]
    monthly_target_assessment: str
    """目標月利に対する率直な評価。届かないなら届かないと書かせる。"""


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic モデルから構造化出力用の JSON Schema を作る。

    すべてのオブジェクトに `additionalProperties: false` を強制し、
    サポート外のキーワード(数値範囲など)を落とす。
    """
    schema = model.model_json_schema()
    _sanitize(schema)
    return schema


_UNSUPPORTED_KEYWORDS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "default",
    "examples",
}


def _sanitize(node: Any) -> None:
    if isinstance(node, dict):
        for keyword in list(node):
            if keyword in _UNSUPPORTED_KEYWORDS:
                node.pop(keyword, None)
        if node.get("type") == "object":
            node["additionalProperties"] = False
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
        for value in node.values():
            _sanitize(value)
    elif isinstance(node, list):
        for item in node:
            _sanitize(item)
