"""Claude を使った補助レイヤ.

役割は 3 つ。売買判断そのものはルール側が持ち、LLM は判断を「増やさない」:

  1. 拒否権   : ルールが出したエントリーを見送るかどうかだけを決める
  2. 所感     : なぜ入ったか / なぜ負けたか を言語化して残す
  3. 改善提案 : 蓄積した所感と成績から、次に試すべき変更を提案する

API キーが無い場合はすべて自動的に無効化され、ルールベース部分だけで動く。
"""

from .client import LLMClient, LLMUnavailable
from .gate import EntryGate
from .journalist import Journalist

__all__ = ["LLMClient", "LLMUnavailable", "EntryGate", "Journalist"]
