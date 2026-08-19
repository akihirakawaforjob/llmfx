"""時間帯とイベントによるエントリー制限.

狙い:
    ダウ転換はブレイクを捉える手法なので、ブレイクが機能する時間帯と、
    そうでない時間帯があるはず。また、経済指標や為替介入のように
    「価格構造の外側」で動く場面では、構造から出したシグナルに意味が無い。

先読みを持ち込まないこと:
    「2024 年の介入で 164 円が壁だと分かった」を条件に埋め込むのは反則。
    その水準を知り得たのは介入が起きた後であって、2021 年時点では誰も
    知らない。バックテストの成績だけが良くなり、実運用では再現しない。

    ここで扱うのは、**その時点で手に入る情報だけ**で判定できるものに限る:

      - 時刻・曜日          カレンダーから事前に分かる
      - 主要指標の予定      雇用統計は毎月第 1 金曜、FOMC は年 8 回。事前に分かる
      - 直近のボラティリティ 過去のバーだけから計算できる

    「過去 N 本の高値圏にいるか」のような相対位置も、N 本が過去のバーだけで
    あれば正当に使える。絶対水準を決め打ちするのが反則。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# 米雇用統計。毎月第 1 金曜の 8:30 ET。
# 夏時間(3 月第 2 日曜〜11 月第 1 日曜)は 12:30 UTC、それ以外は 13:30 UTC。
NFP_HOUR_UTC_DST = 12
NFP_HOUR_UTC_STD = 13
NFP_MINUTE = 30


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
    """その月の第 n 週の指定曜日(weekday: 月=0 … 日=6)。"""
    first = datetime(year, month, 1, tzinfo=timezone.utc)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _us_dst(moment: datetime) -> bool:
    """米国の夏時間中か(3 月第 2 日曜 〜 11 月第 1 日曜)。"""
    start = _nth_weekday(moment.year, 3, 6, 2)   # 3 月第 2 日曜
    end = _nth_weekday(moment.year, 11, 6, 1)    # 11 月第 1 日曜
    return start <= moment < end


def nfp_time(year: int, month: int) -> datetime:
    """その月の米雇用統計の発表時刻(UTC)。"""
    day = _nth_weekday(year, month, 4, 1)  # 第 1 金曜
    hour = NFP_HOUR_UTC_DST if _us_dst(day) else NFP_HOUR_UTC_STD
    return day.replace(hour=hour, minute=NFP_MINUTE)


@dataclass(frozen=True)
class SessionFilter:
    """時刻・曜日・指標予定でエントリーを絞る。

    どれも None / 空なら素通しする。既定は素通し。
    """

    allowed_hours_utc: tuple[tuple[int, int], ...] = ()
    """許可する時間帯(UTC、開始時 <= h < 終了時)。空なら全時間帯を許可。

    例: ((7, 16),) でロンドン序盤〜NY 前場だけ。
    終了 < 開始 なら日をまたぐ帯として扱う(例: (22, 6))。
    """
    blocked_weekdays: tuple[int, ...] = ()
    """見送る曜日(月=0 … 日=6)。FX の月曜早朝や金曜終盤を外す用。"""
    nfp_blackout_minutes: int = 0
    """米雇用統計の前後この分数を遮断する。0 で無効。

    毎月第 1 金曜と決まっているので、事前に分かる情報だけで判定できる。
    """

    def allows(self, moment: datetime) -> tuple[bool, str | None]:
        """(通すか, 落とした理由) を返す。"""
        if moment.tzinfo is None:
            raise ValueError("時刻はタイムゾーン付きで渡してください")
        moment = moment.astimezone(timezone.utc)

        if self.blocked_weekdays and moment.weekday() in self.blocked_weekdays:
            return False, "weekday_blocked"

        if self.allowed_hours_utc and not any(
            _in_hour_range(moment.hour, lo, hi) for lo, hi in self.allowed_hours_utc
        ):
            return False, "outside_session"

        if self.nfp_blackout_minutes > 0 and self._near_nfp(moment):
            return False, "nfp_blackout"

        return True, None

    def _near_nfp(self, moment: datetime) -> bool:
        window = timedelta(minutes=self.nfp_blackout_minutes)
        # 月初・月末の跨ぎを拾うため、前月・当月・翌月を見る。
        for year, month in _neighbour_months(moment.year, moment.month):
            if abs(moment - nfp_time(year, month)) <= window:
                return True
        return False


def _in_hour_range(hour: int, lo: int, hi: int) -> bool:
    if lo == hi:
        return False
    if lo < hi:
        return lo <= hour < hi
    # 日をまたぐ帯(例: 22 時 〜 翌 6 時)
    return hour >= lo or hour < hi


def _neighbour_months(year: int, month: int) -> list[tuple[int, int]]:
    out = [(year, month)]
    out.append((year - 1, 12) if month == 1 else (year, month - 1))
    out.append((year + 1, 1) if month == 12 else (year, month + 1))
    return out
