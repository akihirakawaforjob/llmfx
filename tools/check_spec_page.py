"""仕様ページの図が、全部ちゃんと描かれるかを見る.

節を差し替えたときに **描画の呼び出しだけ残る** と、そこで
`getElementById` が null を返して例外になり、**以降の図が全部止まる。**
実際に止めた(損切りの節の図が丸ごと出なくなった)。

    python tools/check_spec_page.py docs/spec-v4.html
"""

from __future__ import annotations

import json
import re
import sys


def check(path: str) -> list[str]:
    s = open(path, encoding="utf-8").read()
    bad: list[str] = []

    ids = set(re.findall(r'id="(c-[a-z0-9]+)"', s))
    calls = set(re.findall(r'getElementById\("(c-[a-z0-9]+)"\)', s))
    # 名前を組み立てて回している枠(`"c-" + key`)は、ここでは名前が
    # 分からない。組み立てが 1 つでもあれば、拾えなかった枠は見逃す。
    built = bool(re.search(r'getElementById\("c-" \+', s))

    if (ids - calls) and not built:
        bad.append(f"枠はあるが描いていない: {sorted(ids - calls)}")
    # **こちらは組み立てがあっても必ず見る。**残った呼び出しが例外を出し、
    # 以降の図を全部止める。
    if calls - ids:
        bad.append(f"描こうとしているが枠が無い: {sorted(calls - ids)}")

    m = re.search(r"const F = (\{.*?\});\n", s, re.S)
    if m is None:
        pass                                   # 図のデータを別名で持つ頁もある
    else:
        try:
            fig = json.loads(m.group(1))
        except Exception as exc:                      # noqa: BLE001
            bad.append(f"図のデータが JSON として読めない: {exc}")
        else:
            # `_` を入れないと `F.v2_bounce` を `F.v2` として拾い、
            # 存在しない名前を報告してしまう。
            used = set(re.findall(r"F\.([a-z0-9_]+)", s))
            missing = used - set(fig)
            if missing:
                bad.append(f"参照しているのに図のデータに無い: {sorted(missing)}")

    for tag in (r"<!doctype", r"<html\b", r"<head\b", r"<body\b"):
        if re.search(tag, s, re.I):
            bad.append(f"Artifact に置けないタグがある: {tag}")
    return bad


if __name__ == "__main__":
    fails = []
    for path in sys.argv[1:] or ["docs/spec-v4.html"]:
        problems = check(path)
        print(f"{path}: " + ("ok" if not problems else "NG"))
        for line in problems:
            print("  " + line)
        fails += problems
    sys.exit(1 if fails else 0)
