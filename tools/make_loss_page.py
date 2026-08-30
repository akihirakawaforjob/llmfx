"""負けパターンの上位 3 つを、実際のローソク足で見せるページを組む.

**手で節を差し込まないこと。**以前それで容れ物が途中で閉じ、描画の
呼び出しが定義ブロックへ入って図が全部止まった。必ずここから生成する。

    python tools/make_loss_page.py
"""

from __future__ import annotations

import json
import re

SRC = "docs/spec-v4.html"
DATA = "docs/handoff/loss-patterns.json"
DST = "docs/loss-patterns.html"
LADDER = "H4-H1-M15"
TOP = 3


def build() -> str:
    src = open(SRC, encoding="utf-8").read()
    css = re.search(r"<style>\n(:root\{.*?)\n</style>", src, re.S).group(1)
    draw = re.search(
        r"(const NS=\"http://www\.w3\.org/2000/svg\";.*?\n\}\n)</script>",
        src, re.S).group(1)
    d = json.load(open(DATA, encoding="utf-8"))

    cards, figs, calls = [], [], []
    for mech in ("跳ね返り", "ブレイク"):
        sub = {k: v for k, v in d["counts"][LADDER].items()
               if k.startswith(mech)}
        st = sum(sub.values())
        top = sorted(sub.items(), key=lambda z: -z[1])[:TOP]
        rows = "".join(
            f'<tr><td>{k.split("|")[1]}</td><td>{k.split("|")[2]}</td>'
            f'<td class="n">{v:,}</td><td class="n">{v/st:.1%}</td></tr>'
            for k, v in sorted(sub.items(), key=lambda z: -z[1]))
        cards.append(
            f'<div class="panel"><div class="caphead"><span class="t">{mech}</span>'
            f'<span class="m">負け {st:,} 件</span></div>'
            f'<div class="tblbox pad"><table class="tbl"><thead><tr>'
            f'<th>順行</th><th>出口</th><th class="n">件数</th><th class="n">割合</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div></div>')

        for rank, (key, cnt) in enumerate(top, 1):
            ex = d["picks"][LADDER][key]
            cid = f"{'b' if mech == '跳ね返り' else 'k'}{rank}"
            _, adv, why = key.split("|")
            side = "売り" if not ex["long_side"] else "買い"
            figs.append(f'''<section class="stack g24">
  <div class="sechead"><span class="eyebrow">{mech} / {rank} 位 — {cnt:,} 件({cnt/st:.1%})</span>
    <h2>{adv} → {why}</h2></div>
  <div class="panel">
    <div class="caphead"><span class="t">{ex["pair"].upper()} {ex["when"]}</span>
      <span class="m">{side} / 順行 {ex["mfe"]:.2f} R / 逆行 {ex["mae"]:.2f} R /
      {ex["bars"]} 本保有 / 結果 {ex["r"]:+.2f} R</span></div>
    <div class="chartbox" id="c-{cid}" role="img"
      aria-label="{mech}で{adv}のまま{why}になった実際の建玉"></div>
    <div class="legend" id="l-{cid}"></div>
  </div>
</section>''')
            calls.append(f'''(function(){{
  const f=F["{cid}"];
  draw(document.getElementById("c-{cid}"), document.getElementById("l-{cid}"),
    f.candles,
    [{{y:f.band,c:"--zone",w:2.5,t:"帯 "+f.band.toFixed(3),k:"抵抗帯"}},
     {{y:f.stop,c:"--stop",w:2.5,dash:"4 4",t:"損切り",k:"損切り"}}],
    [{{i:f.entry_at,y:f.entry,c:f.long_side?"--buy":"--sell",kind:"entry",
       t:"{side}",down:!f.long_side}},
     {{i:f.exit_at,y:f.exit,c:"--muted",kind:"ghost",t:"決済 "+f.r.toFixed(2)+" R"}}],
    {{h:330,alt:"実際の負け建玉"}});
}})();''')
            d["picks"][LADDER][key]["_cid"] = cid

    fig = {v["_cid"]: {k: v[k] for k in
                       ("candles", "band", "entry", "stop", "exit",
                        "entry_at", "exit_at", "long_side", "r")}
           for v in d["picks"][LADDER].values() if "_cid" in v}

    head = f'''<title>負けはどこで起きているか</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Old+Mincho:wght@500;700&family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
{css}
.two{{display:grid; gap:20px; grid-template-columns:1fr 1fr}}
@media (max-width:860px){{.two{{grid-template-columns:1fr}}}}
</style>
'''
    body = f'''<div class="wrap">
<header class="top stack g16">
  <span class="eyebrow">llmfx / 抵抗帯トレード / 負けの内訳</span>
  <h1>負けはどこで起きているか</h1>
  <p class="lede prose">負けを R の大小で割っても何も見えません。損切りが効くので
  ほぼ全部 -1.0 R に揃うからです。<strong>順行したかどうか</strong>で割ります。
  打つ手が真逆になるためです。</p>
</header>

<div class="stack g64">

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">全体</span>
    <h2>足の組ごとの内訳</h2></div>
  <div class="tblbox panel pad"><table class="tbl"><thead><tr>
    <th>足の組</th><th>機構</th><th class="n">負け</th>
    <th class="n">転換で終了</th><th class="n">一度も順行せず</th>
    <th class="n">1R 以上順行</th></tr></thead><tbody>
    {"".join(
        f'<tr><td class="mono">{lad}</td><td>{m}</td>'
        f'<td class="n">{sum(v for k, v in d["counts"][lad].items() if k.startswith(m)):,}</td>'
        f'<td class="n">{sum(v for k, v in d["counts"][lad].items() if k.startswith(m) and "転換" in k) / sum(v for k, v in d["counts"][lad].items() if k.startswith(m)):.1%}</td>'
        f'<td class="n"><strong>{sum(v for k, v in d["counts"][lad].items() if k.startswith(m) and "順行せず" in k) / sum(v for k, v in d["counts"][lad].items() if k.startswith(m)):.1%}</strong></td>'
        f'<td class="n">{sum(v for k, v in d["counts"][lad].items() if k.startswith(m) and "大きく順行" in k) / sum(v for k, v in d["counts"][lad].items() if k.startswith(m)):.1%}</td></tr>'
        for lad in ("H4-H1-M15", "D1-H4-H1") for m in ("跳ね返り", "ブレイク"))}
  </tbody></table></div>
  <div class="note bad">
    <span class="lb">ダウ理論のときと逆</span>
    ダウ理論の押し目では、負けの <strong>38.5%</strong> が一度 +1.0 R まで
    順行してから戻されていました。<strong>持ち方の問題</strong>です。
    抵抗帯では逆で、<strong>1R 以上順行した負けは 8〜19% しかありません。</strong>
    代わりに <strong>4〜6 割が一度も順行しません</strong>。
    これは<strong>入る場所の問題</strong>です。
  </div>
</section>

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">{LADDER}</span>
    <h2>機構ごとの全区分</h2></div>
  <div class="two">{"".join(cards)}</div>
</section>

{"".join(figs)}

</div>
</div>
'''
    script = ("<script>\n" + draw + "</script>\n<script>\nconst F = "
              + json.dumps(fig, ensure_ascii=False) + ";\n"
              + "\n".join(calls) + "\n</script>")
    return head + body + script


if __name__ == "__main__":
    open(DST, "w", encoding="utf-8").write(build())
    print(f"{DST} を生成しました")
