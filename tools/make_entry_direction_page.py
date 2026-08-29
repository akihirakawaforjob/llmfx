"""「帯に来た時点では方向を決めない」の認識合わせページを組む.

**手で節を差し込まないこと。**最初に見つかった `</div></div>` や
`</script>` を置き換える形で足したところ、容れ物が途中で閉じ、
描画の呼び出しが定義ブロックへ入って、以降の図が全部止まった。
ページは必ずここから生成する。

    python tools/make_entry_direction_page.py
"""

from __future__ import annotations

import json
import re

SRC = "docs/spec-v4.html"
FIGS = "docs/handoff/direction-figures.json"
DST = "docs/entry-direction.html"


def chart(cid: str, title: str, meta: str, alt: str) -> str:
    return (f'<div class="panel"><div class="caphead"><span class="t">{title}</span>'
            f'<span class="m">{meta}</span></div>'
            f'<div class="chartbox" id="c-{cid}" role="img" aria-label="{alt}"></div>'
            f'<div class="legend" id="l-{cid}"></div></div>')


def build() -> str:
    src = open(SRC, encoding="utf-8").read()
    css = re.search(r"<style>\n(:root\{.*?)\n</style>", src, re.S).group(1)
    draw = re.search(
        r"(const NS=\"http://www\.w3\.org/2000/svg\";.*?\n\}\n)</script>",
        src, re.S).group(1)
    quote_css = re.search(r"(\.quote\{.*?\.quote li\{[^}]*\})", src, re.S).group(1)
    fig = json.load(open(FIGS, encoding="utf-8"))

    head = f'''<title>帯に来た時点では方向を決めない</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Old+Mincho:wght@500;700&family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
{css}
{quote_css}
.add{{color:var(--be); font-weight:700}}
.two{{display:grid; gap:20px; grid-template-columns:1fr 1fr}}
@media (max-width:860px){{.two{{grid-template-columns:1fr}}}}
</style>
'''

    body = f'''<div class="wrap">
<header class="top stack g16">
  <span class="eyebrow">llmfx / 抵抗帯トレード / 認識合わせ</span>
  <h1>帯に来た時点では方向を決めない</h1>
  <p class="lede prose">跳ね返りとブレイクで、<strong>入り方の機構が別</strong>です。
  こちらは両側を「執行足の折り返しを抜けたら入る」で揃えていました。
  その差を図で確認します。</p>
</header>

<div class="stack g64">

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">仕様</span>
    <h2>あなたの手順</h2></div>
  <blockquote class="quote">
    <span class="who">利用者 — 2026-08-29</span>
    まず抵抗帯に向かう現在値を貼り続ける。抵抗帯に触れるギリギリになったら、
    跳ね返り用の指値を<strong>その少し下</strong>に置く。つまり、そのまま抜けたら
    この指値は約定にならず、<strong>無意味に損切りをする必要がなくなる</strong>。<br><br>
    大事なのは<strong>自分が指値を置く値を、価格が既に通過している事</strong>。<br><br>
    抵抗帯をブレイクして抜けた後、高値切り上げ・安値切り上げが見えたら、
    <strong>次の</strong>高値切り上げ・安値切り上げの 2 つ下の時間足の
    ダウ転換時にエントリー。損切りラインもいつもの。<br><br>
    ポイントは、<strong>ブレイクに向かう値を指値で上手く拾わないようにすること</strong>。
  </blockquote>
</section>

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">いまの実装</span>
    <h2>帯へ触れる前に、向きが固定されている</h2></div>
  <p class="prose"><code>reverse_entry</code> という設定で、帯に触れる前に売買の向きが
  決まります。<code>False</code> なら必ず跳ね返りに賭け、抜けたら見送る。
  <code>True</code> なら必ずブレイクに賭け、跳ね返ったら見送る。
  <strong>同じ設定で両方は取れません。</strong></p>
  <div class="two">
    {chart("reject", "跳ね返りに賭ける設定", "reverse_entry=False",
           "抵抗帯に触れた後、執行足の安値1を下抜けて売りに入る図")}
    {chart("breakout", "ブレイクに賭ける設定", "reverse_entry=True",
           "抵抗帯に触れた後、執行足の高値1を上抜けて買いに入る図")}
  </div>
  <div class="note bad">
    <span class="lb">水準そのものが違う</span>
    どちらも<strong>執行足の直近の折り返し</strong>を待っています。
    執行足の折り返しは<strong>帯から離れた場所に出る</strong>ので、
    入る位置が悪く、損切りまでの幅も広くなります。
    実際の手法はこれとは別の場所に注文を置きます。
  </div>
</section>

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">跳ね返り側 1</span>
    <h2>すでに通過した値に置く。折り返せば発動する</h2></div>
  {chart("v2b", "跳ね返った場合", "帯 100.00 / 注文 99.88 / ATR 0.10",
         "帯へ届く直前に、価格がすでに通過した水準へ注文を置き、折り返して発動する図")}
  <p class="prose">帯へ届く直前に、<strong>価格がすでに通過した水準</strong>へ売り注文を置きます。
  値段が戻ってこないと発動しないので、<span class="mark">折り返したときだけ入る</span>。
  損切りは帯の外。</p>
</section>

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">跳ね返り側 2</span>
    <h2>抜けても発動しない。取り消すのは帯が死んでから</h2></div>
  {chart("v2k", "抜けた場合", "同じ注文 99.88 / 発動せず",
         "同じ注文が、価格がそのまま上抜けたため発動しない図")}
  <div class="note">
    <span class="lb">ここが要点</span>
    <strong>ブレイクに向かう値を拾わない。</strong>帯に置いた指値だと、
    抜ける瞬間に必ず約定して、そのまま損切りへ運ばれます。
    すでに通過した値に置けば、抜けるときは<strong>そもそも約定しない</strong>。
    無駄な損切りが構造的に発生しません。
  </div>
  <div class="note">
    <span class="lb">取り消しは急がない</span>
    注文はすでに通過した値にあるので、抜けている間は約定しません。
    <strong>残しておいても損をせず、ダマシで戻ってきたら拾えます。</strong>
    取り消すのは<strong>抜けた後に高値切り上げ・安値切り上げが起き、
    帯が機能しなくなったと判断できた時点</strong>。抜けた瞬間ではありません。
  </div>
</section>

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">ブレイク側</span>
    <h2>抜けた後、構造ができてから、次の押し目で入る</h2></div>
  {chart("v2p", "ブレイク後の押し目", "高値切り上げ + 安値切り上げ を確認してから",
         "帯を抜けた後、高値切り上げと安値切り上げを確認し、次の押し目のダウ転換で買う図")}
  <p class="prose">抜けた瞬間には入りません。<strong>高値1 → 安値1 → 高値2 → 安値2</strong> と
  切り上がるのを見てから、<strong>次の</strong>押し目で 2 つ下の足のダウ転換で買い。
  損切りはいつもどおり <strong>1 つ前の安値</strong>(安値2)。</p>
</section>

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">まとめ</span>
    <h2>決まったことと、掃引で決めること</h2></div>
  <div class="panel pad"><div class="steps">
    <div class="step"><span class="n">決</span><span class="b">
      <span class="h">注文を置く位置</span>
      <span class="d"><strong class="add">自分が注文を置く値を、価格がすでに通過していること。</strong>
      これが定義です。「帯から 0.2 ATR」のような恣意的な数字ではなく、
      <strong>価格の軌跡そのものが位置を決めます</strong></span></span></div>
    <div class="step"><span class="n">決</span><span class="b">
      <span class="h">取り消しの条件</span>
      <span class="d"><strong class="add">抜けた後に構造が確定し、帯が機能しなくなった時点。</strong>
      閾値ではなく、既にあるダウ構造の判定を使います</span></span></div>
    <div class="step"><span class="n">検</span><span class="b">
      <span class="h">どれだけ手前に置くか</span>
      <span class="d">通過済みという条件は満たしたうえで、直近の折り返しか、
      ATR に対する比か。浅いとヒゲで発動し、深いと取り逃します</span></span></div>
    <div class="step"><span class="n">検</span><span class="b">
      <span class="h">跳ね返り側の損切り</span>
      <span class="d">約定位置が帯に近くなるので、<strong>リスク幅は今よりかなり狭くなります</strong>。
      同じ値幅がより大きな R になり、コスト比率も下がります</span></span></div>
    <div class="step"><span class="n">検</span><span class="b">
      <span class="h">同じ帯での入り直し</span>
      <span class="d">跳ね返りで損切りになった後、その帯が抜けたらブレイク側で入り直すか。
      裁量では見送るが、<strong>機械なら入ってもよいかもしれない</strong>とのこと。両方測ります</span></span></div>
  </div></div>
  <div class="note bad">
    <span class="lb">掃引の作法</span>
    要検証が 3 つあります。<strong>軸を増やすほど、偶然プラスになるセルが出ます。</strong>
    このプロジェクトはその形で 11 回崩れました。掃引は
    <strong>コストを抜いた全銘柄合算</strong>で行い、
    <strong>表が台地になっているか</strong>(隣のセルでも成立するか)を必ず見ます。
    1 点だけ跳ねていたら採りません。
  </div>
</section>

</div>
</div>
'''

    script = f'''<script>
{draw}</script>
<script>
const F = {json.dumps(fig, ensure_ascii=False)};

// --- いまの実装。両側とも執行足の折り返しを待つ ------------------------
(function(){{
  const f=F.reject;
  draw(document.getElementById("c-reject"), document.getElementById("l-reject"),
    f.candles,
    [{{y:f.band,c:"--zone",w:3,t:"帯 "+f.band.toFixed(2),k:"抵抗帯"}},
     {{y:f.sell_line,c:"--sell",w:2.5,t:"安値1 下抜け → 売り",k:"下抜けたら売り"}},
     {{y:f.stop,c:"--stop",w:2,dash:"3 4",t:"損切り",k:"損切り(帯の外)"}}],
    [{{i:f.touch_at,y:f.band,c:"--zone",kind:"ghost",t:"帯へ触れる"}},
     {{i:f.sell_at,y:f.sell_line,c:"--sell",kind:"entry",t:"売り",down:true}}],
    {{h:320,swings:f.swings,alt:"跳ね返りに賭ける設定"}});
}})();
(function(){{
  const f=F.breakout;
  draw(document.getElementById("c-breakout"), document.getElementById("l-breakout"),
    f.candles,
    [{{y:f.band,c:"--zone",w:3,t:"帯 "+f.band.toFixed(2),k:"抵抗帯"}},
     {{y:f.buy_line,c:"--buy",w:2.5,t:"高値1 上抜け → 買い",k:"上抜けたら買い"}},
     {{y:f.stop,c:"--stop",w:2,dash:"3 4",t:"損切り",k:"損切り(帯の外)"}}],
    [{{i:f.touch_at,y:f.band,c:"--zone",kind:"ghost",t:"帯へ触れる"}},
     {{i:f.buy_at,y:f.buy_line,c:"--buy",kind:"entry",t:"買い"}}],
    {{h:320,swings:f.swings,alt:"ブレイクに賭ける設定"}});
}})();

// --- 実際の手法。跳ね返り側は「すでに通過した値」に置く ----------------
(function(){{
  const f=F.v2_bounce;
  draw(document.getElementById("c-v2b"), document.getElementById("l-v2b"),
    f.candles,
    [{{y:f.band,c:"--zone",w:3,t:"抵抗帯 "+f.band.toFixed(2),k:"抵抗帯"}},
     {{y:f.order,c:"--sell",w:2.5,from:f.arm_at,t:"注文 "+f.order.toFixed(2),
       k:"すでに通過した値に置く注文"}},
     {{y:f.stop,c:"--stop",w:2.5,dash:"3 4",t:"損切り "+f.stop.toFixed(2),
       k:"損切り(帯の外)"}}],
    [{{i:f.arm_at,y:f.order,c:"--muted",kind:"ghost",t:"ここで置く"}},
     {{i:f.fill_at,y:f.order,c:"--sell",kind:"entry",t:"売り",down:true}}],
    {{h:340,alt:"折り返して発動する"}});
}})();
(function(){{
  const f=F.v2_break;
  draw(document.getElementById("c-v2k"), document.getElementById("l-v2k"),
    f.candles,
    [{{y:f.band,c:"--zone",w:3,t:"抵抗帯 "+f.band.toFixed(2),k:"抵抗帯"}},
     {{y:f.order,c:"--sell",w:2,dash:"5 5",from:f.arm_at,
       t:"注文 "+f.order.toFixed(2)+"(発動せず)",k:"発動しなかった注文"}}],
    [{{i:f.arm_at,y:f.order,c:"--muted",kind:"ghost",t:"ここで置く"}},
     {{i:f.cancel_at,y:f.band,c:"--be",kind:"ghost",t:"抜けた(まだ取り消さない)"}}],
    {{h:340,alt:"抜けても発動しない"}});
}})();
(function(){{
  const f=F.v2_pullback;
  draw(document.getElementById("c-v2p"), document.getElementById("l-v2p"),
    f.candles,
    [{{y:f.band,c:"--zone",w:2.5,t:"抜けた帯 "+f.band.toFixed(2),k:"抜けた帯"}},
     {{y:f.stop,c:"--stop",w:2.5,t:"損切り = 1 つ前の安値",k:"損切り(1 つ前の安値)"}}],
    [{{i:f.confirmed_at,y:f.candles[f.confirmed_at].l,c:"--muted",kind:"ghost",
       t:"切り上げを確認"}},
     {{i:f.entry_at,y:f.entry,c:"--buy",kind:"entry",t:"買い"}}],
    {{h:360,swings:f.swings,alt:"ブレイク後の押し目で買う"}});
}})();
</script>'''
    return head + body + script


if __name__ == "__main__":
    open(DST, "w", encoding="utf-8").write(build())
    print(f"{DST} を生成しました")
