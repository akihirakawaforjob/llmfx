"""帯の手前でしか入らない / 推進波の起点はどこか — 認識合わせページ.

**手で節を差し込まないこと。**必ずここから生成する。

    python tools/make_fix_page.py
"""

from __future__ import annotations

import json
import re

SRC = "docs/spec-v4.html"
FIX = "docs/handoff/fix-figures.json"
LOSS = "docs/handoff/loss-patterns.json"
DST = "docs/entry-fix.html"
L = "H4-H1-M15"


def chart(cid, title, meta, alt):
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
    quote = re.search(r"(\.quote\{.*?\.quote li\{[^}]*\})", src, re.S).group(1)
    fig = json.load(open(FIX, encoding="utf-8"))
    loss = json.load(open(LOSS, encoding="utf-8"))

    # 実例。跳ね返りの 1 位(帯より上で売っている)と ブレイクの 1 位
    def pick(mech):
        sub = {k: v for k, v in loss["counts"][L].items() if k.startswith(mech)}
        key = max(sub, key=lambda k: sub[k])
        return loss["picks"][L][key]

    bad_fade, bad_break = pick("跳ね返り"), pick("ブレイク")
    fig["real_fade"] = {k: bad_fade[k] for k in
                        ("candles", "band", "entry", "stop", "exit",
                         "entry_at", "exit_at", "long_side")}
    fig["real_break"] = {k: bad_break[k] for k in
                         ("candles", "band", "entry", "stop", "exit",
                          "entry_at", "exit_at", "long_side")}

    head = f'''<title>帯の手前でしか入らない</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Old+Mincho:wght@500;700&family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
{css}
{quote}
.add{{color:var(--be); font-weight:700}}
.two{{display:grid; gap:20px; grid-template-columns:1fr 1fr}}
@media (max-width:860px){{.two{{grid-template-columns:1fr}}}}
.pick{{display:grid; grid-template-columns:2.2rem 1fr; gap:14px; padding:15px 0;
  border-bottom:1px solid var(--line2)}}
.pick:last-child{{border-bottom:0}}
.pick .n{{font-family:"IBM Plex Mono",monospace; font-size:13px; font-weight:700;
  color:var(--sell); padding-top:.3em}}
.pick .b{{display:flex; flex-direction:column; gap:4px}}
.pick .h{{font-weight:700}}
.pick .d{{font-size:14.5px; color:var(--ink2); line-height:1.75}}
</style>
'''

    body = f'''<div class="wrap">
<header class="top stack g16">
  <span class="eyebrow">llmfx / 抵抗帯トレード / 認識合わせ</span>
  <h1>帯の手前でしか入らない</h1>
  <p class="lede prose">負けの図から、こちらの実装に <strong>2 つの誤り</strong>が見つかりました。
  直し方を図で確認します。<strong>起点の定義は 3 案あるので選んでください。</strong></p>
</header>

<div class="stack g64">

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">指摘</span>
    <h2>図を見て分かったこと</h2></div>
  <blockquote class="quote">
    <span class="who">利用者 — 2026-09-14</span>
    帯に対して向かう様に売りに入っている。<strong>抵抗帯という名の通り、
    抵抗帯の抵抗帯力を信じて</strong>買いに入るべき。
    それと、損切り幅がやたらと広く見える。<strong>僕の想定では見ている足の
    推進波一つ分と同じくらいのイメージ</strong>。<br><br>
    一見ブレイクに見えた挙動でロングポジションに入るものに対して、
    <strong>鉄槌を下している様な構図</strong>だ。恐らく、これに狩られ続けて
    いるのだろう。彼らも馬鹿ではない。<strong>守りたいポジションは何重にも
    構えているはず</strong>だ。そして、押し込まれれば押し込まれる程、
    押し返した時に彼らにとっては利益になる。<br><br>
    機関投資家の守りたいポイントは、<strong>大きいローソク足の付け根部分</strong>な
    はずだし、そこに指値を置いた方が良い。
  </blockquote>
  <div class="tblbox panel pad"><table class="tbl"><thead><tr>
    <th>機構</th><th>位置</th><th class="n">件数</th><th class="n">割合</th>
    <th class="n">損切り幅/ATR</th><th class="n">期待値 R</th></tr></thead><tbody>
    <tr><td>跳ね返り</td><td>帯の手前</td><td class="n">2,959</td>
      <td class="n">50.9%</td><td class="n">2.54</td><td class="n">-0.028</td></tr>
    <tr><td>跳ね返り</td><td><strong>帯を越えた側</strong></td><td class="n">2,856</td>
      <td class="n"><strong>49.1%</strong></td><td class="n">2.00</td>
      <td class="n neg"><strong>-0.138</strong></td></tr>
    <tr><td>ブレイク</td><td>帯の手前</td><td class="n">1,565</td>
      <td class="n">99.7%</td><td class="n neg"><strong>7.01</strong></td>
      <td class="n">-0.099</td></tr>
  </tbody></table></div>
  <p class="prose"><strong>跳ね返りの 49.1% が帯を越えた側で発動していました。</strong>
  そちらは 5 倍悪い。ブレイク側は損切り幅の中央値が <strong>7.01 ATR</strong>、
  4 分の 3 が 4 ATR を超え、最大 29.91 ATR。<span class="mark">推進波 1 本分とは桁が違います。</span></p>
</section>

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">誤り 1 / 実例</span>
    <h2>抵抗帯より上で売っている</h2></div>
  {chart("rf", f"{bad_fade['pair'].upper()} {bad_fade['when']}",
         f"帯 {bad_fade['band']:.3f} / 約定 {bad_fade['entry']:.3f}(帯より上)/ 結果 {bad_fade['r']:+.2f} R",
         "抵抗帯を上抜けた後に、帯より上で売りに入っている実際の建玉")}
  <p class="prose">価格は既に帯を上抜けています。<strong>この時点で帯は抵抗ではなく支持。</strong>
  そこへ向かって売っている。注文を「すでに通過した値」に置き直すたび、
  <strong>帯の外へついて行ってしまう</strong>のが原因です。</p>
</section>

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">直し方 1</span>
    <h2>注文は帯の手前に留める — 両向き</h2></div>
  <div class="two">
    {chart("ns", "売り(上の帯)", "注文は帯より下にしか置かない",
           "上の帯で売る場合、注文は帯より下に留める")}
    {chart("nb", "買い(下の帯)", "注文は帯より上にしか置かない",
           "下の帯で買う場合、注文は帯より上に留める")}
  </div>
  <div class="note">
    <span class="lb">規則</span>
    注文を置く水準は<strong>「すでに通過した値」かつ「帯の手前」</strong>の両方を満たすこと。
    価格が帯を越えても、<strong>注文は帯の手前に留めたまま</strong>にします。
    ダマシで戻ってきたら拾える、という当初の狙いはそのまま残ります。
    <strong>買い側も売り側も同じ扱い</strong>です。
  </div>
</section>

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">誤り 2 / 実例</span>
    <h2>帯から 10 ATR 離れて入り、損切りが 9 ATR</h2></div>
  {chart("rb", f"{bad_break['pair'].upper()} {bad_break['when']}",
         f"帯 {bad_break['band']:.3f} / 約定 {bad_break['entry']:.3f} / "
         f"損切り幅 {abs(bad_break['entry']-bad_break['stop'])/bad_break['atr']:.1f} ATR",
         "帯から大きく離れた位置で約定し、損切りだけ帯の近くに残っている建玉")}
  <p class="prose">構造が 2 段育つのを待つ間に、価格が帯から離れます。
  その先で拾うので、<strong>損切りだけ帯の近くに残って幅が膨らむ。</strong>
  「一見ブレイクに見えた挙動で入って鉄槌を下される」形が、これです。</p>
</section>

<section class="stack g24">
  <div class="sechead"><span class="eyebrow">直し方 2 / 選んでください</span>
    <h2>「推進波の起点」はどこか</h2></div>
  {chart("or", "抜けた後の推進波", "3 案の線を重ねてある",
         "帯を抜けた後の推進波に、起点の候補 3 案を重ねた図")}
  <div class="panel pad"><div class="steps">
    <div class="pick"><span class="n">A</span><span class="b">
      <span class="h">直近の確定した押し安値(構造足)</span>
      <span class="d">図の <strong>安値2</strong>。いまの実装が損切りに使っている水準です。
      指値もここに置けば、<strong>損切りは安値2 のすぐ下</strong>になり幅が最も狭い。
      ただし<strong>浅いので、届く前に走り去ることが多い</strong></span></span></div>
    <div class="pick"><span class="n">B</span><span class="b">
      <span class="h">大きく動いた足の付け根</span>
      <span class="d">図の <strong>付け根</strong>。あなたの言葉に最も近い案です。
      彼らが守っている水準。損切りはその下。<strong>A より深く、C より浅い</strong>。
      「大きく動いた足」の定義が要ります(値幅が ATR の何倍か)</span></span></div>
    <div class="pick"><span class="n">C</span><span class="b">
      <span class="h">抜けた帯そのもの(リテスト)</span>
      <span class="d">図の <strong>帯 100.00</strong>。抜けた帯が支持に変わるのを待つ、
      教科書的な形。<strong>最も深く、最も届きにくい</strong>。
      届いたときは損切りが帯のすぐ下で済む</span></span></div>
  </div></div>
  <div class="note">
    <span class="lb">聞きたいこと</span>
    <ol style="margin:6px 0 0; padding-left:1.3em">
      <li><strong>A・B・C のどれか。</strong>あなたの言葉は B に読めますが、確認させてください</li>
      <li>B なら、<strong>「大きく動いた足」の定義</strong>。値幅が ATR の何倍以上か</li>
      <li><strong>指値に届かなかったらどうするか。</strong>見送るか、次の押し目まで待つか</li>
      <li><strong>損切りは起点のすぐ外でよいか。</strong>推進波 1 本分に収まります</li>
    </ol>
  </div>
</section>

</div>
</div>
'''

    keep = {k: fig[k] for k in ("near_sell", "near_buy", "origin",
                                "real_fade", "real_break")}
    script = f'''<script>
{draw}</script>
<script>
const F = {json.dumps(keep, ensure_ascii=False)};

(function(){{
  const f=F.real_fade;
  draw(document.getElementById("c-rf"), document.getElementById("l-rf"), f.candles,
    [{{y:f.band,c:"--zone",w:3,t:"抵抗帯 "+f.band.toFixed(3),k:"抵抗帯"}},
     {{y:f.stop,c:"--stop",w:2,dash:"4 4",t:"損切り",k:"損切り"}}],
    [{{i:f.entry_at,y:f.entry,c:"--sell",kind:"entry",t:"売り",down:true}},
     {{i:f.exit_at,y:f.exit,c:"--muted",kind:"ghost",t:"決済"}}],
    {{h:330,alt:"帯より上で売っている"}});
}})();
(function(){{
  const f=F.real_break;
  draw(document.getElementById("c-rb"), document.getElementById("l-rb"), f.candles,
    [{{y:f.band,c:"--zone",w:3,t:"抜けた帯 "+f.band.toFixed(3),k:"抜けた帯"}},
     {{y:f.stop,c:"--stop",w:2.5,dash:"4 4",t:"損切り",k:"損切り"}}],
    [{{i:f.entry_at,y:f.entry,c:"--sell",kind:"entry",t:"売り",down:true}},
     {{i:f.exit_at,y:f.exit,c:"--muted",kind:"ghost",t:"決済"}}],
    {{h:330,alt:"帯から離れて約定している"}});
}})();
(function(){{
  const f=F.near_sell;
  draw(document.getElementById("c-ns"), document.getElementById("l-ns"), f.candles,
    [{{y:f.band,c:"--zone",w:3,t:"帯 "+f.band.toFixed(2),k:"抵抗帯"}},
     {{y:f.keep,c:"--be",w:2.5,from:f.arm_at,t:"正 "+f.keep.toFixed(2)+"(帯の手前)",
       k:"正: 帯の手前に留める"}},
     {{y:f.drift,c:"--stop",w:2,dash:"4 4",from:f.arm_at,
       t:"誤 "+f.drift.toFixed(2)+"(帯の外)",k:"誤: 帯の外へついて行く"}},
     {{y:f.stop,c:"--muted",w:2,dash:"2 5",t:"損切り",k:"損切り"}}],
    [{{i:f.fill_at,y:f.keep,c:"--be",kind:"entry",t:"売り",down:true}}],
    {{h:340,alt:"売りは帯より下にしか置かない"}});
}})();
(function(){{
  const f=F.near_buy;
  draw(document.getElementById("c-nb"), document.getElementById("l-nb"), f.candles,
    [{{y:f.band,c:"--zone",w:3,t:"帯 "+f.band.toFixed(2),k:"支持帯"}},
     {{y:f.keep,c:"--be",w:2.5,from:f.arm_at,t:"正 "+f.keep.toFixed(2)+"(帯の手前)",
       k:"正: 帯の手前に留める"}},
     {{y:f.drift,c:"--stop",w:2,dash:"4 4",from:f.arm_at,
       t:"誤 "+f.drift.toFixed(2)+"(帯の外)",k:"誤: 帯の外へついて行く"}},
     {{y:f.stop,c:"--muted",w:2,dash:"2 5",t:"損切り",k:"損切り"}}],
    [{{i:f.fill_at,y:f.keep,c:"--be",kind:"entry",t:"買い"}}],
    {{h:340,alt:"買いは帯より上にしか置かない"}});
}})();
(function(){{
  const f=F.origin;
  draw(document.getElementById("c-or"), document.getElementById("l-or"), f.candles,
    [{{y:f.a_swing,c:"--sell",w:2.5,t:"A 直近の押し安値",k:"A 直近の押し安値"}},
     {{y:f.b_base,c:"--buy",w:2.5,t:"B 大きな足の付け根",k:"B 大きな足の付け根"}},
     {{y:f.c_band,c:"--zone",w:3,t:"C 抜けた帯(リテスト)",k:"C 抜けた帯"}}],
    [{{i:f.big_at,y:f.b_base,c:"--buy",kind:"ghost",t:"大きく動いた足"}}],
    {{h:380,swings:f.swings,alt:"起点の候補 3 案"}});
}})();
</script>'''
    return head + body + script


if __name__ == "__main__":
    open(DST, "w", encoding="utf-8").write(build())
    print(f"{DST} を生成しました")
