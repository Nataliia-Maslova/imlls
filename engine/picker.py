"""
engine/picker.py — Lesson navigator UI widgets.

Contains:
  - Lesson category / topic constants (_VOCAB_CATEGORIES, _TOPIC_META,
    _GRAMMAR_CATEGORIES)
  - Wave/snake path lesson pickers (Plotly + HTML + native Streamlit)
  - Vocab hierarchical navigator (_render_vocab_nav)
  - Flat grammar/reading wave navigator (_render_flat_wave_nav)
  - Lesson starter helper (_start_grammar_lesson)

Extracted from grammar.py to keep it manageable.
Public API used by grammar.py: _render_flat_wave_nav, _render_vocab_nav
"""
from pathlib import Path
import streamlit as st

_APP_ROOT = Path(__file__).parent.parent   # imlls/engine/ → imlls/
LESSON_IMG_DIR = _APP_ROOT / "static" / "lesson_images"

from engine.loader import TTS_LANG, WHISPER_LANG
from engine.session import LessonSession

# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical vocabulary navigator (Category -> Topic -> Lesson)
# ═══════════════════════════════════════════════════════════════════════════

_VOCAB_CATEGORIES = [
    {
        "id": "speaking", "icon": "\U0001f5e3\ufe0f", "name": "Communication",
        "desc": "Greetings, questions, emergencies",
        "sheets": [
            "Greetings, Basics & Courtesy",
            "Questions, Directions & Emergen",
            "Daily Life, Routine & Feelings",
        ],
    },
    {
        "id": "wordbank", "icon": "\U0001f4da", "name": "Word Bank",
        "desc": "Core vocabulary: adjectives, verbs, food, city (words + sentences)",
        "sheets": ["Basic", "Verbs", "Food", "City"],
    },
    {
        "id": "situations", "icon": "\U0001f30d", "name": "Situations",
        "desc": "Restaurant, travel, shopping, work, school",
        "sheets": [
            "Restaurant, Food & Shopping", "Travel, Lodging & Weather",
            "Shopping", "At the Doctor", "Work", "School", "Travel", "Restaurant",
        ],
    },
    {
        "id": "people", "icon": "\U0001f465", "name": "People",
        "desc": "Friends, family, emotions, relationships",
        "sheets": ["Friends and Relationships", "Family", "Emotions"],
    },
    {
        "id": "home", "icon": "\U0001f3e0", "name": "Home & Routine",
        "desc": "Home, daily routine, weather, clothes, transport",
        "sheets": [
            "House and Home", "Daily Routine", "Weather",
            "Clothes", "Transport", "Hobbies", "Food and Drinks",
        ],
    },
    {
        "id": "leisure", "icon": "\U0001f389", "name": "Leisure",
        "desc": "Sports, holidays, technology, city",
        "sheets": ["Sports", "Holidays", "Technology", "City and Directions"],
    },
]

_TOPIC_META = {
    # Word Bank (Type A \u2014 words + sentences)
    "Basic":                           ("\U0001f524", "Basic Adjectives & Words"),
    "Verbs":                           ("\u26a1",     "Verbs"),
    "Food":                            ("\U0001f34e", "Food Vocabulary"),
    "City":                            ("\U0001f3d9\ufe0f", "City & Shopping"),
    # Communication
    "Greetings, Basics & Courtesy":    ("\U0001f44b", "Greetings & Courtesy"),
    "Questions, Directions & Emergen": ("\u2753",      "Questions, Directions & Emergencies"),
    "Daily Life, Routine & Feelings":  ("\u2600\ufe0f","Daily Life, Routine & Feelings"),
    # Situations
    "Restaurant, Food & Shopping":     ("\U0001f37d\ufe0f", "Restaurant, Food & Shopping"),
    "Travel, Lodging & Weather":       ("\u2708\ufe0f",     "Travel, Lodging & Weather"),
    "Shopping":                        ("\U0001f6cd\ufe0f", "Shopping"),
    "At the Doctor":                   ("\U0001f3e5",        "At the Doctor"),
    "Work":                            ("\U0001f4bc",        "Work"),
    "School":                          ("\U0001f393",        "School"),
    "Travel":                          ("\U0001f5fa\ufe0f", "Travel"),
    "Restaurant":                      ("\U0001f374",        "Restaurant"),
    "Friends and Relationships":       ("\U0001f91d",        "Friends & Relationships"),
    "Family":                          ("\U0001f46a",        "Family"),
    "Emotions":                        ("\U0001f60a",        "Emotions"),
    "House and Home":                  ("\U0001f3e1",        "House and Home"),
    "Daily Routine":                   ("\u23f0",            "Daily Routine"),
    "Weather":                         ("\U0001f324\ufe0f", "Weather"),
    "Clothes":                         ("\U0001f457",        "Clothes"),
    "Transport":                       ("\U0001f68c",        "Transport"),
    "Hobbies":                         ("\U0001f3a8",        "Hobbies"),
    "Food and Drinks":                 ("\U0001f957",        "Food and Drinks"),
    "Sports":                          ("\u26bd",            "Sports"),
    "Holidays":                        ("\U0001f384",        "Holidays"),
    "Technology":                      ("\U0001f4bb",        "Technology"),
    "City and Directions":             ("\U0001f5fa\ufe0f", "City and Directions"),
}


# ── Grammar lesson categories (for lesson-picker navigation) ─────────────────
# Each category covers a lesson-id range that matches the grammar curriculum.
_GRAMMAR_CATEGORIES = [
    {"id": "basics",    "icon": "🔤", "name": "Basics",
     "desc": "Things, identity, possession, location",    "range": (1,   23)},
    {"id": "nouns",     "icon": "📊", "name": "Nouns & Quantities",
     "desc": "Plurals, numbers, existence, containers",   "range": (24,  42)},
    {"id": "habits",    "icon": "📅", "name": "Habits & Commands",
     "desc": "Daily habits, commands, third-person",      "range": (43,  65)},
    {"id": "present",   "icon": "⚡", "name": "Present & Future",
     "desc": "Present continuous, future plans",          "range": (66,  83)},
    {"id": "modals",    "icon": "💪", "name": "Modals & Comparisons",
     "desc": "Ability, obligation, permission, conditions, comparisons",
                                                          "range": (84, 113)},
    {"id": "past",      "icon": "⏳", "name": "Past Tense",
     "desc": "Past actions — regular and irregular verbs","range": (114, 138)},
    {"id": "advanced",  "icon": "🎓", "name": "Advanced",
     "desc": "Passive, perfect tense, verb patterns, indefinites",
                                                          "range": (139, 999)},
]


def _build_wave_html(lesson_data_json, default_gid, native, target, user_id, resume_step, extra_params=None):
    """
    Returns a full HTML document for use with components.html().
    Navigation uses <a target="_top"> which reliably escapes the iframe sandbox.
    """
    from urllib.parse import quote as _q
    import json as _json
    n_enc = _q(native  or "")
    t_enc = _q(target  or "")
    u_enc = _q(user_id or "student1")
    ep_js = _json.dumps(extra_params or {})

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    background:transparent;overflow-x:hidden}}
.wo{{width:100%;overflow-x:auto;overflow-y:visible;padding:4px 0 8px;
    scrollbar-width:thin;scrollbar-color:#ccc transparent}}
.wo::-webkit-scrollbar{{height:4px}}
.wo::-webkit-scrollbar-thumb{{background:#ccc;border-radius:2px}}
.wi{{position:relative;height:180px}}
.wsvg{{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}}
.ln{{position:absolute;transform:translateX(-50%);cursor:pointer;text-align:center;
    width:88px;transition:transform .15s;user-select:none}}
.ln:hover{{transform:translateX(-50%) scale(1.1)}}
.ln.act{{transform:translateX(-50%) scale(1.06)}}
.nc{{width:64px;height:64px;border-radius:50%;margin:0 auto 5px;display:flex;
    align-items:center;justify-content:center;font-size:1.4rem;
    border:3px solid transparent;transition:border-color .15s;position:relative}}
.ln.act .nc{{border-color:#7F77DD;box-shadow:0 0 0 4px rgba(127,119,221,.18)}}
.nl{{font-size:11px;line-height:1.3;color:#777;max-width:86px;margin:0 auto}}
.ln.act .nl{{color:#534AB7;font-weight:600}}
.nb{{position:absolute;top:-5px;right:-5px;width:20px;height:20px;background:#534AB7;
    color:#fff;border-radius:50%;font-size:9px;font-weight:700;display:none;
    align-items:center;justify-content:center;border:2px solid #fff}}
.ln.act .nb{{display:flex}}
.sb{{display:block;width:200px;margin:6px auto 0;padding:9px 0;background:#7F77DD;
    color:#fff;border:none;border-radius:10px;font-size:13px;font-weight:600;
    cursor:pointer;transition:background .15s;text-align:center;text-decoration:none}}
.sb:hover{{background:#534AB7;color:#fff}}
.it{{text-align:center;font-size:11px;color:#999;margin-top:3px}}
</style></head><body>
<div class="wo" id="wo"><div class="wi" id="wi">
  <svg class="wsvg" id="ws"></svg>
</div></div>
<a class="sb" id="sb" href="#" target="_top">&#9654; Start Lesson</a>
<p class="it" id="it"></p>
<script>
var LS={lesson_data_json};
var DG={default_gid};
var RS={resume_step};
var NE="{n_enc}",TE="{t_enc}",UE="{u_enc}";
var EP={ep_js};
var CL=['#CECBF6','#9FE1CB','#F5C4B3','#B5D4F4','#C0DD97','#FAC775','#F4C0D1','#D3D1C7'];
var EM=['📖','🌟','💡','🎯','🔤','🗣️','✏️','📝','🎓','💬','🔑','🏆'];
var sg=DG;

function getURL(){{
  var u='?vnav_lesson='+sg+'&vnav_native='+NE+'&vnav_target='+TE+'&vnav_user='+UE;
  for(var k in EP)u+='&'+k+'='+encodeURIComponent(EP[k]);
  return u;
}}

function build(){{
  var n=LS.length; if(!n)return;
  var cw=104,pad=56,W=Math.max(700,n*cw+pad*2);
  var wi=document.getElementById('wi');
  wi.style.minWidth=W+'px';
  var svg=document.getElementById('ws'),H=180,my=H/2,amp=50;
  svg.setAttribute('viewBox','0 0 '+W+' '+H);
  var pts=[];
  for(var i=0;i<n;i++){{
    var x=pad+i*(W-pad*2)/Math.max(n-1,1);
    var y=my-Math.sin(i*Math.PI/2.8)*amp;
    pts.push({{x:x,y:y}});
  }}
  var d='M '+pts[0].x+' '+pts[0].y;
  for(var i=1;i<pts.length;i++){{
    var mx=(pts[i-1].x+pts[i].x)/2;
    d+=' C '+mx+' '+pts[i-1].y+' '+mx+' '+pts[i].y+' '+pts[i].x+' '+pts[i].y;
  }}
  svg.innerHTML='<path d="'+d+'" fill="none" stroke="#D3D1C7" stroke-width="4" stroke-linecap="round"/>';
  var wi2=document.getElementById('wi');
  var old=wi2.querySelectorAll('.ln');
  for(var j=0;j<old.length;j++)old[j].remove();
  for(var i=0;i<pts.length;i++){{
    (function(i){{
      var l=LS[i],p=pts[i];
      var div=document.createElement('div');
      div.className='ln'+(l.gid===sg?' act':'');
      div.style.left=p.x+'px';
      div.style.top=(p.y-43)+'px';
      div.innerHTML='<div class="nc" style="background:'+CL[i%CL.length]+'">'+
        EM[i%EM.length]+'<div class="nb">'+(i+1)+'</div></div>'+
        '<div class="nl">'+l.name+'</div>';
      div.onclick=function(){{sel(l.gid);}};
      wi2.appendChild(div);
    }})(i);
  }}
  upd();
  var ai=LS.findIndex(function(l){{return l.gid===sg;}});
  if(ai>2){{
    var wo=document.getElementById('wo');
    setTimeout(function(){{wo.scrollLeft=pts[ai].x-wo.offsetWidth/2;}},80);
  }}
}}

function sel(gid){{
  sg=gid;
  var wi=document.getElementById('wi');
  var nodes=wi.querySelectorAll('.ln');
  for(var i=0;i<nodes.length;i++)
    nodes[i].className='ln'+(LS[i].gid===gid?' act':'');
  upd();
}}

function upd(){{
  var l=null;
  for(var i=0;i<LS.length;i++)if(LS[i].gid===sg){{l=LS[i];break;}}
  if(!l)l=LS[0]; if(!l)return;
  var isR=(l.gid===DG&&RS>1);
  var sb=document.getElementById('sb');
  sb.textContent=isR?'▶ Resume at Step '+RS:'▶ Start Lesson';
  sb.href=getURL();
  document.getElementById('it').textContent=l.name+' · '+l.phrases+' phrases';
}}

build();
</script></body></html>"""


def _start_grammar_lesson(lid, cfg, native, target, user_id, lang_pair):
    """Start a grammar/vocab lesson directly — no URL navigation required."""
    try:
        # Vocab: resolve topic so we lazy-load only that one sheet
        extra = {}
        if cfg.get("lang_suffix") == "vocab":
            from engine.vocab_loader import get_topic_for_lesson
            _topic, _ = get_topic_for_lesson(str(cfg["db_path"]), lid)
            if _topic:
                extra["topic"] = _topic
        df = cfg["load"](str(cfg["db_path"]), native, target, **extra)
        lesson_df = cfg["get_lesson"](df, lid)
        if not lesson_df.empty:
            st.session_state.update({
                "session":     LessonSession(user_id, lesson_df, lid,
                                              native, target,
                                              language_pair=lang_pair),
                "lesson_step": 1,
                "tts_lang":    TTS_LANG.get(target, "en"),
                "wh_lang":     WHISPER_LANG.get(target),
                "lang_pair":   lang_pair,
            })
            st.rerun()
        else:
            st.warning(f"No data for lesson {lid} ({native} → {target}).")
    except Exception as _e:
        st.error(f"Error starting lesson {lid}: {_e}")


def _render_wave_plotly(
    lessons, lesson_names, lesson_counts,
    default_lid, resume_step, key_suffix="wave",
    show_lesson_image=True,
):
    """
    Snake-path lesson picker using Plotly interactive scatter chart.
    Lessons are arranged in a zigzag/snake pattern (like Duolingo):
      row 0 → left to right, row 1 → right to left, etc.
    Each row holds COLS lessons and forms a sine arc.
    Returns the lesson ID that was clicked, or None.
    Requires plotly + Streamlit >= 1.33 (on_select support).
    """
    import math, inspect
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if 'on_select' not in inspect.signature(st.plotly_chart).parameters:
        return None

    n = len(lessons)
    if n == 0:
        return None

    COLORS = ['#CECBF6','#9FE1CB','#F5C4B3','#B5D4F4','#C0DD97','#FAC775','#F4C0D1','#D3D1C7']
    COLS  = 7     # lessons per row
    ROW_H = 120   # vertical distance between row baselines (data units)
    AMP   = 40    # sine arc amplitude

    # ── Compute (x, y) position for each lesson ──────────────────────────────
    xs, ys = [], []
    for i in range(n):
        row      = i // COLS
        col      = i % COLS
        n_in_row = min(COLS, n - row * COLS)
        t        = col / max(n_in_row - 1, 1)   # 0 → 1 across this row

        x = col if row % 2 == 0 else (COLS - 1 - col)
        y = -row * ROW_H + math.sin(t * math.pi) * AMP
        xs.append(x)
        ys.append(y)

    n_rows  = math.ceil(n / COLS)
    chart_h = max(280, n_rows * (ROW_H + 15) + 90)
    y_min   = -(n_rows - 1) * ROW_H - AMP - 80
    y_max   = AMP + 65

    names  = [lesson_names.get(lid, f"Lesson {lid}") for lid in lessons]
    counts = [lesson_counts.get(lid, 0) for lid in lessons]
    shorts = [(nm[:12] + "…") if len(nm) > 12 else nm for nm in names]

    m_colors = [COLORS[i % len(COLORS)] for i in range(n)]
    b_colors = ['#5865F2' if lid == default_lid else '#ffffff' for lid in lessons]
    b_widths = [4 if lid == default_lid else 2 for lid in lessons]
    m_sizes  = [52 if lid == default_lid else 42 for lid in lessons]

    hover = [
        f"<b>{nm}</b><br>{cnt} phrases"
        + (f"<br>↩ resume step {resume_step}" if lid == default_lid and resume_step > 1 else "")
        for nm, cnt, lid in zip(names, counts, lessons)
    ]

    fig = go.Figure()

    # ── Snake path (connects all lesson positions in order) ──────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines',
        line=dict(color='#D0D0D0', width=6),
        hoverinfo='skip', showlegend=False,
    ))

    # ── Lesson circles ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='markers+text',
        marker=dict(
            size=m_sizes, color=m_colors,
            line=dict(color=b_colors, width=b_widths),
            symbol='circle',
        ),
        text=shorts,
        textposition='bottom center',
        textfont=dict(size=9, color='#444', family='Inter, sans-serif'),
        hovertext=hover,
        hoverinfo='text',
        showlegend=False,
    ))

    # ── Resume badge ─────────────────────────────────────────────────────────
    if resume_step > 1 and default_lid in lessons:
        di = lessons.index(default_lid)
        fig.add_annotation(
            x=xs[di] + 0.15, y=ys[di] + 22,
            text=f"↩{resume_step}",
            showarrow=False,
            font=dict(size=9, color='white'),
            bgcolor='#5865F2', bordercolor='#5865F2',
            borderwidth=1, borderpad=3,
        )

    fig.update_layout(
        height=chart_h,
        margin=dict(l=20, r=20, t=10, b=55),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, range=[-0.8, COLS - 0.2], fixedrange=True),
        yaxis=dict(visible=False, range=[y_min, y_max],     fixedrange=True),
        dragmode=False,
        clickmode='event+select',
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select='rerun',
        key=f"wplot_{key_suffix}",
        config=dict(displayModeBar=False),
    )

    # ── Handle click ─────────────────────────────────────────────────────────
    clicked_lid = None
    if event:
        try:
            pts = getattr(event.selection, 'points', None) or event.selection.get('points', [])
        except Exception:
            pts = []
        for pt in pts:
            if pt.get('curve_number', -1) == 1:
                idx = pt.get('point_number', pt.get('point_index', -1))
                if 0 <= idx < n:
                    clicked_lid = lessons[idx]
                    break

    # ── Lesson preview card (grammar only) ───────────────────────────────────
    if show_lesson_image:
        preview_lid = clicked_lid if clicked_lid is not None else default_lid
        preview_img = LESSON_IMG_DIR / f"lesson_{preview_lid:03d}.jpg"
        preview_name = lesson_names.get(preview_lid, f"Lesson {preview_lid}")
        if preview_img.exists():
            col_l, col_m, col_r = st.columns([2, 1, 2])
            with col_m:
                st.image(str(preview_img), use_container_width=True)
                st.caption(f"**{preview_name}**")

    return clicked_lid


def _render_wave_native(
    lessons, lesson_names, lesson_counts,
    default_lid, resume_step,
    cfg, native, target, user_id, lang_pair,
):
    """Wave-style lesson picker using native Streamlit buttons.
    Clicking a circle immediately starts that lesson — no iframe needed."""
    import math

    n = len(lessons)
    if n == 0:
        return

    EMOJIS = ['📖','🌟','💡','🎯','🔤','🗣️','✏️','📝','🎓','💬','🔑','🏆']

    st.markdown("""
<style>
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button {
    border-radius: 50% !important;
    width: 76px !important; height: 76px !important;
    min-width: 76px !important; padding: 3px !important;
    font-size: 0.57rem !important; line-height: 1.22 !important;
    white-space: pre-wrap !important; overflow: hidden !important;
    text-align: center !important;
}
</style>""", unsafe_allow_html=True)

    ROW = 10
    for row_start in range(0, n, ROW):
        row_lids = lessons[row_start:row_start + ROW]
        cols = st.columns(len(row_lids))
        for ci, (col, lid) in enumerate(zip(cols, row_lids)):
            gidx  = row_start + ci
            emoji = EMOJIS[gidx % len(EMOJIS)]
            name  = lesson_names.get(lid, f"{cfg.get('lesson_word','Lesson')} {lid}")
            short = (name[:10] + "…") if len(name) > 10 else name
            is_def = (lid == default_lid)
            badge  = f"↩{resume_step} " if (is_def and resume_step > 1) else ""
            label  = f"{badge}{emoji}\n{short}"
            count  = lesson_counts.get(lid, 0)
            with col:
                if st.button(
                    label,
                    key=f"wv_{lid}_{gidx}",
                    type="primary" if is_def else "secondary",
                    use_container_width=False,
                    help=f"{name} · {count} phrases",
                ):
                    _start_grammar_lesson(lid, cfg, native, target, user_id, lang_pair)

def _render_lesson_dropdown_fallback(
    lessons, lesson_names, default_lid, resume_step,
    cfg, native, target, user_id, lang_pair,
):
    """Clean dropdown + Start button — replaces the circular-button grid fallback."""
    if not lessons:
        return
    lw   = cfg.get("lesson_word", "Lesson")
    opts = [lesson_names.get(lid, f"{lw} {lid}") for lid in lessons]
    defi = lessons.index(default_lid) if default_lid in lessons else 0
    sel  = st.selectbox(f"Select {lw}", opts, index=defi,
                        key=f"dd_{lang_pair}")
    sel_lid    = lessons[opts.index(sel)]
    is_resume  = (sel_lid == default_lid and resume_step > 1)
    btn_label  = (f"▶ Resume at Step {resume_step}" if is_resume
                  else f"▶ Start {lw}")
    if st.button(btn_label, type="primary", use_container_width=True,
                 key=f"dd_btn_{lang_pair}"):
        _start_grammar_lesson(sel_lid, cfg, native, target, user_id, lang_pair)


def _render_flat_wave_nav(
    df, cfg, lang_pair, native, target, user_id,
    lessons, default_idx, resume_step, resume_msg, progress,
    counts_by_lid, topics_map,
):
    """Flat wave lesson picker with category/unit filter and dropdown fallback."""
    if resume_msg:
        st.info(resume_msg)

    default_gid = lessons[default_idx]

    def _lesson_name(lid):
        if topics_map and lid in topics_map:
            return topics_map[lid]
        return f"{cfg['lesson_word']} {lid}"

    lesson_names = {lid: _lesson_name(lid) for lid in lessons}

    # ── Category / Unit filter ──────────────────────────────────────────────
    # Grammar: predefined categories by lesson-id range.
    # Other modules: auto-generated numeric unit blocks (~15 lessons each).
    module = cfg.get("lang_suffix", "grammar")
    available_set = set(lessons)

    if module == "grammar":
        cats = [c for c in _GRAMMAR_CATEGORIES
                if any(c["range"][0] <= lid <= c["range"][1] for lid in available_set)]
        if cats:
            cat_labels = [f'{c["icon"]} {c["name"]}' for c in cats]
            def_cat_idx = next(
                (i for i, c in enumerate(cats)
                 if c["range"][0] <= default_gid <= c["range"][1]),
                0,
            )
            sel_cat_lbl = st.selectbox(
                "📚 Category", cat_labels, index=def_cat_idx,
                key=f"gram_cat_{lang_pair}",
            )
            sel_cat = cats[cat_labels.index(sel_cat_lbl)]
            filtered = [lid for lid in lessons
                        if sel_cat["range"][0] <= lid <= sel_cat["range"][1]]
        else:
            filtered = lessons
    else:
        BLOCK = 15
        blocks = []
        for i in range(0, len(lessons), BLOCK):
            blk = lessons[i:i + BLOCK]
            first = lesson_names.get(blk[0], f"Lesson {blk[0]}")
            short = (first[:22] + "…") if len(first) > 22 else first
            blocks.append({"label": f"Unit {i // BLOCK + 1}: {short}", "lids": blk})
        def_blk = next((i for i, b in enumerate(blocks) if default_gid in b["lids"]), 0)
        sel_blk_lbl = st.selectbox(
            "📚 Unit", [b["label"] for b in blocks], index=def_blk,
            key=f"gram_unit_{lang_pair}",
        )
        sel_blk  = blocks[[b["label"] for b in blocks].index(sel_blk_lbl)]
        filtered = sel_blk["lids"]

    if default_gid not in filtered:
        default_gid = filtered[0] if filtered else default_gid

    if not filtered:
        st.warning("No lessons available in this category.")
        return

    _key = f"{lang_pair}_{cfg.get('lang_suffix', 'g')}"
    clicked = _render_wave_plotly(
        lessons=filtered,
        lesson_names=lesson_names,
        lesson_counts=counts_by_lid,
        default_lid=default_gid,
        resume_step=resume_step,
        key_suffix=_key,
    )
    if clicked is not None:
        _start_grammar_lesson(clicked, cfg, native, target, user_id, lang_pair)
    else:
        _render_lesson_dropdown_fallback(
            lessons=filtered,
            lesson_names=lesson_names,
            default_lid=default_gid,
            resume_step=resume_step,
            cfg=cfg,
            native=native,
            target=target,
            user_id=user_id,
            lang_pair=lang_pair,
        )

def _render_vocab_nav(
    df, cfg, db_path, lang_pair, native, target, user_id,
    lessons, default_idx, resume_step, resume_msg, progress, counts_by_lid,
):
    """
    Compact vocab picker: two dropdowns (Category → Topic) + horizontal wave
    lesson path. Clicking a lesson card navigates via query params.
    """
    from engine.vocab_loader import get_vocab_nav_data

    nav_data       = get_vocab_nav_data(str(db_path))
    available_gids = set(lessons)

    # ── Resume banner ─────────────────────────────────────────────────────────
    if resume_msg:
        st.info(resume_msg)

    # ── Dropdown 1: Category ──────────────────────────────────────────────────
    cat_names = [f'{c["icon"]} {c["name"]}' for c in _VOCAB_CATEGORIES]
    cat_ids   = [c["id"] for c in _VOCAB_CATEGORIES]

    prev_cat_id  = st.session_state.get("vocab_nav_cat", cat_ids[0])
    prev_cat_idx = cat_ids.index(prev_cat_id) if prev_cat_id in cat_ids else 0

    sel_cat_name = st.selectbox("Category", cat_names, index=prev_cat_idx,
                                key="vnav_sel_category")
    sel_cat_idx = cat_names.index(sel_cat_name)
    sel_cat     = _VOCAB_CATEGORIES[sel_cat_idx]
    sel_cat_id  = sel_cat["id"]
    st.session_state["vocab_nav_cat"] = sel_cat_id

    # ── Dropdown 2: Topic ─────────────────────────────────────────────────────
    sheets_in_cat = [s for s in sel_cat["sheets"] if s in nav_data]
    if not sheets_in_cat:
        st.warning("No topics available for the selected language pair.")
        return

    topic_labels = []
    for s in sheets_in_cat:
        icon, label = _TOPIC_META.get(s, ("\U0001f4d6", s))
        n = len(nav_data.get(s, []))
        topic_labels.append(f"{icon} {label}  ({n} lessons)")

    prev_sheet = st.session_state.get("vocab_nav_topic", sheets_in_cat[0])
    if prev_sheet not in sheets_in_cat:
        prev_sheet = sheets_in_cat[0]
    prev_topic_idx = sheets_in_cat.index(prev_sheet)

    sel_topic_label = st.selectbox(
        "Topic", topic_labels, index=prev_topic_idx,
        key=f"vnav_sel_topic_{sel_cat_id}",
    )
    sel_topic_idx = topic_labels.index(sel_topic_label)
    sel_sheet     = sheets_in_cat[sel_topic_idx]
    st.session_state["vocab_nav_topic"] = sel_sheet

    # ── Wave component ────────────────────────────────────────────────────────
    lessons_in = nav_data.get(sel_sheet, [])
    radio_opts = [l for l in lessons_in if l["gid"] in available_gids]

    if not radio_opts:
        st.warning("No lessons available for this topic and language pair.")
        return

    default_gid = lessons[default_idx]
    if default_gid not in {l["gid"] for l in radio_opts}:
        default_gid = radio_opts[0]["gid"]

    lesson_names_dict = {l["gid"]: l["name"] for l in radio_opts}
    _vocab_lessons = [l["gid"] for l in radio_opts]

    _vkey = f"{lang_pair}_v"
    _vclicked = _render_wave_plotly(
        lessons=_vocab_lessons,
        lesson_names=lesson_names_dict,
        lesson_counts=counts_by_lid,
        default_lid=default_gid,
        resume_step=resume_step,
        key_suffix=_vkey,
        show_lesson_image=False,
    )
    if _vclicked is not None:
        _start_grammar_lesson(_vclicked, cfg, native, target, user_id, lang_pair)
    else:
        _render_lesson_dropdown_fallback(
            lessons=_vocab_lessons,
            lesson_names=lesson_names_dict,
            default_lid=default_gid,
            resume_step=resume_step,
            cfg=cfg,
            native=native,
            target=target,
            user_id=user_id,
            lang_pair=lang_pair,
        )
