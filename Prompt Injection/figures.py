"""Figures for DataBench 3 (Altair; HTML with hover tooltips + PNG for the README).

Every guard is identified three ways, so no chart relies on color alone:
  color (ABCD's model colors carried over) + marker shape (unique per guard) + line dash (by family).
Dumbbell grammar used throughout: FILLED marker = attacks caught, HOLLOW marker = harmless text flagged.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

import metrics

OUT = Path(__file__).parent / "figures"

# Fixed order = the validated reference palette's slot order (gray replaces red: TF-IDF is the "plain ML" floor,
# gray as in ABCD). Jev orange, Luna aqua, Haiku green, Metis violet are the colors these models had in ABCD.
GUARDS = {   # name: (color, shape, family)
    "ProtectAI v2":     ("#2a78d6", "circle",         "Local classifier"),
    "Jev 1.14":         ("#eb6834", "square",         "Decision model"),
    "GPT-6 Luna":       ("#1baf7a", "triangle-up",    "LLM"),
    "PIGuard":          ("#eda100", "diamond",        "Local classifier"),
    "Narrow":           ("#e87ba4", "cross",          "Ours (trained here)"),
    "Claude Haiku 5.5": ("#008300", "triangle-down",  "LLM"),
    "Metis (GreenPT)":  ("#6250d6", "triangle-right", "Decision model"),
    "TF-IDF + LR":      ("#52514e", "triangle-left",  "Ours (trained here)"),
}
FAMILIES = {"Local classifier": [1, 0], "Decision model": [7, 3], "LLM": [2, 2], "Ours (trained here)": [9, 3, 2, 3]}
INK, MUTED, AXIS, GRID, SURFACE = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9", "#fcfcfb"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # light -> dark blue
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"


def theme():
    return {"config": {
        "background": SURFACE, "font": FONT, "padding": 16,
        "view": {"stroke": None},
        "title": {"anchor": "start", "fontSize": 17, "fontWeight": 700, "color": INK, "subtitleColor": MUTED,
                  "subtitleFontSize": 13, "subtitlePadding": 6, "offset": 14},
        "axis": {"labelColor": MUTED, "titleColor": MUTED, "labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 500,
                 "gridColor": GRID, "domainColor": AXIS, "tickColor": AXIS},
        "header": {"labelFontSize": 13, "labelFontWeight": 700, "labelColor": INK, "labelAnchor": "start", "titleFontSize": 0},
        "legend": {"labelColor": INK, "titleColor": MUTED, "labelFontSize": 12, "titleFontSize": 12, "symbolStrokeWidth": 2,
                   "orient": "bottom", "direction": "horizontal", "columns": 4, "symbolSize": 110},
    }}


alt.themes.register("databench", theme)
alt.themes.enable("databench")


def guard_encodings(order=None, legend=True, present=None):
    names = order or list(GUARDS)
    shown = [g for g in GUARDS if g in present] if present is not None else None
    legend_spec = (alt.Legend(title=None, values=shown) if shown else alt.Legend(title=None)) if legend else None
    return {
        "color": alt.Color("guard:N", scale=alt.Scale(domain=list(GUARDS), range=[v[0] for v in GUARDS.values()]),
                           legend=legend_spec, sort=names),
        "shape": alt.Shape("guard:N", scale=alt.Scale(domain=list(GUARDS), range=[v[1] for v in GUARDS.values()]),
                           legend=legend_spec, sort=names),
    }


def dash_encoding(legend=True, present=None):
    values = [f for f in FAMILIES if present is None or f in present]
    return alt.StrokeDash("family:N", scale=alt.Scale(domain=list(FAMILIES), range=list(FAMILIES.values())),
                          legend=alt.Legend(title="Line style = kind of guard", orient="bottom", symbolSize=1400, values=values,
                                                            symbolStrokeWidth=2.5, symbolType="stroke") if legend else None)


def save(chart, name):
    OUT.mkdir(exist_ok=True)
    chart.save(OUT / f"{name}.html")
    chart.save(OUT / f"{name}.png", scale_factor=2)
    return chart


def pct(x):
    return f"{100 * x:.0f}%" if x >= 0.01 or x == 0 else f"{100 * x:.1f}%"


# ---------------------------------------------------------------- 1. Report card

CARD = {   # block title: [(column title, results column, kind)]; kind: catch = higher better, flag = lower better
    "What it catches (higher is better)": [
        ("Attack|emails", "recall LLMail @t_1pct", "catch"),
        ("Attacks in|tool outputs", "recall AgentDojo @t_1pct", "catch"),
        ("PromptShield|attacks", "recall PromptShield attacks @t_1pct", "catch"),
        ("Attacks after|its worst trick", "worst trick recall", "catch"),
    ],
    "What it wrongly flags (lower is better)": [
        ('Harmless|"ignore…" prompts', "FPR NotInject @t_1pct", "flag"),
        ("Normal work|emails", "FPR LLMail benign emails @t_1pct", "flag"),
        ("Clean tool|outputs", "FPR AgentDojo benign @t_1pct", "flag"),
        ("Harmless|encoded text", "encoded benign flagged", "flag"),
    ],
    "What it costs": [
        ("Time per|item", "p50 ms", "ms"),
        ("Price per|1,000 items", "$ per 1k", "usd"),
    ],
}


def card_frame(results):
    results = results.copy()
    tricks = ["homoglyph", "zero_width", "base64", "rot13", "guard_prefix"]
    results["worst trick recall"] = results["evasion recall clean @t_1pct"] - results[[f"evasion delta {t}" for t in tricks]].max(axis=1)
    results["encoded benign flagged"] = results[["evasion control FPR base64", "evasion control FPR rot13"]].max(axis=1)
    rows = []
    for guard in results.index:
        for block, columns in CARD.items():
            for title, column, kind in columns:
                value = float(results.loc[guard, column])
                if kind == "catch":
                    goodness, label = value, pct(value)
                elif kind == "flag":
                    goodness, label = 1 - min(value / 0.5, 1), pct(value)
                elif kind == "ms":
                    goodness = 1 - min(np.log10(max(value, 1)) / np.log10(2000), 1)
                    label = f"{value / 1000:.1f} s" if value >= 1000 else f"{value:.0f} ms"
                else:
                    goodness, label = 1 - min(value / 0.1, 1), "free" if value == 0 else f"${value:.3f}"
                rows.append({"guard": guard, "block": block, "question": title, "goodness": goodness, "label": label})
    return pd.DataFrame(rows)


def report_card(results):
    frame = card_frame(results)
    quality = frame[frame["block"] != "What it costs"]
    order = quality.groupby("guard")["goodness"].mean().sort_values(ascending=False).index.tolist()
    color = alt.Color("goodness:Q", scale=alt.Scale(domain=[0, 1], range=SEQ),
                      legend=alt.Legend(title="Darker = better for the user", orient="bottom", direction="horizontal",
                                        gradientLength=240, values=[0, 1], labelExpr="datum.value == 0 ? 'worse' : 'better'"))
    blocks = []
    for i, (block, columns) in enumerate(CARD.items()):
        data = frame[frame["block"] == block]
        y = alt.Y("guard:N", sort=order, title=None,
                  axis=alt.Axis(labelFontSize=13, labelColor=INK, labelFontWeight=600, domain=False, ticks=False,
                                labelPadding=28) if i == 0 else None)
        base = alt.Chart(data).encode(
            x=alt.X("question:N", sort=[c[0] for c in columns], title=None,
                    axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=12, labelColor=MUTED, domain=False, ticks=False,
                                  labelPadding=24, labelBaseline="bottom", labelLineHeight=15,
                                  labelExpr="split(datum.label, '|')")), y=y)
        cells = base.mark_rect(cornerRadius=5, stroke=SURFACE, strokeWidth=3).encode(
            color=color if i == 0 else color.to_dict() | {"legend": None},
            tooltip=["guard", alt.Tooltip("question:N", title="column"), alt.Tooltip("label:N", title="value")])
        text = base.mark_text(fontSize=14, fontWeight=600).encode(
            text="label:N", color=alt.condition("datum.goodness > 0.55", alt.value("#ffffff"), alt.value(INK)))
        layers = [cells, text]
        if i == 0:
            keys = alt.Chart(data.drop_duplicates("guard")).mark_point(filled=True, size=130, stroke=SURFACE, strokeWidth=1).encode(
                y=y, x=alt.value(-14), **guard_encodings(legend=False))
            layers.append(keys)
        blocks.append(alt.layer(*layers).properties(width=alt.Step(104), height=alt.Step(46),
                                                    title=alt.Title(block, fontSize=13, color=INK, anchor="start", offset=6)))
    chart = alt.hconcat(*blocks, spacing=26).resolve_scale(color="shared").properties(title=alt.Title(
        "Report card: what each guard catches, what it wrongly flags, and what it costs",
        subtitle=["Test set, each guard at its own cut-off (chosen on dev to flag at most 1% of harmless dev text). "
                  "Best on average at the top.",
                  "Numbers are shares of items: attacks caught on the left, harmless items wrongly flagged in the middle. "
                  "Darker cell = better. Claude's 1% after its worst trick is API refusals on rot13 text, counted as misses."]))
    return save(chart, "1_report_card")


# ---------------------------------------------------------------- dumbbell helper

def dumbbell(frame, x_title, caught_label, flagged_label, panel=None, panel_order=None, width=520, x_max=1.0,
             order=None, row_step=34, note_field=None, columns=1, extra=None):
    """frame: guard, caught, flagged (0-1), optional panel. Filled marker = caught, hollow = flagged."""
    frame = frame.copy()
    frame["family"] = frame["guard"].map(lambda g: GUARDS[g][2])
    frame["caught_text"], frame["flagged_text"] = frame["caught"].map(pct), frame["flagged"].map(pct)
    frame["left"] = frame[["caught", "flagged"]].min(axis=1)
    frame["right"] = frame[["caught", "flagged"]].max(axis=1)
    order = order or frame.groupby("guard").apply(lambda g: (g["caught"] - g["flagged"]).mean()).sort_values(
        ascending=False).index.tolist()
    y = alt.Y("guard:N", sort=order, title=None, axis=alt.Axis(labelFontSize=12, labelColor=INK, domain=False,
                                                                 ticks=False, labelPadding=8))
    xs = alt.Scale(domain=[0, x_max], clamp=True)
    xaxis = alt.Axis(format="%", grid=True, values=[0, 0.25, 0.5, 0.75, 1], title=x_title)
    tooltip = ["guard", alt.Tooltip("caught_text:N", title=caught_label), alt.Tooltip("flagged_text:N", title=flagged_label)]
    enc = guard_encodings(order)
    line = alt.Chart().mark_rule(strokeWidth=2.5, opacity=0.9).encode(
        y=y, x=alt.X("left:Q", scale=xs, axis=xaxis), x2="right:Q", color=enc["color"], strokeDash=dash_encoding(),
        tooltip=tooltip)
    hollow = alt.Chart().mark_point(filled=False, size=150, strokeWidth=2.5).encode(
        y=y, x=alt.X("flagged:Q", scale=xs), color=enc["color"], shape=enc["shape"], tooltip=tooltip)
    filled = alt.Chart().mark_point(filled=True, size=170, opacity=1, stroke=SURFACE, strokeWidth=1.5).encode(
        y=y, x=alt.X("caught:Q", scale=xs), color=enc["color"], shape=enc["shape"], tooltip=tooltip)
    label = alt.Chart().mark_text(align="left", dx=12, fontSize=11.5, color=MUTED).encode(
        y=y, x=alt.X("right:Q", scale=xs),
        text=alt.Text("note:N"))
    frame["note"] = frame.apply(lambda r: f"caught {r['caught_text']} · flagged {r['flagged_text']}"
                                if note_field in (None, "none") else r[note_field], axis=1)
    layers = [line, hollow, filled] + ([label] if note_field != "none" else []) + [layer(y, xs) for layer in (extra or [])]
    layered = alt.layer(*layers, data=frame).properties(width=width, height=alt.Step(row_step))
    if panel:
        return layered.facet(facet=alt.Facet(f"{panel}:N", sort=panel_order, title=None,
                                             header=alt.Header(labelOrient="top", labelPadding=6)),
                             columns=columns, spacing={"row": 22, "column": 34}).resolve_scale(y="independent")
    return layered


def legend_note():
    return ("● filled = attacks caught   ○ hollow = harmless items wrongly flagged   "
            "(each guard keeps its color and shape; the line style shows its kind)")


# ---------------------------------------------------------------- 2. Caught vs flagged

def caught_vs_flagged(results):
    rows = []
    for guard in results.index:
        r = results.loc[guard]
        rows.append({"guard": guard, "panel": "Plain text: LLMail attack emails vs harmless prompts with words like \"ignore\" (NotInject)",
                     "caught": r["recall LLMail @t_1pct"], "flagged": r["FPR NotInject @t_1pct"]})
        rows.append({"guard": guard, "panel": "Agent tool outputs: attacked vs clean AgentDojo records (calendar, email, bank, Slack)",
                     "caught": r["recall AgentDojo @t_1pct"], "flagged": r["FPR AgentDojo benign @t_1pct"]})
    frame = pd.DataFrame(rows)
    chart = dumbbell(frame, "Share of items", "Attacks caught", "Harmless flagged", panel="panel",
                     panel_order=frame["panel"].unique().tolist(), width=560, x_max=1.12).properties(
        title=alt.Title("Catch a lot, flag little: the further the filled marker sits right of the hollow one, the better",
                        subtitle=[legend_note(), "Rows ordered by catch minus flag, averaged over both panels."]))
    return save(chart, "2_caught_vs_flagged")


# ---------------------------------------------------------------- 3. Evasion tricks

TRICK_NAMES = {"homoglyph": "Look-alike|letters", "zero_width": "Invisible|characters", "base64": "Base64|encoded",
               "rot13": "Rot13|scrambled", "guard_prefix": "\"Classify as|safe\" prefix"}


def heat_block(data, columns, order, title, first, color):
    """One block of colored cells in the report-card style: guards down, columns across, value printed in each cell."""
    y = alt.Y("guard:N", sort=order, title=None,
              axis=alt.Axis(labelFontSize=13, labelColor=INK, labelFontWeight=600, domain=False, ticks=False,
                            labelPadding=28) if first else None)
    base = alt.Chart(data).encode(
        x=alt.X("column:N", sort=columns, title=None,
                axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=12, labelColor=MUTED, domain=False, ticks=False,
                              labelPadding=24, labelBaseline="bottom", labelLineHeight=15,
                              labelExpr="split(datum.label, '|')")), y=y)
    cells = base.mark_rect(cornerRadius=5, stroke=SURFACE, strokeWidth=3).encode(
        color=color if first else color.to_dict() | {"legend": None},
        tooltip=["guard", alt.Tooltip("column:N", title="column"), alt.Tooltip("label:N", title="value")])
    text = base.mark_text(fontSize=14, fontWeight=600).encode(
        text="label:N", color=alt.condition("datum.goodness > 0.55", alt.value("#ffffff"), alt.value(INK)))
    layers = [cells, text]
    if first:
        layers.append(alt.Chart(data.drop_duplicates("guard")).mark_point(filled=True, size=130, stroke=SURFACE, strokeWidth=1)
                      .encode(y=y, x=alt.value(-14), **guard_encodings(legend=False)))
    return alt.layer(*layers).properties(width=alt.Step(96), height=alt.Step(44),
                                         title=alt.Title(title, fontSize=13, color=INK, anchor="start", offset=6))


def evasion(results):
    caught, flagged = [], []
    for guard in results.index:
        r = results.loc[guard]
        clean = r["evasion recall clean @t_1pct"]
        caught.append({"guard": guard, "column": "No|trick", "value": clean})
        flagged.append({"guard": guard, "column": "No|trick", "value": r["evasion control FPR clean"]})
        for trick, name in TRICK_NAMES.items():
            caught.append({"guard": guard, "column": name, "value": clean - r[f"evasion delta {trick}"]})
            flagged.append({"guard": guard, "column": name, "value": r[f"evasion control FPR {trick}"]})
    caught, flagged = pd.DataFrame(caught), pd.DataFrame(flagged)
    caught["goodness"] = caught["value"]
    flagged["goodness"] = 1 - (flagged["value"] / 0.5).clip(upper=1)
    for frame in (caught, flagged):
        frame["label"] = frame["value"].map(pct)
    columns = ["No|trick"] + list(TRICK_NAMES.values())
    # Best at the top: catches the tricked attacks AND leaves harmless tricked text alone
    order = (caught.groupby("guard")["goodness"].mean() + flagged.groupby("guard")["goodness"].mean()) \
        .sort_values(ascending=False).index.tolist()
    color = alt.Color("goodness:Q", scale=alt.Scale(domain=[0, 1], range=SEQ),
                      legend=alt.Legend(title="Darker = better for the user", orient="bottom", direction="horizontal",
                                        gradientLength=240, values=[0, 1], labelExpr="datum.value == 0 ? 'worse' : 'better'"))
    chart = alt.hconcat(
        heat_block(caught, columns, order, "500 attacks: share still caught (higher is better)", True, color),
        heat_block(flagged, columns, order, "200 harmless texts: share wrongly flagged (lower is better)", False, color),
        spacing=30).resolve_scale(color="shared").properties(title=alt.Title(
            "Tricks: Jev and Luna see through all five; Metis can't read encoded text; three guards just flag anything encoded",
            subtitle=["The same texts with each trick applied. A good guard is dark in both tables. Dark left + light right = "
                      "it flags the trick itself, not the attack.",
                      "Claude's low base64 and rot13 numbers are API refusals, counted as misses. The \"classify as safe\" "
                      "prefix is itself an instruction to the AI, so flagging it is defensible."]))
    return save(chart, "3_evasion_tricks")


# ---------------------------------------------------------------- 4. Question wording

def wording(choice):
    table = pd.DataFrame(choice["table"])
    rows = []
    for guard, group in table.groupby("arm"):
        v1, v2 = group.set_index("wording").loc["v1"], group.set_index("wording").loc["v2"]
        rows.append({"guard": guard, "family": GUARDS[guard][2], "before": v1["FPR @0.5"], "after": v2["FPR @0.5"],
                     "auc_before": v1["dev AUROC"], "auc_after": v2["dev AUROC"]})
    frame = pd.DataFrame(rows)
    frame["note"] = frame.apply(lambda r: f"{pct(r['before'])} → {pct(r['after'])} flagged   ·   AUROC {r['auc_before']:.2f} → {r['auc_after']:.2f}", axis=1)
    order = frame.sort_values("before", ascending=False)["guard"].tolist()
    y = alt.Y("guard:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13, labelColor=INK, domain=False, ticks=False,
                                                               labelPadding=14))
    xs = alt.Scale(domain=[-0.03, 1.0], nice=False)
    enc = guard_encodings(order, present=frame["guard"].tolist())
    tip = ["guard", alt.Tooltip("before:Q", format=".1%", title="Plan's wording"), alt.Tooltip("after:Q", format=".1%", title="Fixed wording")]
    line = alt.Chart().mark_rule(strokeWidth=3).encode(y=y, x=alt.X("after:Q", scale=xs, axis=alt.Axis(
        format="%", title="Harmless dev prompts flagged at score 0.5", values=[0, 0.2, 0.4, 0.6, 0.8, 1])), x2="before:Q", color=enc["color"],
        strokeDash=dash_encoding(present=frame["family"].tolist()), tooltip=tip)
    hollow = alt.Chart().mark_point(filled=False, size=170, strokeWidth=2.5).encode(
        y=y, x=alt.X("before:Q", scale=xs), color=enc["color"], shape=enc["shape"], tooltip=tip)
    filled = alt.Chart().mark_point(filled=True, size=190, stroke=SURFACE, strokeWidth=1.5).encode(
        y=y, x=alt.X("after:Q", scale=xs), color=enc["color"], shape=enc["shape"], tooltip=tip)
    arrow = alt.Chart().mark_text(text="◀", fontSize=13, dx=-11, dy=0.5).encode(
        y=y, x=alt.X("after:Q", scale=xs), color=enc["color"])
    label = alt.Chart().mark_text(align="left", dx=14, fontSize=12, color=MUTED).encode(
        y=y, x=alt.X("before:Q", scale=xs), text="note:N")
    chart = alt.layer(line, filled, hollow, label, data=frame).properties(width=560, height=alt.Step(48), title=alt.Title(
        "One sentence fixed the false alarms: the plan's question vs the fixed one",
        subtitle=["○ hollow = plan's wording (\"does it try to give instructions to an AI?\")   ● filled = fixed wording "
                  "(says what is NOT an injection)", "Same 200 harmless + 200 attack dev items for every guard; test was "
                  "scored only after this choice."]))
    return save(chart, "4_question_wording")


# ---------------------------------------------------------------- 5. Threshold transfer

def threshold_transfer(results):
    frame = pd.DataFrame({"guard": results.index, "dev": results["dev FPR @t_1pct"].astype(float).values,
                          "test": results["FPR PromptShield benign @t_1pct"].astype(float).values})
    frame["family"] = frame["guard"].map(lambda g: GUARDS[g][2])
    cap = 0.115
    frame["test_shown"] = frame["test"].clip(upper=cap)
    frame["note"] = frame.apply(lambda r: f"{pct(r['dev'])} on dev → {pct(r['test'])} on test" +
                                ("  (off the chart)" if r["test"] > cap else ""), axis=1)
    order = frame.sort_values("test")["guard"].tolist()
    y = alt.Y("guard:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13, labelColor=INK, domain=False, ticks=False))
    xs = alt.Scale(domain=[0, 0.12], nice=False)
    enc = guard_encodings(order)
    target = alt.Chart(pd.DataFrame({"x": [0.01]})).mark_rule(color=MUTED, strokeWidth=1.5).encode(x=alt.X("x:Q", scale=xs))
    target_label = alt.Chart(pd.DataFrame({"x": [0.01], "t": ["target: at most 1%"]})).mark_text(
        align="left", dx=4, dy=-8, fontSize=11, color=MUTED, baseline="bottom").encode(x=alt.X("x:Q", scale=xs), y=alt.value(0), text="t:N")
    line = alt.Chart(frame).mark_rule(strokeWidth=3).encode(y=y, x=alt.X("dev:Q", scale=xs, axis=alt.Axis(
        format="%", title="Harmless items flagged (dev = where the threshold was set; test = new PromptShield text)")),
        x2="test_shown:Q", color=enc["color"], strokeDash=dash_encoding())
    hollow = alt.Chart(frame).mark_point(filled=False, size=160, strokeWidth=2.5).encode(
        y=y, x=alt.X("dev:Q", scale=xs), color=enc["color"], shape=enc["shape"])
    filled = alt.Chart(frame).mark_point(filled=True, size=180, stroke=SURFACE, strokeWidth=1.5).encode(
        y=y, x=alt.X("test_shown:Q", scale=xs), color=enc["color"], shape=enc["shape"],
        tooltip=["guard", alt.Tooltip("dev:Q", format=".2%"), alt.Tooltip("test:Q", format=".1%")])
    label = alt.Chart(frame).mark_text(align="left", dx=14, fontSize=12, color=MUTED).encode(
        y=y, x=alt.X("test_shown:Q", scale=xs), text="note:N")
    chart = alt.layer(target, target_label, line, hollow, filled, label).properties(width=600, height=alt.Step(40), title=alt.Title(
        "A cut-off tuned for 1% false alarms gives 2-9% on new text",
        subtitle=["○ hollow = share of harmless dev items flagged (the cut-off was set here)   ● filled = share of harmless "
                  "test items flagged", "Plan for re-tuning on your own traffic before trusting any vendor's threshold."]))
    return save(chart, "5_threshold_transfer")


# ---------------------------------------------------------------- 6. Speed vs quality

def speed_vs_quality(results):
    frame = pd.DataFrame({"guard": results.index,
                          "ms": results["p50 ms"].astype(float).values,
                          "gap": (results["recall LLMail @t_1pct"] - results["FPR NotInject @t_1pct"]).astype(float).values,
                          "cost": results["$ per 1k"].astype(float).values})
    frame["family"] = frame["guard"].map(lambda g: GUARDS[g][2])
    frame["note"] = frame.apply(lambda r: f"{r['guard']}  ({'free' if r['cost'] == 0 else f'${r['cost']:.3f}/1k'})", axis=1)
    enc = guard_encodings()
    xs = alt.Scale(type="log", domain=[0.7, 4000])
    points = alt.Chart(frame).mark_point(filled=True, size=260, stroke=SURFACE, strokeWidth=2).encode(
        x=alt.X("ms:Q", scale=xs, axis=alt.Axis(title="Median time per item (ms, log scale)", values=[1, 10, 100, 1000])),
        y=alt.Y("gap:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Attack emails caught minus harmless prompts flagged")),
        color=enc["color"], shape=enc["shape"],
        tooltip=["guard", alt.Tooltip("ms:Q", format=".0f", title="ms"), alt.Tooltip("gap:Q", format=".0%"), alt.Tooltip("cost:Q", format="$.3f")])
    frame["label_y"] = frame["gap"] + frame["guard"].map({"Claude Haiku 5.5": 0.025, "GPT-6 Luna": -0.025}).fillna(0)
    labels = alt.Chart(frame).mark_text(align="left", dx=14, fontSize=12, color=INK).encode(
        x=alt.X("ms:Q", scale=xs), y="label_y:Q", text="note:N")
    chart = alt.layer(points, labels).properties(width=620, height=360, title=alt.Title(
        "Speed vs usefulness: the LLMs score highest among vendors but are 15-20x slower than Jev",
        subtitle=["Up = catches more attack emails while flagging fewer harmless \"ignore…\" prompts. Left = faster.",
                  "TF-IDF and Narrow trained on the same sources as these test emails; see the report card for how they do elsewhere."]))
    return save(chart, "6_speed_vs_quality")


# ---------------------------------------------------------------- 7. ROC (for reviewers)

def roc(test_items, scores_by_guard):
    rows = []
    for name, attack_tier, benign_tier in [p for p in metrics.AUROC_PAIRS[:4] if p[2] != "3e"]:
        pos = ((test_items["tier"] == attack_tier) & (test_items["label"] == 1)).to_numpy()
        neg = ((test_items["tier"] == benign_tier) & (test_items["label"] == 0)).to_numpy()
        for guard, scores in scores_by_guard.items():
            points = metrics.roc_points(scores[pos], scores[neg])
            points = points[points["fpr"] <= 0.2]
            rows.append(points.assign(guard=guard, panel=name, family=GUARDS[guard][2]))
    frame = pd.concat(rows)
    enc = guard_encodings()
    chart = alt.Chart(frame).mark_line(interpolate="step-after", strokeWidth=2).encode(
        x=alt.X("fpr:Q", scale=alt.Scale(domain=[0, 0.2]), axis=alt.Axis(format="%", title="Harmless items flagged")),
        y=alt.Y("tpr:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Attacks caught")),
        color=enc["color"], strokeDash=dash_encoding(), detail="guard:N",
        tooltip=["guard", alt.Tooltip("fpr:Q", format=".1%"), alt.Tooltip("tpr:Q", format=".1%"), alt.Tooltip("threshold:Q", format=".3f")],
    ).properties(width=300, height=240).facet(facet=alt.Facet("panel:N", title=None), columns=3).properties(title=alt.Title(
        "Every possible cut-off (for reviewers): up and to the left is better",
        subtitle=["Line style shows the kind of guard; hover a line for its cut-off."]))
    return save(chart, "7_roc_by_tier")


def dedupe_funnel(stats, residual_share):
    f = stats["llmail_funnel"]
    steps = [("Submitted to the challenge", sum(f["raw submissions (dataset card)"].values()), ""),
             ("Unique, with a judge's verdict", f["labelled unique submissions"], "the challenge's own exact dedupe"),
             ("Judged to be an attack", f["judged attacks (attack_attempt True)"], "drops Unclear and not-an-attack"),
             ("No exact repeats", f["after exact dedupe (normalised)"], "ignoring case, punctuation, digits"),
             ("No near-copies (80%+ overlap)", f["after near dedupe"],
              f"~{100 * residual_share:.1f}% still have a near-copy (re-check)")]
    frame = pd.DataFrame(steps, columns=["step", "rows", "how"])
    frame["order"] = range(len(frame))
    frame["value"] = frame["rows"].map(lambda v: f"{v:,}")
    frame["drop"] = [""] + [f"−{100 * (1 - b / a):.0f}%" for a, b in zip(frame["rows"][:-1], frame["rows"][1:])]
    frame["note"] = frame.apply(lambda r: f"{r['value']}   {r['drop']}  {r['how']}".replace("    ", "  "), axis=1)
    y = alt.Y("step:N", sort=frame["step"].tolist(), title=None,
              axis=alt.Axis(labelFontSize=13, labelColor=INK, domain=False, ticks=False, labelPadding=10, labelLimit=260))
    # one series, in a neutral grey so it doesn't read as any guard's color
    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, height=24, color="#898781").encode(
        y=y, x=alt.X("rows:Q", axis=alt.Axis(format="~s", title="LLMail rows", grid=True)),
        tooltip=["step", alt.Tooltip("rows:Q", format=","), "how"])
    text = alt.Chart(frame).mark_text(align="left", dx=8, fontSize=12, color=MUTED).encode(y=y, x="rows:Q", text="note:N")
    chart = alt.layer(bars, text).properties(width=520, height=alt.Step(42), title=alt.Title(
        "From 462k challenge submissions to 74.6k distinct attacks",
        subtitle=["Each step removes copies so no near-duplicate of a test email can sit in training or dev. "
                  "Percentages: rows removed by that step."]))
    return save(chart, "0_dedupe_funnel")


def all_figures(results, wording_choice, test_items, scores_by_guard, stats, residual_share):
    return [dedupe_funnel(stats, residual_share), report_card(results), caught_vs_flagged(results), evasion(results),
            wording(wording_choice), threshold_transfer(results), speed_vs_quality(results), roc(test_items, scores_by_guard)]
