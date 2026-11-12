import re
import textwrap
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import transforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from mplsoccer import VerticalPitch

import io
from matplotlib.text import Text
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link
from pypdf.generic import ArrayObject, NameObject, NullObject, FloatObject, Fit

project_root = Path(__file__).resolve().parent.parent      # main/, wherever the notebook runs from
plots_dir = project_root / "plots"

match_keys = ["match_id", "sea_team_id", "match_week", "side"]     # per-match tables
team_keys = ["sea_team_id", "side"]                                 # per-team tables

def side_view(df, side, opp):
    return pd.DataFrame({
        "match_id": df["match_id"],
        "competition_id": df["competition_id"],
        "competition_name": df["competition_name"],
        "season_id": df["season_id"],
        "season": df["season"],
        "match_week": df["match_week"],
        "match_date": df["match_date"],
        "kick_off": df["kick_off"],
        "competition_stage_id": df["competition_stage_id"],
        "competition_stage": df["competition_stage"],
        "team_id": df[f"{side}_team_id"],
        "team": df[f"{side}_team"],
        "manager_id": df[f"{side}_manager_id"],
        "manager_name": df[f"{side}_manager_name"],
        "opponent_id": df[f"{opp}_team_id"],
        "opponent": df[f"{opp}_team"],
        "stadium_country_id": df["stadium_country_id"],
        "stadium_country_name": df["stadium_country_name"],
        "goals_for": df[f"{side}_score"],
        "goals_against": df[f"{opp}_score"],
    })

nomial_end = pd.Series({1: 45, 2: 90, 3: 105, 4: 120})

def build_match_time_anchors(events_dfs):
    """Timing anchors per match, all in elapsed minutes (continuous playing time).

    Returns
    -------
    period_end    : DataFrame (match_id × period 1–4), elapsed minute each period ended; NaN if not played
    half_time     : Series (match_id), elapsed minute the first half ended
    match_end     : Series (match_id), elapsed minute of the final whistle
    period_offset : Series ((match_id, period)), minutes to add to minute + second/60
    """
    events_all = pd.concat(
        [df[["match_id", "period", "minute", "second", "type"]] for df in events_dfs],
        ignore_index=True,
    )
    events_all["clock_min"] = events_all["minute"] + events_all["second"] / 60

    # End of each played period on the match clock (period 5 = shootout, ignored)
    half_end = events_all[(events_all["type"] == "Half End") & (events_all["period"] <= 4)]
    period_end_clock = (half_end.groupby(["match_id", "period"])["clock_min"].max()
                        .unstack("period")
                        .reindex(columns=[1, 2, 3, 4]))

    # Stoppage per period -> cumulative stoppage before each period
    stoppage = (period_end_clock - nomial_end).fillna(0)
    offsets = stoppage.cumsum(axis=1).shift(1, axis=1).fillna(0)

    # Convert to elapsed time, then take what the rest of the code needs
    period_end = period_end_clock + offsets
    half_time = period_end[1].rename("half_time")
    match_end = period_end.max(axis=1).rename("match_end")
    period_offset = offsets.stack().rename("period_offset")
    return period_end, half_time, match_end, period_offset

def add_elapsed_time(df, period_offset):
    df = df.copy()
    raw_time = df['minute'] + df['second'] / 60
    keys = pd.MultiIndex.from_frame(df[['match_id', 'period']])
    df['elapsed_min'] = raw_time + period_offset.reindex(keys).fillna(0).to_numpy()
    return df

def label_state(diff):
    return np.select(
        [diff > 0, diff == 0, diff == -1, diff == -2],
        ["leading", "level", "trailing by 1", "trailing by 2"],
        "trailing by 3+",
    )

def build_state_intervals(goals, match_end):
    rows = []
    for mid, end in match_end.items():
        start, score, concede = 0.0, 0, 0
        for _, r in goals[goals["match_id"] == mid].sort_values("elapsed_min").iterrows():
            rows.append((mid, start, r["elapsed_min"], score, concede))
            score, concede, start = r["score_for"], r["score_against"], r["elapsed_min"]
        rows.append((mid, start, end, score, concede))
    si = pd.DataFrame(rows, columns=["match_id", "start", "end", "score_for", "score_against"])
    d = si["score_for"] - si["score_against"]
    si["game_state"] = label_state(si["score_for"] - si["score_against"])
    si["duration"] = si["end"] - si["start"]
    return si

def display_minute(period, minute):
    if period == 1 and minute >= 45:
        return f"45+{minute - 44}'"
    if period == 2 and minute >= 90:
        return f"90+{minute - 89}'"
    return f"{minute + 1}'"

def build_formation_timeline(match_id, team_id, events_df, all_position_slots):
    team_events = events_df[
        (events_df["match_id"] == match_id) & (events_df["team_id"] == team_id)
    ].sort_values("index")
 
    timeline_rows = []
    current_state = {slot: None for slot in all_position_slots}
    current_formation = None                                                    # NEW
 
    for _, row in team_events.iterrows():
        if row["type"] in ["Starting XI", "Tactical Shift"]:
            lineup = row["tactics"]["lineup"]
            current_formation = row["tactics"]["formation"]                     # NEW
            current_state = {slot: None for slot in all_position_slots}
            sent_off_ids = team_events[
                (team_events["type"] == "Red Card") & (team_events["elapsed_min"] <= row["elapsed_min"])
            ]["player_id"].tolist()
 
            for entry in lineup:
                if entry["player"]["id"] not in sent_off_ids:
                    current_state[entry["position"]["name"]] = entry["player"]["id"]
 
        elif row["type"] == "Substitution":
            outgoing_id = row["player_id"]
            incoming_id = row["substitution_replacement_id"]
            for slot, occupant in current_state.items():
                if occupant == outgoing_id:
                    current_state[slot] = incoming_id
                    break
 
        elif row["type"] in ["Player Off", "Red Card"]:  # both remove a player from the pitch
            for slot, occupant in current_state.items():
                if occupant == row["player_id"]:
                    current_state[slot] = None
                    break
 
        elif row["type"] == "Player On":
            current_state[row["position"]] = row["player_id"]
 
        snapshot = current_state.copy()
        snapshot["timestamp"] = row["elapsed_min"]
        snapshot["event_type"] = row["type"]
        snapshot["formation"] = current_formation                               # NEW
        timeline_rows.append(snapshot)
 
    return pd.DataFrame(timeline_rows)

def build_final_stints(formation_timeline, all_position_slots, match_end):
    """Returns (final_stints, formation_timeline_with_checkpoints)."""
    ft = formation_timeline.sort_values(["match_id", "team_id", "timestamp"]).reset_index(drop=True)
    ft["checkpoint_id"] = ft.groupby(["match_id", "team_id"]).cumcount() + 1
    ft["end_time"] = ft.groupby(["match_id", "team_id"])["timestamp"].shift(-1)
    ft["end_time"] = ft["end_time"].fillna(ft["match_id"].map(match_end))
    # groupby silently drops NaN keys, so check formation here, before melt/groupby
    assert ft["formation"].notna().all(), "checkpoints without a formation (team-match with no Starting XI?)"
    ft["formation"] = ft["formation"].astype(int)
 
    formation_long = ft.melt(
        id_vars=["match_id", "team_id", "checkpoint_id", "timestamp", "end_time", "event_type",
                 "formation"],
        value_vars=all_position_slots,
        var_name="position_name", value_name="player_id",
    ).dropna(subset=["player_id"])
    formation_long = formation_long.sort_values(["match_id", "team_id", "player_id", "checkpoint_id"])
    grp = ["match_id", "team_id", "player_id"]
    formation_long["prev_position"] = formation_long.groupby(grp)["position_name"].shift(1)
    formation_long["prev_checkpoint"] = formation_long.groupby(grp)["checkpoint_id"].shift(1)
    formation_long["prev_formation"] = formation_long.groupby(grp)["formation"].shift(1)
    formation_long["new_stint"] = (
        formation_long["prev_checkpoint"].isna()
        | (formation_long["position_name"] != formation_long["prev_position"])
        | (formation_long["checkpoint_id"] != formation_long["prev_checkpoint"] + 1)
        | (formation_long["formation"] != formation_long["prev_formation"])
    )
    formation_long["stint_id"] = formation_long.groupby(grp)["new_stint"].cumsum()
 
    final_stints = (formation_long
                    .groupby(["match_id", "team_id", "player_id", "position_name", "formation", "stint_id"])
                    .agg(from_min=("timestamp", "min"), to_min=("end_time", "max"))
                    .reset_index())
    final_stints["duration"] = final_stints["to_min"] - final_stints["from_min"]
    return final_stints, ft

def minutes_by_state(player_stints, state_intervals):
    x = player_stints.merge(state_intervals, on="match_id", validate="m:m", suffixes=("", "_gs"))
    x["overlap"] = (np.minimum(x["to_min"], x["end"]) - np.maximum(x["from_min"], x["start"])).clip(lower=0)
    out = x.pivot_table(index="match_id", columns="game_state", values="overlap", aggfunc="sum", fill_value=0)
    return out.reindex(columns=["leading", "level", "trailing"], fill_value=0).add_prefix("min_")

def table_before(group_stage, teams, kickoff, team_name):
    """Group table for `teams` from group-stage games that kicked off before `kickoff`.

    group_stage needs: home_team_id, away_team_id, home_score, away_score, kick_off_dt.
    """
    prior = group_stage[
        group_stage["home_team_id"].isin(teams)
        & group_stage["away_team_id"].isin(teams)
        & (group_stage["kick_off_dt"] < kickoff)
    ]
    rows = {t: {"P": 0, "Pts": 0, "GF": 0, "GA": 0} for t in teams}
    for hid, aid, hs, as_ in prior[["home_team_id", "away_team_id", "home_score", "away_score"]].itertuples(index=False):
        for t, gf, ga in [(hid, hs, as_), (aid, as_, hs)]:
            r = rows[t]
            r["P"] += 1; r["GF"] += gf; r["GA"] += ga
            r["Pts"] += 3 if gf > ga else 1 if gf == ga else 0
    tab = pd.DataFrame.from_dict(rows, orient="index")
    tab["GD"] = tab["GF"] - tab["GA"]
    tab["team"] = tab.index.map(team_name)
    return tab.sort_values(["Pts", "GD", "GF"], ascending=False)

def table_string(tab):
    return ", ".join(f"{r.team} {r.Pts} ({r.GD:+d}, P{r.P})" for r in tab.itertuples())

group_stage = 10
stage_order = {10: 0, 33: 1, 11: 2, 15: 3, 25: 4, 26: 5}
stage_short = {10: "GS", 33: "RO16", 11: "QF", 15: "SF", 25: "3P", 26: "F"}

def result_row(r, ko_info, team_name):
    """Result for one row of team_matches (one team's view of one match).

    r          : a row with match_id, team_id, vs_id, gf, ga
    ko_info    : {match_id: {"max_period": int, "pens": {team_id: shootout goals}}}
    team_name  : {team_id: display name}
    """
    info = ko_info.get(r["match_id"], {"max_period": 2, "pens": {}})
    gf, ga = r["gf"], r["ga"]
    res = "W" if gf > ga else "L" if gf < ga else "D"
    note = ""
    if info["max_period"] == 5:
        pf = info["pens"].get(r["team_id"], 0)
        pa = info["pens"].get(r["vs_id"], 0)
        note = f" ({'won' if pf > pa else 'lost'} {pf}–{pa} on pens)"
    elif info["max_period"] >= 3:
        note = " (aet)"
    label = f"{res} {gf}–{ga}{note} v {team_name[r['vs_id']]}"
    if info["max_period"] == 5:
        res = "W" if pf > pa else "L"   # who went through; still a draw in W–D–L
    return pd.Series({"res": res, "label": label, "drawn": gf == ga})

def placing(team_matches, team_id, season_id):
    """Final placing of a team at one tournament. team_matches must already have `res`."""
    tm = team_matches[(team_matches["team_id"] == team_id) & (team_matches["season_id"] == season_id)]
    last = tm.loc[tm["competition_stage_id"].map(stage_order).idxmax()]
    won = last["res"] == "W"
    return {
        26: "Champions" if won else "Runners-up",
        25: "Third Place" if won else "Fourth Place",
        11: "Quarter-finals",
        33: "Round of 16",
        10: "Group stage",
    }[last["competition_stage_id"]]

def opponent_context(team_matches, opp_id, sea_id, season_id):
    """Opponent's record and results at the same tournament, excluding the match against sea_id."""
    o = team_matches[
        (team_matches["team_id"] == opp_id)
        & (team_matches["season_id"] == season_id)
        & (team_matches["vs_id"] != sea_id)
    ].sort_values(["match_date", "kick_off"])
    w = ((o["res"] == "W") & ~o["drawn"]).sum()
    d = o["drawn"].sum()
    l = ((o["res"] == "L") & ~o["drawn"]).sum()
    run = "; ".join(
        f"{stage_short[stage]}: " + ", ".join(g["label"])
        for stage, g in o.groupby("competition_stage_id", sort=False)
    )
    return pd.Series({
        "opp_record_other": f"W{w} D{d} L{l}",
        "opp_avg_margin_other": round((o["gf"] - o["ga"]).mean(), 2),
        "opp_placing": placing(team_matches, opp_id, season_id),
        "opp_run_other": run,
    })

def add_labels(df, matches_sea, sea_ids):
    """Add sea_team_id and match_week (from match_id), and side (from team_id, if present). Safe to re-run."""
    out = df.copy()                                              # don't change the original table
    m = matches_sea.set_index("match_id")                        # one row per match, looked up by match_id
    out["sea_team_id"] = out["match_id"].map(m["team_id"])      # SEA team of that match
    out["match_week"] = out["match_id"].map(m["match_week"])    # matchday of that match
    if "team_id" in out.columns:
        out["side"] = np.where(out["team_id"].isin(sea_ids), "sea", "opp")
    return out

own_third_x, own_half_x, final_third_x = 40, 60, 80
quick_shot_secs = 15

def in_own_half(x):
    return x < own_half_x

def in_own_third(x):
    return x < own_third_x

def in_final_third(x):
    return x >= final_third_x

def is_long_pass(pass_length):
    return pass_length >= 35

def in_own_box(x, y):
    return (x <= 18) & (y >= 18) & (y <= 62)

def in_opposition_box(x, y):
    return (x >= 102) & (y >= 18) & (y <= 62)

goal_x, goal_y = 120, 40
box_x, box_y_min, box_y_max = 102, 18, 62
six_y_min, six_y_max = 30, 50          # width of the six-yard box, used as the "central" channel
big_chance_xg = 0.3

def distance_to_goal(x, y):
    dx = goal_x - x
    dy = goal_y - y
    return np.sqrt(dx**2 + dy**2)

def distance(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx**2 + dy**2)

def is_left_side(y):
    return y < 26.6

def is_right_side(y):
    return y > 53.3

def is_central(y):
    return (y >= 26.6) & (y <= 53.3)

surface = "#fcfcfb"
text = "#0b0b0b"
text_2 = "#52514e"
muted = "#898781"
grid = "#e1e0d9"
team_colors = {1107: "#2e5bba", 16769: "#da251d", 1106: "#d4a100"}
opp_color = muted

# Event symbols: a football for every goal; text tags mark penalties and own goals
event_tags = {"goal": "", "pen_goal": "P", "pen_miss": "P", "og": "OG"}
event_labels = {"goal": "Goal", "pen_goal": "Penalty scored (P)",
                "pen_miss": "Penalty missed (× P)", "og": "Own goal, for this team (OG)"}
ball_size = 10

metric_labels = {"xg": "Cumulative xG (incl. penalties)", "npxg": "Cumulative non-penalty xG"}

def xg_timeline(events, match_id, sea_ids):
    """Shots and own goals in one match, with cumulative xG and npxG for each side.

    kind: "goal", "pen_goal", "pen_miss", "og" (own goal, credited to the benefiting team), or "shot"
    """
    e = events[events["match_id"] == match_id]
    shots = e[e["type"] == "Shot"]
    og = e[e["type"] == "Own Goal For"]
    is_pen = shots["shot_type"] == "Penalty"
    is_goal = shots["shot_outcome"] == "Goal"

    tl = pd.concat([
        pd.DataFrame({
            "elapsed_min": shots["elapsed_min"],
            "team_id": shots["team_id"],
            "xg": shots["shot_statsbomb_xg"],
            "npxg": np.where(is_pen, 0.0, shots["shot_statsbomb_xg"]),
            "kind": np.select([is_pen & is_goal, is_pen, is_goal], ["pen_goal", "pen_miss", "goal"], "shot"),
        }),
        pd.DataFrame({
            "elapsed_min": og["elapsed_min"],
            "team_id": og["team_id"],
            "xg": 0.0,
            "npxg": 0.0,
            "kind": "og",
        }),
    ]).sort_values("elapsed_min").reset_index(drop=True)

    tl["is_goal"] = tl["kind"].isin(["goal", "pen_goal", "og"])
    tl["side"] = np.where(tl["team_id"].isin(sea_ids), "sea", "opp")
    tl["cum_xg"] = tl.groupby("side")["xg"].cumsum()
    tl["cum_npxg"] = tl.groupby("side")["npxg"].cumsum()
    return tl

def darker(color, amount=0.35):
    """Mix a colour towards black, e.g. for outlines that keep light colours visible."""
    r, g, b, _ = to_rgba(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount), 1.0)

def _style_axes(ax):
    ax.figure.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.grid(axis="y", color=grid, lw=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(grid)
    ax.tick_params(colors=text_2)


def _steps(tl, side, match_end, col):
    s = tl[tl["side"] == side]
    last = s[col].iloc[-1] if len(s) else 0.0
    x = np.concatenate([[0.0], s["elapsed_min"].to_numpy(), [match_end]])
    y = np.concatenate([[0.0], s[col].to_numpy(), [last]])
    return x, y


def _ball(ax, x, y, color, size=ball_size):
    """A small football: white disc, coloured rim, coloured centre panel."""
    ax.plot(x, y, "o", ms=size, mfc=surface, mec=color, mew=1.8, zorder=4, ls="")
    ax.plot(x, y, "p", ms=size * 0.45, mfc=color, mec=color, mew=0, zorder=5, ls="")


def _event_markers(ax, tl, side, col, color, match_end):
    """Football for goals (incl. penalties and own goals), a cross for missed penalties, plus text tags."""
    pts = tl[(tl["side"] == side) & (tl["kind"].isin(event_tags.keys()))]
    for _, r in pts.iterrows():
        x, y = r["elapsed_min"], r[col]
        if r["kind"] == "pen_miss":
            ax.plot(x, y, "X", ms=ball_size * 0.9, mfc=color, mec=surface, mew=1.5, zorder=5, ls="")
        else:
            _ball(ax, x, y, color)
        tag = event_tags[r["kind"]]
        if tag:
            near_end = x > match_end - 4          # keep the tag clear of the end-of-line label
            ax.annotate(tag, (x, y), xytext=(-6 if near_end else 6, 7), textcoords="offset points",
                        fontsize=7.5, color=text_2, ha="right" if near_end else "left",
                        va="center", zorder=6)


def symbol_key_handles(kinds=None):
    """Legend entries for the event symbols, in a neutral colour. `kinds` limits it to those present."""
    ball = (Line2D([], [], ls="", marker="o", ms=ball_size, mfc=surface, mec=text_2, mew=1.8),
            Line2D([], [], ls="", marker="p", ms=ball_size * 0.45, mfc=text_2, mec=text_2, mew=0))
    cross = Line2D([], [], ls="", marker="X", ms=ball_size * 0.9, mfc=text_2, mec=surface, mew=1.5)
    handles, labels = [], []
    for kind in event_tags:
        if kinds is not None and kind not in kinds:
            continue
        handles.append(cross if kind == "pen_miss" else ball)
        labels.append(event_labels[kind])
    return handles, labels


def set_match_clock_axis(ax, half_time, match_end, step=15):
    """Label the x-axis with the match clock: 0–45, then the second half restarting at 45.

    Points stay at their real elapsed time; only the tick labels change.
    Stoppage time in each half is shaded.
    """
    first = list(range(0, 46, step))
    second = list(range(45 + step, 91, step))
    ax.set_xticks(first + [half_time + (m - 45) for m in second])
    ax.set_xticklabels([str(m) for m in first + second])

    ax.axvspan(45, half_time, color=grid, alpha=0.6, lw=0, zorder=0)
    full_time = half_time + 45
    if match_end > full_time:
        ax.axvspan(full_time, match_end, color=grid, alpha=0.6, lw=0, zorder=0)
    ax.set_xlabel("Match minute (shaded = stoppage time)", color=text_2)


def plot_xg_timeline(tl, match_end, half_time, sea_id, opp_id, team_name,
                     ax=None, title=None, ymax=None, metric="xg", clock_axis=True, symbol_key=True):
    """Cumulative xG step lines for both sides, with goals, penalties and own goals marked.

    metric     : "xg" (penalties included) or "npxg"
    symbol_key : add the event-symbol key to this axes' legend (turn off in multi-panel figures)
    """
    col = f"cum_{metric}"
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.8))
    _style_axes(ax)

    handles, labels = [], []
    sides = [("opp", team_name[opp_id], opp_color), ("sea", team_name[sea_id], team_colors[sea_id])]
    for side, name, color in sides:
        x, y = _steps(tl, side, match_end, col)
        line, = ax.step(x, y, where="post", color=color, lw=2, solid_joinstyle="round", label=name)
        handles.append(line)
        labels.append(name)
        _event_markers(ax, tl, side, col, color, match_end)
        ax.annotate(f"{name} {y[-1]:.2f}", (match_end, y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=text)

    ax.axvline(half_time, color=muted, lw=1)
    ax.set_xlim(0, match_end)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    if clock_axis:
        set_match_clock_axis(ax, half_time, match_end)
    else:
        ax.set_xlabel("Minutes played (including stoppage time)", color=text_2)
    ax.set_ylabel(metric_labels[metric], color=text_2)

    if symbol_key:
        h, l = symbol_key_handles(set(tl["kind"]))
        handles += h
        labels += l
    # Legend sits above the plot area, so it never covers the lines or the half-time marker
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              frameon=False, fontsize=9, labelcolor=text, handlelength=1.5, columnspacing=1.2)
    if title:
        ax.set_title(title, loc="left", color=text, fontsize=11, pad=24 if not symbol_key else 46)
    return ax


def plot_team_timelines(sea_id, team_matches, timelines, team_name, match_end, half_time,
                        ymax=None, title=None, metric="xg"):
    """One row of xG timelines for a team's matches, on a shared y-axis, with one symbol key.

    Panel titles show the matchday and the score from the SEA team's side, e.g. "MD2: 0–2 Portugal".
    The figure title defaults to the team and tournament; pass `title` to replace it.
    """
    col = f"cum_{metric}"
    team_matches = team_matches.sort_values("match_week")
    if ymax is None:
        ymax = max(timelines[m][col].max() for m in team_matches["match_id"]) * 1.1

    fig, axes = plt.subplots(1, len(team_matches), figsize=(16, 5.4), sharey=True)
    for ax, (_, r) in zip(np.atleast_1d(axes), team_matches.iterrows()):
        mid, opp_id = r["match_id"], r["opponent_id"]
        tl = timelines[mid]
        goals = tl[tl["is_goal"]].groupby("side").size()
        score = f"{goals.get('sea', 0)}–{goals.get('opp', 0)}"
        plot_xg_timeline(tl, match_end[mid], half_time[mid], sea_id, opp_id, team_name,
                         ax=ax, title=f"MD{r['match_week']}: {score} {team_name[opp_id]}", ymax=ymax,
                         metric=metric, symbol_key=False)
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_ylabel("")

    kinds = set().union(*(timelines[m]["kind"] for m in team_matches["match_id"]))
    h, l = symbol_key_handles(kinds)
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=text, bbox_to_anchor=(0.5, -0.02))

    if title is None:
        season = f" {team_matches['season'].iloc[0]}" if "season" in team_matches else ""
        title = f"{team_name[sea_id]} at the Women's World Cup{season}"
    fig.suptitle(title, x=0.01, ha="left", color=text, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig

def split_at_minute(tl, minute, match_end, side="opp", metrics=("npxg", "xg"), per=15):
    """Shots, totals and per-`per`-minute rates for one side, before and after a given minute.

    The event at `minute` itself (e.g. the goal) counts as "before": it was created in that state.
    """
    s = tl[tl["side"] == side]
    before = s["elapsed_min"] <= minute
    mins_before, mins_after = minute, match_end - minute
    out = {
        "mins_before": mins_before,
        "mins_after": mins_after,
        "shots_before": int(before.sum()),
        "shots_after": int((~before).sum()),
    }
    for m in metrics:
        tot_b, tot_a = s.loc[before, m].sum(), s.loc[~before, m].sum()
        out[f"{m}_before"] = tot_b
        out[f"{m}_after"] = tot_a
        out[f"{m}_per{per}_before"] = tot_b / mins_before * per
        out[f"{m}_per{per}_after"] = tot_a / mins_after * per
    return out


def first_goal_rates(timelines, matches, match_end, team_name, metrics=("npxg", "xg"), per=15):
    """Opponent chances before and after the first goal conceded, one row per match.

    For each metric: totals before/after and rates per `per` minutes.
    """
    rows = []
    for _, r in matches.iterrows():
        mid = r["match_id"]
        tl = timelines[mid]
        conceded = tl[(tl["side"] == "opp") & tl["is_goal"]]
        if conceded.empty:
            continue
        first = conceded["elapsed_min"].min()
        row = {
            "match_id": mid,
            "team": team_name[r["team_id"]],
            "opponent": team_name[r["opponent_id"]],
            "first_conceded_elapsed": first,
        }
        row.update({f"opp_{k}": v for k, v in
                    split_at_minute(tl, first, match_end[mid], metrics=metrics, per=per).items()})
        rows.append(row)
    return pd.DataFrame(rows)

def shot_table(events, sea_ids):
    """One row per shot across all matches, from the shooting team's side (attacking towards x = 120)."""
    s = events[events["type"] == "Shot"].copy()
    is_pen = s["shot_type"] == "Penalty"
    is_goal = s["shot_outcome"] == "Goal"

    out = pd.DataFrame({
        "match_id": s["match_id"],
        "team_id": s["team_id"],
        "side": np.where(s["team_id"].isin(sea_ids), "sea", "opp"),
        "elapsed_min": s["elapsed_min"],
        "location_x": s["location_x"],
        "location_y": s["location_y"],
        "xg": s["shot_statsbomb_xg"],
        "npxg": np.where(is_pen, 0.0, s["shot_statsbomb_xg"]),
        "kind": np.select([is_pen & is_goal, is_pen, is_goal], ["pen_goal", "pen_miss", "goal"], "shot"),
        "outcome": s["shot_outcome"],
        "body_part": s["shot_body_part"],
        "play_pattern": s["play_pattern"],
        "possession": s["possession"],
        "possession_team_id": s["possession_team_id"],
        "index": s["index"],
        "player": s["player"],
        "minute": s["minute"],
        "period": s["period"],
    })
    out["is_goal"] = out["kind"].isin(["goal", "pen_goal"])
    out["distance"] = np.hypot(goal_x - out["location_x"], goal_y - out["location_y"])
    out["in_box"] = (out["location_x"] >= box_x) & out["location_y"].between(box_y_min, box_y_max)
    out["central_box"] = out["in_box"] & out["location_y"].between(six_y_min, six_y_max)
    return out.reset_index(drop=True)

def shot_profile(shots, by=("match_id", "side")):
    """Where shots came from and how good they were, per group. Penalties are left out."""
    s = shots[~shots["kind"].isin(["pen_goal", "pen_miss"])].copy()
    s["big_chance"] = s["npxg"] >= big_chance_xg
    g = s.groupby(list(by))
    out = g.agg(
        shots=("npxg", "size"),
        npxg=("npxg", "sum"),
        goals=("is_goal", "sum"),
        in_box=("in_box", "sum"),
        central_box=("central_box", "sum"),
        big_chances=("big_chance", "sum"),
        avg_distance=("distance", "mean"),
    ).reset_index()
    out["npxg_per_shot"] = out["npxg"] / out["shots"]
    out["in_box_pct"] = out["in_box"] / out["shots"]
    out["central_box_pct"] = out["central_box"] / out["shots"]
    return out

def score_from_timeline(tl):
    """'SEA–OPP' score string from an xg_timeline table (own goals included)."""
    goals = tl[tl["is_goal"]].groupby("side").size()
    return f"{goals.get('sea', 0)}–{goals.get('opp', 0)}"

shot_size_scale = 1200      # marker area (points²) per 1.0 xG: area is proportional to xG
shot_size_floor = 6         # only so the very smallest chances stay visible

def _shot_sizes(xg):
    return np.maximum(np.asarray(xg, dtype=float) * shot_size_scale, shot_size_floor)

def draw_pitch(ax):
    pitch = VerticalPitch(pitch_type="statsbomb", half=True, pitch_color=surface,
                          line_color=grid, linewidth=1, pad_bottom=-15)
    pitch.draw(ax=ax)
    return pitch

def own_goal_counts(timelines):
    """{(match_id, side): number of own goals credited to that side}, from xg_timeline tables."""
    counts = {}
    for mid, tl in timelines.items():
        og = tl[tl["kind"] == "og"].groupby("side").size()
        for side in ("sea", "opp"):
            counts[(mid, side)] = int(og.get(side, 0))
    return counts

def penalty_line(shots):
    """'Penalties: scored/taken (rate)' text for a set of shots; '0/0' when none were taken."""
    pens = shots[shots["kind"].isin(["pen_goal", "pen_miss"])]
    taken, scored = len(pens), int((pens["kind"] == "pen_goal").sum())
    return f"Penalties: {scored}/{taken} scored" + (f" ({scored / taken:.0%})" if taken else "")

def shot_summary_text(shots, own_goals=0):
    """Three summary lines: non-penalty shots, own goals, penalties."""
    play = shots[~shots["kind"].isin(["pen_goal", "pen_miss"])]
    pens = shots[shots["kind"].isin(["pen_goal", "pen_miss"])]
    n_goals = int(play["is_goal"].sum())
    taken, scored = len(pens), int((pens["kind"] == "pen_goal").sum())
    pen = f"{scored}/{taken} converted" + (f" ({scored / taken:.0%})" if taken else "")
    return (f"Non-penalty: {len(play)} shots · {play['npxg'].sum():.2f} npxG · "
            f"{n_goals} goal{'s' if n_goals != 1 else ''}\n"
            f"Own goals: {own_goals}\n"
            f"Penalties: {pen}")

def plot_shot_map(ax, shots, color, title=None, own_goals=0):
    """Non-penalty shots on a half pitch: size = xG, solid = goal, light = no goal.

    Penalties are left off the pitch; penalties and own goals are summarised underneath.
    """
    pitch = draw_pitch(ax)
    play = shots[~shots["kind"].isin(["pen_goal", "pen_miss"])]
    miss = play[~play["is_goal"]]
    goal = play[play["is_goal"]]
    # No goal: light translucent fill with a solid darker outline, so overlaps build up into
    # darker areas but every shot keeps a visible edge (matters for light colours like yellow)
    pitch.scatter(miss["location_x"], miss["location_y"], s=_shot_sizes(miss["xg"]), ax=ax,
                  facecolors=to_rgba(color, 0.28), edgecolors=darker(color), linewidths=0.8, zorder=3)
    # Goals: solid fill, darker outline, drawn on top
    pitch.scatter(goal["location_x"], goal["location_y"], s=_shot_sizes(goal["xg"]), ax=ax,
                  facecolors=color, edgecolors=darker(color), linewidths=1.2, zorder=4)

    # Summary sits below the pitch, so it never covers a shot
    ax.text(0.0, -0.03, shot_summary_text(shots, own_goals), transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=text_2, linespacing=1.5)
    if title:
        ax.set_title(title, loc="left", color=text, fontsize=11)
    return ax


def shot_size_key(ax_or_fig, values=(0.05, 0.2, 0.5), loc="lower center", bbox=(0.5, 0.0)):
    """Legend for marker size (xG) and fill (goal / no goal), drawn in the same style as the map."""
    miss_style = dict(facecolors=to_rgba(text_2, 0.28), edgecolors=darker(text_2), linewidths=0.8)
    goal_style = dict(facecolors=text_2, edgecolors=darker(text_2), linewidths=1.2)
    handles = [plt.scatter([], [], s=_shot_sizes(v), **miss_style) for v in values]
    labels = [f"xG {v}" for v in values]
    handles += [plt.scatter([], [], s=_shot_sizes(0.2), **goal_style),
                plt.scatter([], [], s=_shot_sizes(0.2), **miss_style)]
    labels += ["Goal", "No goal"]
    ax_or_fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox, ncol=len(labels), frameon=False,
                     fontsize=9, labelcolor=text, scatterpoints=1, handletextpad=0.6, columnspacing=1.4)


def plot_team_shot_maps(sea_id, team_matches, shots, team_name, scores, own_goals=None, title=None):
    """Shots conceded in each match (grey), plus all shots created across the matches (team colour).

    own_goals : {(match_id, side): count} from own_goal_counts(); omit to show 0
    """
    own_goals = own_goals or {}
    team_matches = team_matches.sort_values("match_week")
    n = len(team_matches)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 4.3))
    fig.patch.set_facecolor(surface)
    for ax, (_, r) in zip(axes[:n], team_matches.iterrows()):
        mid, opp_id = r["match_id"], r["opponent_id"]
        conceded = shots[(shots["match_id"] == mid) & (shots["side"] == "opp")]
        plot_shot_map(ax, conceded, opp_color, own_goals=own_goals.get((mid, "opp"), 0),
                      title=f"MD{r['match_week']}: {scores[mid]} {team_name[opp_id]}\nShots conceded")
    mids = team_matches["match_id"]
    created = shots[shots["match_id"].isin(mids) & (shots["side"] == "sea")]
    plot_shot_map(axes[n], created, team_colors[sea_id],
                  own_goals=sum(own_goals.get((m, "sea"), 0) for m in mids),
                  title=f"{team_name[sea_id]}: all {n} matches\nShots created")
    if title is None:
        season = f" {team_matches['season'].iloc[0]}" if "season" in team_matches else ""
        title = f"{team_name[sea_id]} at the Women's World Cup{season}: where the shots came from"
    fig.suptitle(title, x=0.01, y=0.995, ha="left", color=text, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))
    # Size key in the top-right corner, level with the title
    shot_size_key(fig, loc="upper right", bbox=(0.995, 1.0))
    return fig

non_play = ["Starting XI", "Half Start", "Half End", "Substitution", "Player Off", "Player On",
            "Tactical Shift", "Injury Stoppage", "Bad Behaviour", "Referee Ball-Drop"]
set_piece_patterns = ["From Corner", "From Free Kick"]
direct_sp_secs, delayed_sp_secs = 10, 30
origin_order = ["Transition", "Counter", "Direct set piece", "Delayed set piece",
           "Final-third throw-in", "Open play"]


def add_shot_origin(shots, events):
    """Label each shot by how the chance came about (penalties get 'Penalty').

    Transition           : shot by the team NOT credited with the possession (won it and shot at once)
    Counter              : possession that StatsBomb marks as a counter-attack
    Direct set piece     : corner / free-kick possession, shot within 10 s of the restart
    Delayed set piece    : corner / free-kick possession, shot within 30 s
    Final-third throw-in : throw-in taken in the attacking third, shot within 30 s
    Open play            : everything else

    Safe to re-run: columns added by a previous call are dropped first.
    """
    added = ["restart_time", "restart_loc", "secs_since_restart", "origin"]
    shots = shots.drop(columns=added, errors="ignore")

    play = events[~events["type"].isin(non_play)].sort_values(["match_id", "index"])
    starts = (play.groupby(["match_id", "possession"])
                  .agg(restart_time=("elapsed_min", "first"), restart_loc=("location", "first")))
    s = shots.merge(starts, left_on=["match_id", "possession"], right_index=True, how="left")
    s["secs_since_restart"] = (s["elapsed_min"] - s["restart_time"]) * 60
    restart_x = s["restart_loc"].str[0]

    set_piece = s["play_pattern"].isin(set_piece_patterns)
    throw_in = s["play_pattern"].eq("From Throw In") & (in_final_third(restart_x))
    secs = s["secs_since_restart"]
    s["origin"] = np.select(
        [s["kind"].isin(["pen_goal", "pen_miss"]),
         s["team_id"] != s["possession_team_id"],
         s["play_pattern"].eq("From Counter"),
         set_piece & (secs <= direct_sp_secs),
         set_piece & (secs <= delayed_sp_secs),
         throw_in & (secs <= delayed_sp_secs)],
        ["Penalty", "Transition", "Counter", "Direct set piece", "Delayed set piece", "Final-third throw-in"],
        default="Open play",
    )
    return s.drop(columns=["restart_loc"])


def origin_summary(shots, by=("match_id", "side")):
    """Shots, goals and npxG by origin (penalties left out), with each origin's share of the npxG."""
    s = shots[shots["origin"] != "Penalty"]
    out = (s.groupby(list(by) + ["origin"])
             .agg(shots=("npxg", "size"), goals=("is_goal", "sum"), npxg=("npxg", "sum"))
             .reset_index())
    out["npxg_per_shot"] = out["npxg"] / out["shots"]
    out["npxg_share"] = out["npxg"] / out.groupby(list(by))["npxg"].transform("sum")
    return out

def origin_table(origins, side="opp", value="npxg"):
    """One row per match, one column per origin, plus a total. side: 'opp' (conceded) or 'sea' (created)."""
    w = (origins[origins["side"] == side]
         .pivot_table(index=["team", "match_week", "opponent"], columns="origin",
                      values=value, aggfunc="sum", fill_value=0))
    w = w[[o for o in origin_order if o in w.columns]]
    w["total"] = w.sum(axis=1)
    return w

def possession_table(events):
    """One row per possession: team, period, start time, and the next possession's id / team / period."""
    play = events[~events["type"].isin(non_play)]
    p = (play.groupby(["match_id", "possession"])
             .agg(team_id=("possession_team_id", "first"), period=("period", "first"),
                  start_min=("elapsed_min", "min"))
             .reset_index()
             .sort_values(["match_id", "start_min"]))
    g = p.groupby("match_id")
    p["next_possession"] = g["possession"].shift(-1)
    p["next_team_id"] = g["team_id"].shift(-1)
    p["next_period"] = g["period"].shift(-1)
    return p

on_ball_types = ["Pass", "Carry", "Dribble", "Miscontrol", "Dispossessed", "Shot", "Clearance",
                 "Goal Keeper", "Foul Committed", "Duel", "Error", "Offside", "Shield",
                 "Ball Recovery", "Interception", "50/50", "Block"]
open_play_pass_types = ["Recovery", "Interception"]       # plus missing pass_type

def _loss_cause(r):
    """How the ball was lost, from the last on-ball action."""
    t = r["type"]
    if t == "Pass":
        if pd.isna(r["pass_outcome"]):
            return "Other"
        restart = pd.notna(r["pass_type"]) and r["pass_type"] not in open_play_pass_types
        return "Misplaced pass (restart)" if restart else "Misplaced pass (open play)"
    if t == "Miscontrol":
        return "Miscontrol"
    if t == "Dribble":
        return "Failed dribble"
    if t == "Dispossessed":
        return "Dispossessed"
    if t == "Duel" and r.get("duel_type") == "Aerial Lost":
        return "Aerial duel lost"
    if t == "Foul Committed":
        return "Foul committed"
    if t == "Clearance":
        return "Clearance"
    if t == "Shot":
        return "Shot"
    return "Other"


def ball_losses(events, sea_ids):
    """One row per SEA possession that was lost: followed directly by an opponent possession in the
    same half, or containing an opponent shot (StatsBomb sometimes keeps the possession credited to
    the team that lost the ball when the opponent shoots straight away).

    The row describes the SEA team's last on-ball action before the ball was lost. Possessions that
    ended with a SEA shot are flagged (ended_in_shot); transition cases are flagged (transition_loss).
    """
    p = possession_table(events)
    normal = (p["team_id"].isin(sea_ids) & p["next_team_id"].notna()
              & ~p["next_team_id"].isin(sea_ids) & (p["next_period"] == p["period"]))
    # Opponent shots inside a SEA possession
    opp_in_own = events[(events["type"] == "Shot") & ~events["team_id"].isin(sea_ids)
                        & events["possession_team_id"].isin(sea_ids)]
    first_opp_shot = (opp_in_own.groupby(["match_id", "possession"])["index"].min()
                      .rename("opp_shot_index").reset_index())
    p = p.merge(first_opp_shot, on=["match_id", "possession"], how="left")
    lost = p[normal | p["opp_shot_index"].notna()]

    on_ball = events[events["type"].isin(on_ball_types) & events["team_id"].isin(sea_ids)
                     & events["location"].notna()]
    on_ball = on_ball.merge(lost[["match_id", "possession", "opp_shot_index"]], on=["match_id", "possession"])
    # In transition cases, only actions before the opponent's shot can be the losing action
    on_ball = on_ball[on_ball["opp_shot_index"].isna() | (on_ball["index"] < on_ball["opp_shot_index"])]
    last = on_ball.sort_values(["match_id", "index"]).groupby(["match_id", "possession"]).tail(1)

    out = last[["match_id", "possession", "team_id", "elapsed_min", "period", "type",
                "pass_outcome", "pass_type", "duel_type", "location_x", "location_y"]].copy()
    if "game_state" in last:
        out["game_state"] = last["game_state"]
    out["cause"] = out.apply(_loss_cause, axis=1)
    out["ended_in_shot"] = out["cause"] == "Shot"
    out["zone"] = np.select([out["location_x"] < own_third_x, out["location_x"] < final_third_x],
                            ["Own third", "Middle third"], "Final third")
    out["own_half"] = out["location_x"] < own_half_x
    out = out.merge(lost[["match_id", "possession", "next_possession", "opp_shot_index"]],
                    on=["match_id", "possession"])
    out["transition_loss"] = out["opp_shot_index"].notna()
    return out.drop(columns=["opp_shot_index"]).reset_index(drop=True)

def add_next_possession_shots(losses, shots, quick_secs=quick_shot_secs):
    """Add what the opponent did straight after each loss: shots in their next possession, plus any
    shot they took inside the lost possession itself (transition cases). Safe to re-run."""
    added = ["opp_shots", "opp_npxg", "opp_xg", "opp_goals", "first_shot_min",
             "led_to_shot", "secs_to_shot", "quick_shot"]
    losses = losses.drop(columns=added, errors="ignore")
    opp = shots[shots["side"] == "opp"]

    def per_possession(s, key):
        return (s.sort_values("elapsed_min").groupby(["match_id", "possession"])
                 .agg(opp_shots=("xg", "size"), opp_npxg=("npxg", "sum"), opp_xg=("xg", "sum"),
                      opp_goals=("is_goal", "sum"), first_shot_min=("elapsed_min", "first"))
                 .reset_index().rename(columns={"possession": key}))

    # Shots in the opponent's own next possession, and shots inside the lost (SEA) possession
    nxt = per_possession(opp[opp["possession_team_id"] == opp["team_id"]], "next_possession")
    same = per_possession(opp[opp["possession_team_id"] != opp["team_id"]], "possession")

    out = losses.merge(nxt, on=["match_id", "next_possession"], how="left")
    out = out.merge(same, on=["match_id", "possession"], how="left", suffixes=("", "_same"))
    for c in ["opp_shots", "opp_npxg", "opp_xg", "opp_goals"]:
        out[c] = out[c].fillna(0) + out[f"{c}_same"].fillna(0)
    out["first_shot_min"] = out[["first_shot_min_same", "first_shot_min"]].min(axis=1)
    out = out.drop(columns=[c for c in out.columns if c.endswith("_same")])
    out[["opp_shots", "opp_goals"]] = out[["opp_shots", "opp_goals"]].astype(int)
    out["led_to_shot"] = out["opp_shots"] > 0
    out["secs_to_shot"] = (out["first_shot_min"] - out["elapsed_min"]) * 60
    out["quick_shot"] = out["led_to_shot"] & (out["secs_to_shot"] <= quick_secs)
    return out

def loss_summary(losses, by=("match_id",)):
    """Ball losses per group: totals, where, and what they cost (possessions ending in a SEA shot left out)."""
    l = losses[~losses["ended_in_shot"]].copy()
    l["own_half_to_shot"] = l["own_half"] & l["led_to_shot"]
    l["own_half_quick"] = l["own_half"] & l["quick_shot"]
    l["own_half_npxg"] = np.where(l["own_half"], l["opp_npxg"], 0.0)
    l["own_half_goals"] = np.where(l["own_half"], l["opp_goals"], 0)
    out = l.groupby(list(by)).agg(
        losses=("location_x", "size"),
        own_third=("zone", lambda z: (z == "Own third").sum()),
        middle_third=("zone", lambda z: (z == "Middle third").sum()),
        final_third=("zone", lambda z: (z == "Final third").sum()),
        own_half=("own_half", "sum"),
        own_half_to_shot=("own_half_to_shot", "sum"),
        own_half_quick_shot=("own_half_quick", "sum"),
        own_half_opp_npxg=("own_half_npxg", "sum"),
        own_half_opp_goals=("own_half_goals", "sum"),
        led_to_shot=("led_to_shot", "sum"),
        opp_npxg_after_loss=("opp_npxg", "sum"),
        opp_goals_after_loss=("opp_goals", "sum"),
    ).reset_index()
    out["own_half_pct"] = out["own_half"] / out["losses"]
    out["own_half_to_shot_pct"] = out["own_half_to_shot"] / out["own_half"]
    return out

loss_locations = {
    # option: (x column, y column, zone column, own-half column, title wording)
    "played": ("location_x", "location_y", "zone", "own_half", "where they lost the ball"),
    "won": ("won_x", "won_y", "won_zone", "won_own_half", "where opponents won the ball"),
}

def plot_loss_map(ax, losses, color, title=None, location="played"):
    """Where a team lost the ball (its own goal at the bottom, attacking upwards).

    location : "played" = where the team's last action was (e.g. where a pass was played from)
               "won"    = where the opponent won the ball (needs add_won_location)
    Solid = the opponent's next possession produced a shot; light = it didn't.
    """
    x_col, y_col, zone_col, half_col, _ = loss_locations[location]
    pitch = VerticalPitch(pitch_type="statsbomb", pitch_color=surface, line_color=grid, linewidth=1)
    pitch.draw(ax=ax)
    l = losses[~losses["ended_in_shot"]]
    for x0 in (40, 80):
        pitch.lines(x0, 0, x0, 80, ax=ax, color=grid, lw=1, linestyle="-", zorder=1)
    quiet = l[~l["led_to_shot"]]
    costly = l[l["led_to_shot"]]
    pitch.scatter(quiet[x_col], quiet[y_col], s=22, ax=ax, facecolors=to_rgba(color, 0.25),
                  edgecolors=to_rgba(darker(color), 0.6), linewidths=0.5, zorder=3)
    pitch.scatter(costly[x_col], costly[y_col], s=40, ax=ax, facecolors=color,
                  edgecolors=darker(color), linewidths=1, zorder=4)
    for zone, (x0, x1) in {"Own third": (0, 40), "Middle third": (40, 80), "Final third": (80, 120)}.items():
        n = int((l[zone_col] == zone).sum())
        pitch.annotate(f"{n}\nlosses", ((x0 + x1) / 2, 82), ax=ax, ha="left", va="center",
                       fontsize=8.5, color=text_2, annotation_clip=False)

    oh = l[l[half_col]]
    n_goals = int(oh["opp_goals"].sum())
    ax.text(0.0, -0.02,
            f"Losses: {len(l)} · own half: {len(oh)} ({len(oh) / max(len(l), 1):.0%})\n"
            f"Own-half losses → opp shot: {int(oh['led_to_shot'].sum())} "
            f"({int(oh['quick_shot'].sum())} within 15 s)\n"
            f"…worth {oh['opp_npxg'].sum():.2f} npxG · {n_goals} goal{'s' if n_goals != 1 else ''}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=text_2, linespacing=1.5)
    if title:
        ax.set_title(title, loc="left", color=text, fontsize=11)
    return ax

def loss_key(fig, color=None, loc="upper right", bbox=(0.995, 1.0)):
    """Legend for the loss map."""
    c = color or text_2
    handles = [plt.scatter([], [], s=40, facecolors=c, edgecolors=darker(c), linewidths=1),
               plt.scatter([], [], s=22, facecolors=to_rgba(c, 0.25), edgecolors=to_rgba(darker(c), 0.6),
                           linewidths=0.5)]
    labels = ["Opponent's next possession had a shot", "No shot"]
    fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox, ncol=2, frameon=False,
               fontsize=9, labelcolor=text, scatterpoints=1)

def plot_team_loss_maps(sea_id, team_matches, losses, team_name, scores, title=None, location="played"):
    """One loss map per match for a SEA team. location: "played" or "won" (see plot_loss_map)."""
    team_matches = team_matches.sort_values("match_week")
    n = len(team_matches)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 7.4))
    fig.patch.set_facecolor(surface)
    for ax, (_, r) in zip(np.atleast_1d(axes), team_matches.iterrows()):
        mid, opp_id = r["match_id"], r["opponent_id"]
        plot_loss_map(ax, losses[losses["match_id"] == mid], team_colors[sea_id],
                      title=f"MD{r['match_week']}: {scores[mid]} {team_name[opp_id]}", location=location)
    if title is None:
        season = f" {team_matches['season'].iloc[0]}" if "season" in team_matches else ""
        title = f"{team_name[sea_id]} at the Women's World Cup{season}: {loss_locations[location][4]}"
    fig.suptitle(title, x=0.01, y=0.995, ha="left", color=text, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    loss_key(fig, team_colors[sea_id])
    return fig

def add_won_location(losses, events, pitch_length=120, pitch_width=80):
    """Where the opponent won the ball: the first action of their next possession by the team in
    possession, flipped to the SEA team's side of the pitch (own goal at x = 0). Safe to re-run."""
    added = ["won_x", "won_y", "won_zone", "won_own_half"]
    losses = losses.drop(columns=added, errors="ignore")
    own = events[(events["team_id"] == events["possession_team_id"]) & events["location_x"].notna()
                 & ~events["type"].isin(non_play)]
    first = (own.sort_values(["match_id", "index"])
                .groupby(["match_id", "possession"])[["location_x", "location_y"]].first()
                .reset_index()
                .rename(columns={"possession": "next_possession", "location_x": "won_x", "location_y": "won_y"}))
    first["won_x"] = pitch_length - first["won_x"]
    first["won_y"] = pitch_width - first["won_y"]
    out = losses.merge(first, on=["match_id", "next_possession"], how="left")
    out["won_zone"] = np.select([out["won_x"] < own_third_x, out["won_x"] < final_third_x],
                                ["Own third", "Middle third"], "Final third")
    out["won_own_half"] = out["won_x"] < own_half_x
    return out

restart_patterns = ["From Goal Kick", "From Throw In", "From Free Kick", "From Corner", "From Kick Off"]

def ball_regains(events, sea_ids):
    """One row per SEA possession that starts straight after an opponent possession in the same half.

    Describes the SEA team's first on-ball action (where, when, what) and how the possession began:
      Active  : won in open play (interception, recovery, tackle, block, loose ball...)
      Keeper  : goalkeeper gained the ball (save, claim) - StatsBomb 'From Keeper'
      Restart : given the ball from a dead ball (goal kick, throw-in, free kick, corner, kick-off)
    """
    p = possession_table(events)
    g = p.groupby("match_id")
    p["prev_team_id"] = g["team_id"].shift(1)
    p["prev_period"] = g["period"].shift(1)
    p["prev_possession"] = g["possession"].shift(1)
    won = p[p["team_id"].isin(sea_ids) & p["prev_team_id"].notna()
            & ~p["prev_team_id"].isin(sea_ids) & (p["prev_period"] == p["period"])]

    on_ball = events[events["type"].isin(on_ball_types) & events["team_id"].isin(sea_ids)
                     & events["location_x"].notna()]
    first = (on_ball.merge(won[["match_id", "possession", "prev_possession"]], on=["match_id", "possession"])
                    .sort_values(["match_id", "index"])
                    .groupby(["match_id", "possession"]).head(1))

    out = first[["match_id", "possession", "prev_possession", "team_id", "elapsed_min", "period",
                 "type", "play_pattern", "location_x", "location_y"]].copy()
    if "game_state" in first:
        out["game_state"] = first["game_state"]
    out["how"] = np.select([out["play_pattern"].isin(restart_patterns), out["play_pattern"].eq("From Keeper")],
                           ["Restart", "Keeper"], "Active")
    out["zone"] = np.select([out["location_x"] < own_third_x, out["location_x"] < final_third_x],
                            ["Own third", "Middle third"], "Final third")
    out["own_half"] = out["location_x"] < own_half_x
    return out.reset_index(drop=True)

kept_secs, quick_reloss_secs = 10, 5

def add_regain_outcome(regains, events, shots):
    """What the SEA team did with each regained possession. Safe to re-run."""
    added = ["kept_secs", "passes_completed", "max_x", "reached_opp_half", "reached_final_third",
             "progressed", "shots", "npxg", "kept_10s", "lost_quickly"]
    regains = regains.drop(columns=added, errors="ignore")
    e = events[~events["type"].isin(non_play)].copy()
    e["end_min"] = e["elapsed_min"] + e["duration"].fillna(0) / 60
    keys = regains[["match_id", "possession", "team_id"]]
    e = e.merge(keys, on=["match_id", "possession"], suffixes=("", "_sea"))
    own = e[e["team_id"] == e["team_id_sea"]]

    per = e.groupby(["match_id", "possession"]).agg(end_min=("end_min", "max")).reset_index()
    own_per = own.groupby(["match_id", "possession"]).agg(
        passes_completed=("pass_outcome", lambda s: ((own.loc[s.index, "type"] == "Pass") & s.isna()).sum()),
        max_x=("location_x", "max"),
    ).reset_index()
    sea_shots = (shots[shots["side"] == "sea"].groupby(["match_id", "possession"])
                 .agg(shots=("xg", "size"), npxg=("npxg", "sum")).reset_index())

    out = (regains.merge(per, on=["match_id", "possession"], how="left")
                  .merge(own_per, on=["match_id", "possession"], how="left")
                  .merge(sea_shots, on=["match_id", "possession"], how="left"))
    out["kept_secs"] = (out["end_min"] - out["elapsed_min"]) * 60
    out[["shots", "passes_completed"]] = out[["shots", "passes_completed"]].fillna(0).astype(int)
    out["npxg"] = out["npxg"].fillna(0.0)
    out["reached_opp_half"] = out["max_x"] >= own_half_x
    out["reached_final_third"] = out["max_x"] >= final_third_x
    # Got further forward: own-half regains that reached the opponent's half,
    # or opponent-half regains that reached the final third
    out["progressed"] = np.where(out["own_half"], out["reached_opp_half"], out["reached_final_third"])
    out["kept_10s"] = out["kept_secs"] >= kept_secs
    out["lost_quickly"] = out["kept_secs"] < quick_reloss_secs
    return out.drop(columns=["end_min"])

def regain_summary(regains, by=("match_id",)):
    """Regains per group, and what came of the active ones."""
    r = regains.copy()
    act = r["how"] == "Active"
    for c in ["kept_10s", "progressed", "lost_quickly", "own_half"]:
        r[f"act_{c}"] = act & r[c]
    r["act_shots"] = np.where(act, r["shots"], 0)
    r["act_npxg"] = np.where(act, r["npxg"], 0.0)
    r["act_x"] = np.where(act, r["location_x"], np.nan)
    out = r.groupby(list(by)).agg(
        regains=("how", "size"),
        active=("how", lambda s: (s == "Active").sum()),
        keeper=("how", lambda s: (s == "Keeper").sum()),
        restart=("how", lambda s: (s == "Restart").sum()),
        active_own_half=("act_own_half", "sum"),
        active_avg_x=("act_x", "mean"),
        active_kept_10s=("act_kept_10s", "sum"),
        active_progressed=("act_progressed", "sum"),
        active_lost_quickly=("act_lost_quickly", "sum"),
        active_shots=("act_shots", "sum"),
        active_npxg=("act_npxg", "sum"),
    ).reset_index()
    for c in ["own_half", "kept_10s", "progressed", "lost_quickly"]:
        out[f"active_{c}_pct"] = out[f"active_{c}"] / out["active"]
    return out

def defending_profile(events, team_ids, by=("match_id",)):
    """Three separate views of how a team defended (distances in yards from its own goal):
      engagement : tackles, interceptions, dribbled past -> StatsBomb Defensive Distance
      pressing   : pressures -> where and how much they closed the ball down
      last-ditch : clearances and blocks -> how much they were pinned back
    """
    d = events[events["team_id"].isin(team_ids) & events["location_x"].notna()].copy()
    is_tackle = (d["type"] == "Duel") & d["duel_type"].eq("Tackle")
    d["engage"] = is_tackle | d["type"].isin(["Interception", "Dribbled Past"])
    d["press"] = d["type"] == "Pressure"
    d["deep"] = d["type"].isin(["Clearance", "Block"])
    d["engage_x"] = np.where(d["engage"], d["location_x"], np.nan)
    d["press_x"] = np.where(d["press"], d["location_x"], np.nan)
    d["press_opp_half"] = d["press"] & (d["location_x"] >= own_half_x)
    d["counterpress"] = d["press"] & d["counterpress"].eq(True)
    d["deep_own_third"] = d["deep"] & (d["location_x"] < own_third_x)
    out = d.groupby(list(by)).agg(
        engagements=("engage", "sum"), defensive_distance=("engage_x", "mean"),
        pressures=("press", "sum"), pressure_height=("press_x", "mean"),
        pressures_opp_half=("press_opp_half", "sum"), counterpressures=("counterpress", "sum"),
        clearances_blocks=("deep", "sum"), clearances_blocks_own_third=("deep_own_third", "sum"),
    ).reset_index()
    out["pressures_opp_half_pct"] = out["pressures_opp_half"] / out["pressures"]
    return out

def plot_regain_map(ax, regains, color, title=None, block_x=None):
    """Where a team won the ball back in open play (own goal at the bottom, attacking upwards).

    Solid = the possession got further forward (own-half regain reached the opponent's half, or an
    opponent-half regain reached the final third); light = it didn't.
    block_x : optional average distance of defensive actions from own goal, drawn as a line.
    """
    pitch = VerticalPitch(pitch_type="statsbomb", pitch_color=surface, line_color=grid, linewidth=1)
    pitch.draw(ax=ax)
    a = regains[regains["how"] == "Active"]
    for x0 in (40, 80):
        pitch.lines(x0, 0, x0, 80, ax=ax, color=grid, lw=1, zorder=1)
    if block_x is not None:
        pitch.lines(block_x, 0, block_x, 80, ax=ax, color=darker(color), lw=1.5, zorder=2)
        pitch.annotate(f"defensive\ndistance {block_x:.0f} yds", (block_x, 82), ax=ax, ha="left",
                       va="center", fontsize=8, color=text_2, annotation_clip=False)
    stuck = a[~a["progressed"]]
    went = a[a["progressed"]]
    pitch.scatter(stuck["location_x"], stuck["location_y"], s=26, ax=ax, facecolors=to_rgba(color, 0.25),
                  edgecolors=to_rgba(darker(color), 0.6), linewidths=0.5, zorder=3)
    pitch.scatter(went["location_x"], went["location_y"], s=40, ax=ax, facecolors=color,
                  edgecolors=darker(color), linewidths=1, zorder=4)
    n = len(a)
    pct = lambda col: f"{a[col].mean():.0%}" if n else "–"
    ax.text(0.0, -0.02,
            f"Active regains: {n} · own half: {int(a['own_half'].sum())} ({pct('own_half')})\n"
            f"Kept 10+ s: {pct('kept_10s')} · lost within 5 s: {pct('lost_quickly')}\n"
            f"Got further forward: {pct('progressed')} · shots: {int(a['shots'].sum())}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=text_2, linespacing=1.5)
    if title:
        ax.set_title(title, loc="left", color=text, fontsize=11)
    return ax


def regain_key(fig, color=None, loc="upper right", bbox=(0.995, 1.0)):
    """Legend for the regain map."""
    c = color or text_2
    handles = [plt.scatter([], [], s=40, facecolors=c, edgecolors=darker(c), linewidths=1),
               plt.scatter([], [], s=26, facecolors=to_rgba(c, 0.25), edgecolors=to_rgba(darker(c), 0.6),
                           linewidths=0.5)]
    labels = ["Possession got further forward", "Didn't"]
    fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox, ncol=2, frameon=False,
               fontsize=9, labelcolor=text, scatterpoints=1)


def plot_team_regain_maps(sea_id, team_matches, regains, team_name, scores, block=None, title=None):
    """One regain map per match. block: {match_id: avg defensive action x} from block_height(), optional."""
    team_matches = team_matches.sort_values("match_week")
    n = len(team_matches)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 7.4))
    fig.patch.set_facecolor(surface)
    for ax, (_, r) in zip(np.atleast_1d(axes), team_matches.iterrows()):
        mid, opp_id = r["match_id"], r["opponent_id"]
        plot_regain_map(ax, regains[regains["match_id"] == mid], team_colors[sea_id],
                        title=f"MD{r['match_week']}: {scores[mid]} {team_name[opp_id]}",
                        block_x=(block or {}).get(mid))
    if title is None:
        season = f" {team_matches['season'].iloc[0]}" if "season" in team_matches else ""
        title = f"{team_name[sea_id]} at the Women's World Cup{season}: where they won the ball back"
    fig.suptitle(title, x=0.01, y=0.995, ha="left", color=text, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    regain_key(fig, team_colors[sea_id])
    return fig

tabs = [("home", "Home"), ("stats", "Stats"), ("xg_race", "xG race"), ("shots", "Shots"), ("goals", "Goals"),
        ("losses", "Losses"), ("regains", "Regains"), ("build_up", "Build-up"), ("attack", "Attack"),
        ("possessions", "Possessions"), ("game_state", "Game state")]
tab_names = [label for _, label in tabs]

finished_phrase = {"Champions": "champions", "Runners-up": "runners-up", "Third Place": "third place",
                   "Fourth": "fourth", "Quarter-finals": "quarter-finalists",
                   "Round of 16": "out in the round of 16", "Group stage": "out in the group stage"}

def folder_name(name):
    """Team name -> folder name: "Thailand W" -> "thailand", "United States Women's" -> "united_states"."""
    name = re.sub(r" (Women's|W)$", "", name)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def match_folder(card):
    """plots/<team>/md<N>_<opponent>/"""
    return plots_dir / folder_name(card["Team"]) / f"md{card['MD']}_{folder_name(card['Opponent'])}"


def overall_folder(team):
    """plots/<team>/overall/"""
    return plots_dir / folder_name(team) / "overall"


def save_figure(fig, folder, file_name, dpi=100, tight=False):
    """Save a figure into `folder` (created if needed), close it, and return the path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight" if tight else None)
    plt.close(fig)
    return path


def save_tab(fig, card, tab_label):
    """Save one match tab as plots/<team>/md<N>_<opponent>/<NN>_<tab>.png"""
    i = tab_names.index(tab_label)
    return save_figure(fig, match_folder(card), f"{i + 1:02d}_{tabs[i][0]}.png")

def scorer_lines(team, scorers, width=140):
    """Match-card scorers as header lines: 'Team: A 12', 53'; B 48' (og)'.

    Wraps only between players, so one player's goals never split across lines.
    Returns [] if the team didn't score.
    """
    if not isinstance(scorers, str) or scorers.strip() in ("", "–"):
        return []
    lines, line = [], f"{team}: "
    for i, part in enumerate(scorers.split("; ")):
        sep = "" if i == 0 else "; "
        if i > 0 and len(line) + len(sep) + len(part) > width:
            lines.append(line + ";")
            line = part
        else:
            line += sep + part
    lines.append(line)
    return lines


def add_header_lines(cards, stakes_short=None):
    """Add the header text to the match cards: context_line, stakes_line, scorers_line.

    stakes_short: your short stakes lines, in the same row order as `cards`.
                  Until you write them, the full Stakes sentence is used (the header wraps it).
    """
    out = cards.copy()
    out["context_line"] = ("FIFA Women's World Cup " + out["Tournament"].astype(str) + "  ·  Matchday " + out["MD"].astype(str)
                           + "  ·  HT " + out["HT"] + "  ·  " + out["Opponent"] + ": "
                           + out["Opponent finished"].map(finished_phrase) + ", "
                           + out["Opponent elsewhere"] + " elsewhere")
    out["stakes_line"] = list(stakes_short) if stakes_short is not None else out["Stakes"]
    out["scorers_line"] = ["\n".join(scorer_lines(t, s) + scorer_lines(o, x))
                           for t, s, o, x in zip(out["Team"], out["Scorers"], out["Opponent"], out["Opponent scorers"])]
    return out

def _text_width(fig, s, **kw):
    """Width of a piece of text, as a fraction of the figure width."""
    t = fig.text(0, 0, s, **kw)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.bbox.width
    t.remove()
    return w


def draw_match_header(fig, team, score, opp, context, stakes, scorers, current, sea_color,
                      opp_color=text_2, top=0.955, stakes_wrap=120):
    """Centred broadcast-style header with a tab bar. Returns the figure y where the tab's panels start."""
    page = fig.add_axes([0, 0, 1, 1], zorder=-1)      # invisible full-page axes, so Jupyter displays the figure
    page.axis("off")
    fig.canvas.draw()
    y = top

    # result: score in the centre, team names either side
    w_score = _text_width(fig, score, fontsize=26, weight="bold")
    fig.text(0.5, y, score, ha="center", va="baseline", fontsize=26, weight="bold", color=text)
    fig.text(0.5 - w_score / 2 - 0.018, y, team, ha="right", va="baseline", fontsize=22, weight="bold", color=sea_color)
    fig.text(0.5 + w_score / 2 + 0.018, y, opp, ha="left", va="baseline", fontsize=22, weight="bold", color=opp_color)
    y -= 0.030

    fig.text(0.5, y, context, ha="center", va="top", fontsize=10.5, color=text_2)
    y -= 0.020

    stakes = textwrap.fill(stakes, width=stakes_wrap) if isinstance(stakes, str) else ""
    if stakes:
        n_lines = stakes.count("\n") + 1
        fig.text(0.5, y, stakes, ha="center", va="top", fontsize=10.5, style="italic", color=text_2,
                 linespacing=1.4, multialignment="center")
        y -= 0.0155 * n_lines + 0.007

    if scorers:
        n_lines = scorers.count("\n") + 1
        fig.text(0.5, y, scorers, ha="center", va="top", fontsize=9.5, color=text_2,
                 linespacing=1.4, multialignment="center")
        y -= 0.0175 * n_lines + 0.008

    fig.add_artist(Line2D([0.05, 0.95], [y, y], color=grid, lw=1, transform=fig.transFigure))
    y -= 0.012

    # tab bar, centred, with the current tab highlighted and underlined
    gap = _text_width(fig, "    ", fontsize=9)
    styles = [dict(fontsize=9, weight="bold", color=sea_color) if t == current else dict(fontsize=9, color=muted)
              for t in tab_names]
    widths = [_text_width(fig, t.upper(), **s) for t, s in zip(tab_names, styles)]
    x = 0.5 - (sum(widths) + gap * (len(tab_names) - 1)) / 2
    for t, s, w in zip(tab_names, styles, widths):
        label = fig.text(x, y, t.upper(), ha="left", va="top", **s)
        label.set_gid(f"tab:{t}")
        if t == current:
            fig.add_artist(Line2D([x, x + w], [y - 0.016, y - 0.016], color=sea_color, lw=2,
                                  transform=fig.transFigure))
        x += w + gap
    return y - 0.035


def tab_footer(fig, note=""):
    fig.text(0.5, 0.018, "Data: StatsBomb open data" + (f"  ·  {note}" if note else ""),
             ha="center", fontsize=7.5, color=muted)


def plot_stat_comparison(ax, rows, sea_color, opp_color="#9a9a9a", track="#e6e6e6",
                         sea_label=None, opp_label=None):
    """Broadcast-style comparison: SEA values on the left, stat names in the middle, opponent on the right.

    rows: list of dicts, each either
        a section header: dict(header="From goal kicks")
        or a stat row with
            label, sea, opp, fmt
            bar             : "share", "pct" or None
            lower_is_better : True for PPDA (the lower value is bold)
            no_bold         : True for style measures where higher isn't better
    """
    top = 0.9 if sea_label else 0
    n = len(rows)
    ax.set_xlim(-1, 1)
    ax.set_ylim(n + top, 0)
    ax.axis("off")
    if sea_label:
        ax.text(-0.98, 0.45, sea_label, ha="left", va="center", fontsize=13, weight="bold", color=sea_color)
        ax.text(0.98, 0.45, opp_label, ha="right", va="center", fontsize=13, weight="bold", color=opp_color)

    for i, r in enumerate(rows):
        if "header" in r:                                   # section label, no values
            ax.text(-0.98, top + i + 0.62, r["header"].upper(), ha="left", va="center", fontsize=8.5,
                    weight="bold", color=muted)
            ax.plot([-0.98, 0.98], [top + i + 0.85, top + i + 0.85], color=grid, lw=0.8)
            continue

        y, by = top + i + 0.32, top + i + 0.70              # text line, bar line
        sea, opp = r["sea"], r["opp"]
        if not isinstance(sea, (int, float, np.number)) or r.get("no_bold"):
            sea_bold = opp_bold = False
        elif r.get("lower_is_better"):
            sea_bold, opp_bold = sea < opp, opp < sea
        else:
            sea_bold, opp_bold = sea > opp, opp > sea

        ax.text(0, y, r["label"], ha="center", va="center", fontsize=10, color=text_2)
        ax.text(-0.98, y, r["fmt"](sea), ha="left", va="center", fontsize=13, color=sea_color,
                weight="bold" if sea_bold else "normal")
        ax.text(0.98, y, r["fmt"](opp), ha="right", va="center", fontsize=13, color=opp_color,
                weight="bold" if opp_bold else "normal")

        h = 0.16
        if r.get("bar") == "share":
            ax.barh(by, 1.96, left=-0.98, height=h, color=track)
            if sea + opp > 0:
                s = sea / (sea + opp)
                ax.barh(by, 1.96 * s, left=-0.98, height=h, color=sea_color)
                ax.barh(by, 1.96 * (1 - s), left=-0.98 + 1.96 * s, height=h, color=opp_color)
        elif r.get("bar") == "pct":
            ax.barh(by, -0.96, left=-0.02, height=h, color=track)
            ax.barh(by, 0.96, left=0.02, height=h, color=track)
            ax.barh(by, -0.96 * sea / 100, left=-0.02, height=h, color=sea_color)
            ax.barh(by, 0.96 * opp / 100, left=0.02, height=h, color=opp_color)
    return ax

def match_rows(table, card):
    """This match's rows of a per-match table, indexed by side ('sea' / 'opp')."""
    return table[table["match_id"] == card["match_id"]].set_index("side")

def plot_stats_tab(card, team_stats):
    """Tab 1: the 13-row comparison for one match. Rates in team_stats are fractions (0–1); shown as %."""
    sea_id = card["team_id"]
    st = match_rows(team_stats, card)
    s, o = st.loc["sea"], st.loc["opp"]

    for col in ["possession_pct", "pass_completion_pct", "field_tilt"]:
        if st[col].max() > 1:
            print(f"Check {col}: max is {st[col].max():.2f}, so it looks like it's already ×100 in team_stats")

    pct = lambda x: f"{x:.0f}%"
    num = lambda x: f"{x:.0f}"
    pens = lambda t: f"{t[0]}/{t[1]}" if t[1] else "–"
    rows = [
        dict(label="Possession", sea=s["possession_pct"] * 100, opp=o["possession_pct"] * 100, fmt=pct, bar="share"),
        dict(label="Passes", sea=s["passes"], opp=o["passes"], fmt=num, bar="share"),
        dict(label="Pass completion", sea=s["pass_completion_pct"] * 100, opp=o["pass_completion_pct"] * 100,
             fmt=pct, bar="pct"),
        dict(label="Field tilt", sea=s["field_tilt"] * 100, opp=o["field_tilt"] * 100, fmt=pct, bar="share"),
        dict(label="Final-third entries", sea=s["entries_into_final_third"], opp=o["entries_into_final_third"],
             fmt=num, bar="share"),
        dict(label="Shots", sea=s["shots"], opp=o["shots"], fmt=num, bar="share"),
        dict(label="Shots on target", sea=s["shots_on_target"], opp=o["shots_on_target"], fmt=num, bar="share"),
        dict(label="xG", sea=s["xg"], opp=o["xg"], fmt=lambda x: f"{x:.2f}", bar="share"),
        dict(label="Penalties (scored/taken)", sea=(int(s["pens_scored"]), int(s["pens_taken"])),
             opp=(int(o["pens_scored"]), int(o["pens_taken"])), fmt=pens, bar=None),
        dict(label="Corners", sea=s["corners"], opp=o["corners"], fmt=num, bar="share"),
        dict(label="Fouls", sea=s["fouls"], opp=o["fouls"], fmt=num, bar="share", no_bold=True),
        dict(label="Cards (yellow / 2nd yellow / red)",
             sea=(int(s["yellow_cards"]), int(s["second_yellows"]), int(s["straight_reds"])),
             opp=(int(o["yellow_cards"]), int(o["second_yellows"]), int(o["straight_reds"])),
             fmt=lambda t: f"{t[0]} / {t[1]} / {t[2]}", bar=None),
        dict(label="PPDA (lower = more pressing)", sea=s["ppda"], opp=o["ppda"],
             fmt=lambda x: f"{x:.1f}", bar=None, lower_is_better=True),
    ]

    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "Stats", team_colors[sea_id])
    ax = fig.add_axes([0.17, 0.08, 0.66, top - 0.10])
    ax.set_facecolor(surface)
    plot_stat_comparison(ax, rows, sea_color=team_colors[sea_id], sea_label=card["Team"], opp_label=card["Opponent"])
    tab_footer(fig, "xG includes penalties  ·  Field tilt = share of final-third touches  ·  "
                    "PPDA = opponent passes per defensive action")
    return fig

state_colors = {"level": "#f1f0ec", "trailing by 1": "#dcdbd5", "trailing by 2": "#c4c3bc",
                "trailing by 3+": "#a9a8a1"}          # leading uses a light version of the team colour
state_labels = {"leading": "Ahead", "level": "Level", "trailing by 1": "−1", "trailing by 2": "−2",
                "trailing by 3+": "−3 or worse"}

def merge_state_spells(intervals):
    """Join consecutive spells in the same state (0–3 and 0–4 are both 'trailing by 3+')."""
    s = intervals.sort_values("start").copy()
    s["block"] = (s["game_state"] != s["game_state"].shift()).cumsum()
    return (s.groupby("block")
             .agg(game_state=("game_state", "first"), start=("start", "min"), end=("end", "max"))
             .reset_index(drop=True))

red_card_color, yellow_card_color = "#d32f2f", "#f2c200"

def plot_xg_race_tab(card, tl, state_intervals, match_end, half_time, team_names, red_cards=None):
    """Tab 2: cumulative xG with the score state above it and three numbers below."""
    mid, sea_id, opp_id = card["match_id"], card["team_id"], card["opponent_id"]
    me, ht = match_end[mid], half_time[mid]
    sea_color = team_colors[sea_id]
    spells = merge_state_spells(state_intervals[state_intervals["match_id"] == mid])

    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "xG race", sea_color)

    # score-state strip
    ax_s = fig.add_axes([0.08, top - 0.035, 0.74, 0.026])
    for _, sp in spells.iterrows():
        color = to_rgba(sea_color, 0.35) if sp["game_state"] == "leading" else state_colors[sp["game_state"]]
        ax_s.axvspan(sp["start"], sp["end"], color=color, lw=0)
        if sp["end"] - sp["start"] > 6:                 # label only spells wide enough to read
            ax_s.text((sp["start"] + sp["end"]) / 2, 0.5, state_labels[sp["game_state"]],
                      ha="center", va="center", fontsize=8.5, color=text)
    ax_s.set_xlim(0, me)
    ax_s.set_ylim(0, 1)
    ax_s.axis("off")
    ax_s.text(0, 1.2, f"Score state for {card['Team']}", transform=ax_s.transAxes, fontsize=8.5, color=text_2)

    # xG timeline (your Piece 2 function)
    ax = fig.add_axes([0.08, 0.36, 0.74, top - 0.065 - 0.36])
    plot_xg_timeline(tl, me, ht, sea_id, opp_id, team_names, ax=ax, symbol_key=False)
    if ax.get_legend():
        ax.get_legend().remove()                        # the end-of-line labels already name the teams

    # third goal conceded
    conceded = tl[(tl["side"] == "opp") & tl["kind"].isin(["goal", "pen_goal", "og"])].sort_values("elapsed_min")
    if len(conceded) >= 3:
        x3 = conceded["elapsed_min"].iloc[2]
        ax.axvline(x3, color=text_2, lw=1, ls=":")
        ax.text(x3, ax.get_ylim()[1] * 0.97, f" 3rd goal conceded, {card['3rd goal conceded']}",
                fontsize=8.5, color=text_2, va="top", ha="left")

    # sending-offs: a card symbol above the plot at the minute, and a faint dashed line down
    if red_cards is not None:
        top_edge = transforms.blended_transform_factory(ax.transData, ax.transAxes)   # x = minute, y = axes fraction
        card_shape = [(-0.36, -0.5), (0.36, -0.5), (0.36, 0.5), (-0.36, 0.5)]         # upright card, 0.72 : 1
        for _, rc in red_cards[red_cards["match_id"] == card["match_id"]].iterrows():
            x = rc["elapsed_min"]
            ax.axvline(x, color=red_card_color, lw=0.8, ls="--", alpha=0.6, zorder=2)
            if rc["card"] == "Second Yellow":                                        # yellow card peeking out behind
                behind = transforms.offset_copy(top_edge, fig=fig, x=-3, y=2, units="points")
                ax.plot([x], [1.035], marker=card_shape, ms=12, color=yellow_card_color, mec=surface, mew=0.6,
                        transform=behind, clip_on=False, zorder=8)
            ax.plot([x], [1.035], marker=card_shape, ms=12, color=red_card_color, mec=surface, mew=0.6,
                    transform=top_edge, clip_on=False, zorder=9)

    # xG at half-time, written next to each line where it meets the half-time line
    def xg_at(side, t):
        s = tl[(tl["side"] == side) & (tl["elapsed_min"] <= t)]
        return s["cum_xg"].max() if len(s) else 0.0

    ht_xg = {"sea": xg_at("sea", ht), "opp": xg_at("opp", ht)}
    higher = max(ht_xg, key=ht_xg.get)                  # higher line's label goes above, lower's below
    for side, color in [("sea", sea_color), ("opp", text_2)]:
        v = ht_xg[side]
        above = side == higher
        # a goal, penalty or own goal of this side just before the break: move the label further from the line
        marked = tl.loc[(tl["side"] == side) & (tl["kind"] != "shot"), "elapsed_min"]
        dy = 20 if ((marked > ht - 8) & (marked <= ht)).any() else 7
        ax.plot(ht, v, "o", ms=4, color=color, zorder=6)
        ax.annotate(f"HT {v:.2f}", (ht, v), xytext=(-5, dy if above else -dy), textcoords="offset points",
                    ha="right", va="bottom" if above else "top", fontsize=9, weight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc=surface, ec="none", alpha=0.85), zorder=7)

    # two numbers
    mins = (spells["end"] - spells["start"]).groupby(spells["game_state"]).sum()
    numbers = [(f"{mins.get('level', 0) + mins.get('leading', 0):.0f} min", "level or ahead"),
               (f"{mins.get('trailing by 3+', 0):.0f} min", "three or more goals behind")]
    for i, (big, small) in enumerate(numbers):
        x = 0.33 + i * 0.34
        fig.text(x, 0.19, big, ha="center", fontsize=22, weight="bold", color=text)
        fig.text(x, 0.165, small, ha="center", va="top", fontsize=9.5, color=text_2)

    fig.text(0.5, 0.265, "Ball = goal  ·  P = penalty  ·  OG = own goal  ·  HT = xG at half-time",
             ha="center", fontsize=9, color=muted)
    tab_footer(fig, "xG includes penalties")
    return fig

def origin_breakdown(shots, match_id, side, own_goals):
    """Goals, shots and npxG by origin for one side of one match, plus penalty and own-goal rows."""
    s = shots[(shots["match_id"] == match_id) & (shots["side"] == side)]
    rows = [r for r in origin_order if r != "Penalty"]
    t = (s[s["origin"] != "Penalty"].groupby("origin")
         .agg(goals=("is_goal", "sum"), shots=("npxg", "size"), npxg=("npxg", "sum"))
         .reindex(rows, fill_value=0))
    pens = s[s["origin"] == "Penalty"]
    t.loc["Penalty"] = [pens["is_goal"].sum(), len(pens), np.nan]    # no npxG for penalties
    t.loc["Own goal"] = [own_goals, np.nan, np.nan]                  # own goals aren't shots
    return t

track = "#ecebe7"          # light background bar; skip if you already have one


def plot_origin_comparison(ax, sea, opp, sea_color, sea_label, opp_label):
    """Both teams by origin: goals (bold, at the edges), npxG · shots, and npxG bars from the middle.

    sea, opp: tables from origin_breakdown (same rows, same order).
    """
    rows = list(sea.index)
    n = len(rows) + 1                                   # + the "All goals" row
    ax.set_xlim(-1, 1)
    ax.set_ylim(n + 0.9, 0)
    ax.axis("off")

    # column headings
    ax.text(-0.98, 0.35, sea_label, ha="left", va="center", fontsize=11, weight="bold", color=sea_color)
    ax.text(0.98, 0.35, opp_label, ha="right", va="center", fontsize=11, weight="bold", color=opp_color)
    ax.text(-0.98, 0.75, "goals  ·  npxG  ·  shots", ha="left", va="center", fontsize=8, color=muted)
    ax.text(0.98, 0.75, "shots  ·  npxG  ·  goals", ha="right", va="center", fontsize=8, color=muted)

    scale = 0.9 / max(np.nanmax(sea["npxg"]), np.nanmax(opp["npxg"]), 0.3)   # one scale for both teams
    for i, name in enumerate(rows):
        y, by = 1.2 + i + 0.30, 1.2 + i + 0.68          # text line, bar line
        ax.text(0, y, name, ha="center", va="center", fontsize=9.5, color=text)
        for r, color, sign in [(sea.loc[name], sea_color, -1), (opp.loc[name], opp_color, 1)]:
            ha = "left" if sign < 0 else "right"
            scored = r["goals"] > 0
            ax.text(0.98 * sign, y, f"{int(r['goals'])}", ha=ha, va="center", fontsize=13,
                    weight="bold" if scored else "normal", color=color if scored else muted)
            if name == "Penalty":
                detail = f"{int(r['goals'])}/{int(r['shots'])} scored" if r["shots"] else "none"
            elif name == "Own goal":
                detail = ""
            elif sign < 0:
                detail = f"{r['npxg']:.2f}  ·  {int(r['shots'])}"
            else:
                detail = f"{int(r['shots'])}  ·  {r['npxg']:.2f}"
            ax.text(0.88 * sign, y, detail, ha=ha, va="center", fontsize=8.5, color=text_2)
            if pd.notna(r["npxg"]):
                ax.barh(by, 0.9 * sign, left=0.015 * sign, height=0.14, color=track)
                ax.barh(by, r["npxg"] * scale * sign, left=0.015 * sign, height=0.14, color=color)

    # total row: always equals the score
    y = 1.2 + len(rows) + 0.35
    ax.plot([-0.98, 0.98], [y - 0.35, y - 0.35], color=grid, lw=1)
    ax.text(0, y, "All goals", ha="center", va="center", fontsize=9.5, weight="bold", color=text)
    ax.text(-0.98, y, f"{int(sea['goals'].sum())}", ha="left", va="center", fontsize=13, weight="bold", color=sea_color)
    ax.text(0.98, y, f"{int(opp['goals'].sum())}", ha="right", va="center", fontsize=13, weight="bold", color=opp_color)
    return ax

set_piece_origins = ["Direct set piece", "Delayed set piece", "Final-third throw-in"]

def plot_shots_tab(card, shots):
    """Tab 3: both teams' non-penalty shots on one pitch, plus goals, npxG and shots by origin for both teams."""
    mid, sea_id = card["match_id"], card["team_id"]
    sea_color = team_colors[sea_id]
    s = shots[(shots["match_id"] == mid) & (shots["origin"] != "Penalty")]

    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "Shots", sea_color)

    # left: full pitch, SEA shooting up, opponent shooting down
    ax = fig.add_axes([0.02, 0.10, 0.44, top - 0.12])
    pitch = VerticalPitch(pitch_type="statsbomb", pitch_color=surface, line_color=grid, linewidth=1)
    pitch.draw(ax=ax)
    for side, color in [("opp", opp_color), ("sea", sea_color)]:
        d = s[s["side"] == side]
        x = d["location_x"] if side == "sea" else 120 - d["location_x"]
        y = d["location_y"] if side == "sea" else 80 - d["location_y"]
        goal = d["is_goal"].to_numpy()
        pitch.scatter(x[~goal], y[~goal], s=_shot_sizes(d["xg"][~goal]), ax=ax,
                      facecolors=to_rgba(color, 0.28), edgecolors=darker(color), linewidths=0.8, zorder=3)
        pitch.scatter(x[goal], y[goal], s=_shot_sizes(d["xg"][goal]), ax=ax,
                      facecolors=color, edgecolors=darker(color), linewidths=1.2, zorder=4)
    pitch.annotate(f"{card['Team']} shooting ↑", (63, 1.5), ax=ax, ha="left", va="bottom",
                   fontsize=9, color=sea_color)
    pitch.annotate(f"{card['Opponent']} shooting ↓", (57, 1.5), ax=ax, ha="left", va="top",
                   fontsize=9, color=text_2)
    shot_size_key(ax, values=(0.05, 0.2, 0.5), loc="upper center", bbox=(0.5, -0.005))

    # right: both teams by origin
    sea_t = origin_breakdown(shots, mid, "sea", str(card["Scorers"]).count("(og)"))
    opp_t = origin_breakdown(shots, mid, "opp", str(card["Opponent scorers"]).count("(og)"))
    ax2 = fig.add_axes([0.50, 0.25, 0.47, top - 0.06 - 0.25])
    ax2.set_facecolor(surface)
    plot_origin_comparison(ax2, sea_t, opp_t, sea_color, card["Team"], card["Opponent"])

    # one summary line, for chances and goals conceded
    sp_x, tot_x = opp_t.loc[set_piece_origins, "npxg"].sum(), opp_t["npxg"].sum()
    sp_g, tot_g = int(opp_t.loc[set_piece_origins, "goals"].sum()), int(opp_t["goals"].sum())
    fig.text(0.735, 0.20, f"Set pieces and throw-ins: {sp_x:.2f} of {tot_x:.2f} npxG\n"
                          f"and {sp_g} of {tot_g} goals conceded",
             ha="center", va="top", fontsize=11.5, weight="bold", color=text, linespacing=1.35)

    tab_footer(fig, "Size = xG  ·  bars = npxG  ·  penalties not on the pitch  ·  "
                    "origin = how the possession or restart led to the shot")
    return fig

category_colors = {"Restarts & turnovers": "#b4441f", "Low-chance": "#7a6fb0", "Quality gap": "#52514e"}
flag_columns = [("set_piece", "Set piece /\nthrow-in"), ("after_own_half_loss", "After own-\nhalf loss"),
                ("after_goal_kick", "≤30s after own\ngoal kick"), ("low_chance", "Low chance\n(< 0.10)")]

def minute_label(minute, period):
    """StatsBomb minute (0-based) -> match-clock label: 11 -> 12', first-half stoppage -> 45+2'."""
    m = int(minute)
    if period == 1 and m >= 45:
        return f"45+{m - 44}'"
    if period == 2 and m >= 90:
        return f"90+{m - 89}'"
    return f"{m + 1}'"


def build_goal_ledger(events, shots, losses, goal_kicks, sea_ids):
    """One row per goal (shots and own goals), with the score after it and, for goals conceded, what led to it."""
    cols = ["match_id", "index", "period", "minute", "elapsed_min", "side", "player", "possession", "origin", "npxg"]
    g = shots.loc[shots["is_goal"], cols].copy()
    og = events[events["type"] == "Own Goal Against"]
    og = pd.DataFrame({"match_id": og["match_id"], "index": og["index"], "period": og["period"],
                       "minute": og["minute"], "elapsed_min": og["elapsed_min"],
                       "side": np.where(og["team_id"].isin(sea_ids), "opp", "sea"),
                       "player": og["player"], "possession": og["possession"], "origin": "Own goal", "npxg": np.nan})
    g = pd.concat([g, og], ignore_index=True).sort_values(["match_id", "elapsed_min"]).reset_index(drop=True)
    g["minute_label"] = [minute_label(m, p) for m, p in zip(g["minute"], g["period"])]
    by_match = g.groupby("match_id")
    g["sea_score"] = by_match["side"].transform(lambda s: (s == "sea").cumsum())
    g["opp_score"] = by_match["side"].transform(lambda s: (s == "opp").cumsum())
    g["score_after"] = g["sea_score"].astype(str) + "–" + g["opp_score"].astype(str)

    own_half = losses[losses["won_x"] < 60]                     # where the opponent won the ball ("won at")
    loss_keys = set(zip(own_half["match_id"], own_half["next_possession"].astype("Int64")))
    gk = goal_kicks[(goal_kicks["side"] == "sea") & goal_kicks["opp_shot_30s"]]
    gk_keys = set(zip(gk["match_id"], gk["next_shot_index"].astype("Int64")))

    conceded = g["side"] == "opp"
    g["set_piece"] = conceded & g["origin"].isin(set_piece_origins)
    g["after_own_half_loss"] = conceded & pd.Series([(m, p) in loss_keys for m, p in zip(g["match_id"], g["possession"])])
    g["after_goal_kick"] = conceded & pd.Series([(m, i) in gk_keys for m, i in zip(g["match_id"], g["index"])])
    g["low_chance"] = conceded & ~g["origin"].isin(["Penalty", "Own goal"]) & (g["npxg"] < 0.10)

    # restarts (set piece, throw-in, penalty) and turnovers (own-half loss, lost goal kick); own goals are judged by these
    # same flags, so an own goal only counts here if the move behind it started from a restart or a turnover
    restart_or_turnover = (g["set_piece"] | g["after_own_half_loss"] | g["after_goal_kick"] | (g["origin"] == "Penalty"))
    g["category"] = np.select([~conceded, restart_or_turnover, g["low_chance"]],
                              ["", "Restarts & turnovers", "Low-chance"], "Quality gap")
    return g

def _wrap_name(fig, name, max_width, **kw):
    """Keep a name on one line if it fits; otherwise move words from the end onto a second line until the first line fits."""
    if _text_width(fig, name, **kw) <= max_width:
        return name
    words = name.split()
    for k in range(len(words) - 1, 0, -1):
        first = " ".join(words[:k])
        if _text_width(fig, first, **kw) <= max_width:
            return first + "\n" + " ".join(words[k:])
    return words[0] + "\n" + " ".join(words[1:])

def plot_goals_tab(card, ledger, shots):
    """Tab 4: every goal in order, with what led to each goal conceded."""
    mid, sea_id = card["match_id"], card["team_id"]
    sea_color = team_colors[sea_id]
    g = ledger[ledger["match_id"] == mid]

    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "Goals", sea_color)

    # column positions (figure fractions)
    x_min, x_score, x_player, x_origin = 0.05, 0.11, 0.17, 0.41
    x_flags = [0.575, 0.65, 0.725, 0.80]
    x_npxg, x_cat = 0.858, 0.95

    # column headings
    y = top - 0.01
    head = dict(fontsize=8.5, color=muted, va="bottom", linespacing=1.2)
    for x, label in [(x_min, "Min"), (x_score, "Score"), (x_player, "Scorer"), (x_origin, "Origin")]:
        fig.text(x, y, label, ha="left", **head)
    any_conceded = (g["side"] == "opp").any()
    if any_conceded:
        for x, (_, label) in zip(x_flags, flag_columns):
            fig.text(x, y, label, ha="center", **head)
        fig.text(x_cat, y, "Type", ha="right", **head)
    fig.text(x_npxg, y, "npxG", ha="right", **head)
    y -= 0.012
    fig.add_artist(Line2D([0.05, 0.95], [y, y], color=grid, lw=1, transform=fig.transFigure))
    row_h = min(0.036, (y - 0.30) / max(len(g), 1))     # rows shrink only if a match has very many goals

    # one row per goal
    for _, r in g.iterrows():
        y -= row_h
        yc = y + row_h / 2
        conceded = r["side"] == "opp"
        color = text if conceded else sea_color
        fig.text(x_min, yc, r["minute_label"], fontsize=10, color=color, va="center")
        fig.text(x_score, yc, r["score_after"], fontsize=10, color=color, va="center", weight="bold")
        fig.text(x_player, yc, _wrap_name(fig, r["player"], x_origin - x_player - 0.012, fontsize=10),
                 fontsize=10, color=color, va="center", linespacing=1.15)
        fig.text(x_origin, yc, r["origin"], fontsize=10, color=color, va="center")
        if conceded:
            for x, (flag, _) in zip(x_flags, flag_columns):
                if r[flag]:
                    fig.add_artist(Line2D([x], [yc], marker="o", ms=7, color=text_2, transform=fig.transFigure))
            fig.text(x_cat, yc, r["category"].replace(" & ", " &\n"), fontsize=9, color=category_colors[r["category"]],
                     va="center", ha="right", weight="bold", linespacing=1.1)
        if pd.notna(r["npxg"]) and r["origin"] != "Penalty":
            fig.text(x_npxg, yc, f"{r['npxg']:.2f}", fontsize=10, color=color, va="center", ha="right")
        fig.add_artist(Line2D([0.05, 0.95], [y, y], color=grid, lw=0.6, alpha=0.6, transform=fig.transFigure))

    conceded_goals = g[g["side"] == "opp"]
    opp_shots = shots[(shots["match_id"] == mid) & (shots["side"] == "opp")]
    if len(conceded_goals) == 0:
        # a clean sheet: show the biggest chances against instead
        y -= 0.04
        fig.text(0.05, y, "No goals conceded. Biggest chances against:", fontsize=11, weight="bold", color=text)
        fig.text(x_flags[1], y, "Outcome", fontsize=8.5, color=muted, ha="center")
        for _, sh in opp_shots.sort_values("xg", ascending=False).head(3).iterrows():
            y -= 0.034
            fig.text(x_min, y, minute_label(sh["minute"], sh["period"]), fontsize=10, color=text_2, va="center")
            fig.text(x_player, y, _wrap_name(fig, sh["player"], x_origin - x_player - 0.012, fontsize=10),
                     fontsize=10, color=text_2, va="center", linespacing=1.15)
            fig.text(x_origin, y, sh["origin"], fontsize=10, color=text_2, va="center")
            fig.text(x_flags[1], y, sh["outcome"], fontsize=10, color=text_2, va="center", ha="center")
            fig.text(x_npxg, y, f"{sh['xg']:.2f}", fontsize=10, color=text_2, va="center", ha="right")
    else:
        counts = conceded_goals["category"].value_counts()
        npxg = opp_shots.loc[opp_shots["origin"] != "Penalty", "npxg"].sum()
        fig.text(0.5, 0.255, f"{len(conceded_goals)} goal{'s' if len(conceded_goals) != 1 else ''} conceded "
                             f"from {npxg:.2f} npxG", ha="center", fontsize=12, weight="bold", color=text)
        for i, cat in enumerate(["Restarts & turnovers", "Low-chance", "Quality gap"]):
            x = 0.2 + i * 0.3
            fig.text(x, 0.185, f"{counts.get(cat, 0)}", ha="center", fontsize=26, weight="bold",
                     color=category_colors[cat])
            fig.text(x, 0.16, cat.lower(), ha="center", va="top", fontsize=10, color=text_2)

    tab_footer(fig, "Restarts & turnovers = set piece, throw-in, penalty, own-half loss or lost goal kick "
                    "(own goals by the same rule)  ·  low-chance = npxG < 0.10  ·  quality gap = the rest")
    return fig

# Losses tab: group the Piece 5 loss causes into six fixed rows
loss_cause_groups = {
    "Misplaced pass (open play)": "Misplaced pass",
    "Misplaced pass (restart)": "Misplaced pass from a restart",
    "Miscontrol": "Miscontrol",
    "Dispossessed": "Dispossessed or failed dribble",
    "Failed dribble": "Dispossessed or failed dribble",
    "Aerial duel lost": "Aerial duel lost",
}                                                    # anything else (fouls, clearances, other) -> "Other"
loss_cause_order = ["Misplaced pass", "Misplaced pass from a restart", "Miscontrol",
                    "Dispossessed or failed dribble", "Aerial duel lost", "Other"]

def plot_losses_tab(card, losses):
    """Tab 5: where the opponent won the ball from the SEA team, how it was lost, and what it cost."""
    mid, sea_id = card["match_id"], card["team_id"]
    sea_color = team_colors[sea_id]
    l = losses[(losses["match_id"] == mid) & ~losses["ended_in_shot"]]
    own_half = l["won_x"] < 60                       # won by the opponent in the SEA team's own half

    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "Losses", sea_color)

    # left: where the opponent won the ball, SEA attacking upwards
    ax = fig.add_axes([0.03, 0.08, 0.50, top - 0.10])
    pitch = VerticalPitch(pitch_type="statsbomb", pitch_color=surface, line_color=grid, linewidth=1)
    pitch.draw(ax=ax)
    for x_line in (40, 80):
        pitch.lines(x_line, 0, x_line, 80, ax=ax, color=grid, lw=1, linestyle="--", zorder=1)
    quiet, costly = l[~l["led_to_shot"]], l[l["led_to_shot"]]
    pitch.scatter(quiet["won_x"], quiet["won_y"], s=26, ax=ax, facecolors=to_rgba(sea_color, 0.22),
                  edgecolors=to_rgba(darker(sea_color), 0.55), linewidths=0.6, zorder=3)
    pitch.scatter(costly["won_x"], costly["won_y"], s=48, ax=ax, facecolors=sea_color,
                  edgecolors=darker(sea_color), linewidths=1, zorder=4)
    for zone, (lo, hi) in {"Own third": (0, 40), "Middle third": (40, 80), "Final third": (80, 120)}.items():
        n = int(((l["won_x"] >= lo) & (l["won_x"] < hi)).sum())
        pitch.annotate(f"{n}", ((lo + hi) / 2, 83), ax=ax, ha="left", va="center", fontsize=12,
                       weight="bold", color=text, annotation_clip=False)
        pitch.annotate(zone.lower(), ((lo + hi) / 2 - 3.5, 83), ax=ax, ha="left", va="center",
                       fontsize=8.5, color=text_2, annotation_clip=False)
    ax.set_title("Where the opponent won the ball  ·  filled = opponent shot followed", fontsize=10, color=text_2)
    ax.text(0.5, -0.01, f"{card['Team']} attacking ↑", transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=sea_color)

    # right: the numbers
    oh = l[own_half]
    x0 = 0.62
    fig.text(x0, top - 0.04, f"{len(l)}", fontsize=30, weight="bold", color=text, va="top")
    fig.text(x0 + 0.10, top - 0.045, "times the ball was lost\n(possessions handed to the opponent)",
             fontsize=9.5, color=text_2, va="top", linespacing=1.3)
    fig.text(x0, top - 0.11, f"{len(oh) / max(len(l), 1):.0%}", fontsize=30, weight="bold", color=text, va="top")
    fig.text(x0 + 0.10, top - 0.115, "won by the opponent\nin their own half", fontsize=9.5, color=text_2,
             va="top", linespacing=1.3)
    n_costly = int(oh["led_to_shot"].sum())
    shots_n, npxg, goals = int(oh["opp_shots"].sum()), oh["opp_npxg"].sum(), int(oh["opp_goals"].sum())
    fig.text(x0, top - 0.18, f"{n_costly} of those were followed by a shot:\n"
                             f"{shots_n} shots · {npxg:.2f} npxG · {goals} goal{'s' if goals != 1 else ''}",
             fontsize=12, weight="bold", color=text, va="top", linespacing=1.35)
    fig.text(x0, top - 0.245, f"{int(oh['quick_shot'].sum())} of them within 15 seconds",
             fontsize=9.5, color=text_2, va="top")

    # right: how the ball was lost (the SEA team's own action), fixed rows
    causes = (l["cause"].map(loss_cause_groups).fillna("Other").value_counts()
              .reindex(loss_cause_order, fill_value=0))
    ax2 = fig.add_axes([x0, 0.12, 0.34, top - 0.30 - 0.12])
    ax2.set_facecolor(surface)
    xmax = max(causes.max(), 1) * 1.45
    for i, (name, n) in enumerate(causes.items()):
        ax2.barh(i, xmax / 1.45, color=track, height=0.42)
        ax2.barh(i, n, color=sea_color, height=0.42)
        ax2.text(0, i - 0.36, name, fontsize=10, color=text, va="bottom")
        ax2.text(n + xmax * 0.02, i, f"{n}  ({n / max(len(l), 1):.0%})", fontsize=9, color=text_2, va="center")
    ax2.set_ylim(len(causes) - 0.4, -1.0)
    ax2.set_xlim(0, xmax)
    ax2.axis("off")
    ax2.set_title("How the ball was lost", loc="left", fontsize=10.5, color=text)

    tab_footer(fig, "Location = where the opponent won the ball (their first action)  ·  possessions ending in a "
                    "SEA shot left out  ·  shot followed = in the opponent's next possession")
    return fig

def plot_regains_tab(card, regains, defending):
    """Tab 6: where the SEA team won the ball back, how high they defended, and what came next."""
    mid, sea_id, opp_id = card["match_id"], card["team_id"], card["opponent_id"]
    sea_color = team_colors[sea_id]
    r = regains[regains["match_id"] == mid]
    act, keeper = r[r["how"] == "Active"], r[r["how"] == "Keeper"]
    n_restart = int((r["how"] == "Restart").sum())
    d = defending[defending["match_id"] == mid].set_index("team_id")["defensive_distance"]
    dd_sea, dd_opp = d.get(sea_id, np.nan), d.get(opp_id, np.nan)

    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "Regains", sea_color)

    # left: where the ball was won back, SEA attacking upwards
    ax = fig.add_axes([0.03, 0.08, 0.50, top - 0.10])
    pitch = VerticalPitch(pitch_type="statsbomb", pitch_color=surface, line_color=grid, linewidth=1)
    pitch.draw(ax=ax)
    for x_line in (40, 80):
        pitch.lines(x_line, 0, x_line, 80, ax=ax, color=grid, lw=1, linestyle="--", zorder=1)
    pitch.scatter(keeper["location_x"], keeper["location_y"], s=40, ax=ax, facecolors="none",
                  edgecolors=darker(sea_color), linewidths=1.2, zorder=3)
    pitch.scatter(act["location_x"], act["location_y"], s=36, ax=ax, facecolors=to_rgba(sea_color, 0.85),
                  edgecolors=darker(sea_color), linewidths=0.8, zorder=4)
    if pd.notna(dd_sea):
        pitch.lines(dd_sea, 0, dd_sea, 80, ax=ax, color=text_2, lw=1.4, linestyle=(0, (5, 3)), zorder=5)
        pitch.annotate(f"defensive distance {dd_sea:.0f} yd", (dd_sea + 1.2, 79), ax=ax, ha="right",
                       va="bottom", fontsize=8.5, color=text_2, zorder=6)
    for zone, (lo, hi) in {"Own third": (0, 40), "Middle third": (40, 80), "Final third": (80, 120)}.items():
        n = int(((act["location_x"] >= lo) & (act["location_x"] < hi)).sum())
        pitch.annotate(f"{n}", ((lo + hi) / 2, 83), ax=ax, ha="left", va="center", fontsize=12,
                       weight="bold", color=text, annotation_clip=False)
        pitch.annotate(zone.lower(), ((lo + hi) / 2 - 3.5, 83), ax=ax, ha="left", va="center",
                       fontsize=8.5, color=text_2, annotation_clip=False)
    ax.set_title("Where the ball was won back  ·  filled = in open play, ring = by the keeper",
                 fontsize=10, color=text_2)
    ax.text(0.5, -0.01, f"{card['Team']} attacking ↑", transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=sea_color)

    # right: the numbers
    x0 = 0.62
    fig.text(x0, top - 0.04, f"{len(act)}", fontsize=30, weight="bold", color=text, va="top")
    fig.text(x0 + 0.10, top - 0.045,
             f"regains in open play\n(+ {len(keeper)} by the keeper, {n_restart} from restarts)",
             fontsize=9.5, color=text_2, va="top", linespacing=1.3)
    opp_half = (~act["own_half"]).mean() if len(act) else 0
    fig.text(x0, top - 0.11, f"{opp_half:.0%}", fontsize=30, weight="bold", color=text, va="top")
    fig.text(x0 + 0.10, top - 0.115, "of them won in the\nopponent's half", fontsize=9.5, color=text_2,
             va="top", linespacing=1.3)
    fig.text(x0, top - 0.18, f"Defensive distance: {dd_sea:.0f} yd\n{card['Opponent']}: {dd_opp:.0f} yd",
             fontsize=12, weight="bold", color=text, va="top", linespacing=1.35)
    fig.text(x0, top - 0.245, "average height of tackles, interceptions and\ndribbled-past events, from their own goal",
             fontsize=9, color=text_2, va="top", linespacing=1.3)

    # right: what came of the open-play regains
    outcomes = [("Kept the ball 10 seconds or more", act["kept_10s"].mean()),
                ("Got further forward", act["progressed"].mean()),
                ("Lost it again within 5 seconds", act["lost_quickly"].mean())]
    ax2 = fig.add_axes([x0, 0.22, 0.34, top - 0.32 - 0.22])
    ax2.set_facecolor(surface)
    for i, (name, v) in enumerate(outcomes):
        v = 0 if pd.isna(v) else v
        ax2.barh(i, 1, color=track, height=0.42)
        ax2.barh(i, v, color=sea_color, height=0.42)
        ax2.text(0, i - 0.36, name, fontsize=10, color=text, va="bottom")
        ax2.text(1.03, i, f"{v:.0%}", fontsize=10, color=text_2, va="center")
    ax2.set_ylim(len(outcomes) - 0.4, -1.0)
    ax2.set_xlim(0, 1.25)
    ax2.axis("off")
    ax2.set_title("After an open-play regain", loc="left", fontsize=10.5, color=text)
    n_sh, xg = int(act["shots"].sum()), act["npxg"].sum()
    fig.text(x0, 0.19, f"Open-play regains led to {n_sh} shot{'s' if n_sh != 1 else ''} · {xg:.2f} npxG",
             fontsize=11, weight="bold", color=text, va="top")

    tab_footer(fig, "Location = the SEA team's first action after winning the ball  ·  got further forward = "
                    "own-half regains that reached the opponent's half, others that reached the final third")
    return fig

def plot_build_up_tab(card, goal_kicks, keeper_match, pass_style_match):
    """Tab 7: goal kicks, the keeper's open-play passing, and open-play passing overall."""
    mid, sea_id = card["match_id"], card["team_id"]
    sea_color = team_colors[sea_id]
    gk = goal_kicks[goal_kicks["match_id"] == mid]
    g_sea, g_opp = gk[gk["side"] == "sea"], gk[gk["side"] == "opp"]
    ps = (pass_style_match[(pass_style_match["sea_team_id"] == sea_id)
                           & (pass_style_match["match_week"] == card["MD"])]
          .set_index("side"))
    kp = (keeper_match[(keeper_match["sea_team_id"] == sea_id)
                       & (keeper_match["match_week"] == card["MD"])]
          .set_index("side"))
    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "Build-up", sea_color)

    # left: where the SEA goal kicks went, and whether they kept the ball
    ax = fig.add_axes([0.03, 0.08, 0.48, top - 0.10])
    pitch = VerticalPitch(pitch_type="statsbomb", pitch_color=surface, line_color=grid, linewidth=1)
    pitch.draw(ax=ax)
    sx, sy = g_sea["location_x"], g_sea["location_y"]
    ex, ey = g_sea["pass_end_location"].str[0], g_sea["pass_end_location"].str[1]
    pitch.lines(sx, sy, ex, ey, ax=ax, color=to_rgba(sea_color, 0.25), lw=1.2, zorder=2)
    kept = g_sea["kept_10s"].to_numpy()
    pitch.scatter(ex[kept], ey[kept], s=46, ax=ax, facecolors=sea_color, edgecolors=darker(sea_color),
                  linewidths=0.8, zorder=4)
    pitch.scatter(ex[~kept], ey[~kept], s=46, ax=ax, facecolors=surface, edgecolors=darker(sea_color),
                  linewidths=1.2, zorder=3)
    if len(g_sea):
        land = ex.mean()
        opp_land = g_opp["pass_end_location"].str[0].mean() if len(g_opp) else np.nan
        label = f"average landing point, {land:.0f} yd"
        if pd.notna(opp_land):
            label += f"  ({card['Opponent']}: {opp_land:.0f} yd)"
        pitch.lines(land, 0, land, 80, ax=ax, color=text_2, lw=1.2, linestyle=(0, (5, 3)), zorder=3)
        pitch.annotate(label, (land + 1.2, 79), ax=ax, ha="right", va="bottom", fontsize=8.5, color=text_2)
    ax.set_title("Where their goal kicks landed  ·  filled = still had the ball 10 s later", fontsize=10, color=text_2)
    ax.text(0.5, -0.01, f"{card['Team']} attacking ↑", transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=sea_color)

    # right: SEA v opponent, in three sections
    pct = lambda x: "–" if pd.isna(x) else f"{x:.0f}%"
    num = lambda x: f"{x:.0f}"
    share = lambda d, col: d[col].mean() * 100 if len(d) else np.nan            # NaN when no goal kicks
    style = lambda side, col: ps.loc[side, col] * 100 if side in ps.index else np.nan
    keeper_n = lambda side: kp.loc[side, "keeper_passes"] if side in kp.index else 0
    keeper_rate = lambda side, col: kp.loc[side, col] * 100 if side in kp.index else np.nan
    rows = [
        dict(header="From goal kicks"),
        dict(label="Goal kicks", sea=len(g_sea), opp=len(g_opp), fmt=num, bar="share", no_bold=True),
        dict(label="Played long", sea=share(g_sea, "pass_long"), opp=share(g_opp, "pass_long"),
             fmt=pct, bar="pct", no_bold=True),
        dict(label="Won the first ball", sea=share(g_sea, "kept"), opp=share(g_opp, "kept"), fmt=pct, bar="pct", no_bold=True),
        dict(label="Still had it 10 seconds later", sea=share(g_sea, "kept_10s"), opp=share(g_opp, "kept_10s"), fmt=pct, bar="pct"),
        dict(header="Goalkeeper passes in open play"),
        dict(label="Passes", sea=keeper_n("sea"), opp=keeper_n("opp"), fmt=num, bar="share", no_bold=True),
        dict(label="Played long", sea=keeper_rate("sea", "keeper_passes_long_share"), opp=keeper_rate("opp", "keeper_passes_long_share"),
             fmt=pct, bar="pct", no_bold=True),
        dict(label="Completed", sea=keeper_rate("sea", "keeper_passes_completed_pct"), opp=keeper_rate("opp", "keeper_passes_completed_pct"),
             fmt=pct, bar="pct"),
        dict(header="All open-play passes"),
        dict(label="Played forward", sea=style("sea", "pass_forward_pct"), opp=style("opp", "pass_forward_pct"),
             fmt=pct, bar="pct", no_bold=True),
        dict(label="Played long", sea=style("sea", "pass_long_pct"), opp=style("opp", "pass_long_pct"),
             fmt=pct, bar="pct", no_bold=True),
        dict(label="Long passes completed", sea=style("sea", "pass_long_completion_pct"), opp=style("opp", "pass_long_completion_pct"),
             fmt=pct, bar="pct"),
        dict(label="Completed under pressure", sea=style("sea", "pass_under_pressure_completion_pct"),
             opp=style("opp", "pass_under_pressure_completion_pct"), fmt=pct, bar="pct"),
    ]
    ax2 = fig.add_axes([0.56, 0.22, 0.41, top - 0.04 - 0.22])
    ax2.set_facecolor(surface)
    plot_stat_comparison(ax2, rows, sea_color=sea_color, sea_label=card["Team"], opp_label=card["Opponent"])

    # what the goal kicks cost
    danger = g_sea[g_sea["opp_shot_30s"]]
    n_d, xg_d = len(danger), danger["next_shot_xg"].sum()
    goals_d = int((danger["next_shot_outcome"] == "Goal").sum())
    fig.text(0.765, 0.185, f"{n_d} of their goal kicks were followed by an\nopponent shot within 30 seconds: "
                           f"{xg_d:.2f} xG · {goals_d} goal{'s' if goals_d != 1 else ''}",
             ha="center", va="top", fontsize=11, weight="bold", color=text, linespacing=1.35)

    tab_footer(fig, "First ball = next completed pass by the kicking team  ·  10 s = still in possession 10 seconds after the kick  ·  "
                    "long = 35+ yards  ·  forward = within 30° of straight ahead  ·  open play excludes set pieces")

    return fig

# Attack tab: the three channels across the pitch, from the attacker's point of view
lane_bounds = {"left": (0, 80 / 3), "central": (80 / 3, 160 / 3), "right": (160 / 3, 80)}
lane_names = {"left": "Left", "central": "Centre", "right": "Right"}

def _channel_pitch(ax, entries, touches, color, title):
    """Middle and final thirds. Lanes shaded by share of final-third touches; arrows sized by share of entries."""
    pitch = VerticalPitch(pitch_type="statsbomb", pitch_color=surface, line_color=grid, linewidth=1)
    pitch.draw(ax=ax)
    ax.set_ylim(38, 121)                                   # show only x = 38–120 (middle + final third)
    n_ent, n_tch = max(sum(entries.values()), 1), max(sum(touches.values()), 1)
    for lane, (y0, y1) in lane_bounds.items():
        yc = (y0 + y1) / 2
        t_share, e_share = touches[lane] / n_tch, entries[lane] / n_ent
        # shading: share of touches
        pitch.polygon([[(80, y0), (120, y0), (120, y1), (80, y1)]], ax=ax,
                      color=to_rgba(color, 0.10 + 0.75 * t_share), lw=0, zorder=1)
        pitch.annotate(f"touches\n{t_share:.0%}", (110, yc), ax=ax, ha="center", va="center",
                       fontsize=9.5, color=text, linespacing=1.2, zorder=4)
        # arrow: share of entries
        pitch.arrows(57, yc, 79, yc, ax=ax, color=darker(color, 0.5), width=4 + 30 * e_share,
                     headwidth=1.8, headlength=1.4, headaxislength=1.3, zorder=3)
        pitch.annotate(f"{e_share:.0%}", (90, yc), ax=ax, ha="center", va="center", fontsize=15,
                       weight="bold", color=text, zorder=4)
        pitch.annotate(f"{entries[lane]} entries", (50, yc), ax=ax, ha="center", va="center",
                       fontsize=8.5, color=text_2, zorder=4)
    for y_line in (80 / 3, 160 / 3):                       # thin gaps between the lanes
        pitch.lines(80, y_line, 120, y_line, ax=ax, color=surface, lw=1.5, zorder=2)
    ax.set_title(title, fontsize=10.5, color=color if color != opp_color else text_2, pad=6)

def plot_attack_tab(card, entries_match, entries_channel_match, touches_channel_match):
    """Tab 8: where each side entered the final third and where they played there, plus entry rates."""
    sea_id = card["team_id"]
    sea_color = team_colors[sea_id]
    in_match = lambda t: (t["sea_team_id"] == sea_id) & (t["match_week"] == card["MD"])
    em = entries_match[in_match(entries_match)].set_index("side")
    ch = entries_channel_match[in_match(entries_channel_match)].set_index("side")
    tc = touches_channel_match[in_match(touches_channel_match)].set_index("side")
    counts = lambda tbl, side: {l: int(tbl.loc[side, l]) if side in tbl.index else 0 for l in lane_bounds}

    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "Attack", sea_color)

    # top: the two channel pitches
    h = 0.36
    ax_a = fig.add_axes([0.04, top - 0.02 - h, 0.44, h])
    ax_b = fig.add_axes([0.52, top - 0.02 - h, 0.44, h])
    _channel_pitch(ax_a, counts(ch, "sea"), counts(tc, "sea"), sea_color, f"{card['Team']} attacking ↑")
    _channel_pitch(ax_b, counts(ch, "opp"), counts(tc, "opp"), opp_color, f"{card['Opponent']} attacking ↑")
    fig.text(0.5, top - 0.02 - h - 0.008,
             "Arrows = share of final-third entries (completed passes and carries)  ·  shading = share of final-third touches",
             ha="center", va="top", fontsize=9.5, color=text_2)

    # below: SEA v opponent
    pct = lambda x: "–" if pd.isna(x) else f"{x:.0f}%"
    num = lambda x: f"{x:.0f}"
    dec = lambda x: f"{x:.1f}"
    v = lambda side, col, scale=1: em.loc[side, col] * scale if side in em.index else np.nan
    rows = [
        dict(header="Into the final third"),
        dict(label="Entries per 10 min of possession", sea=v("sea", "entries_into_final_third_per_10"),
             opp=v("opp", "entries_into_final_third_per_10"), fmt=dec, bar="share"),
        dict(label="Passes into the final third completed", sea=v("sea", "passes_into_final_third_completion_pct", 100),
             opp=v("opp", "passes_into_final_third_completion_pct", 100), fmt=pct, bar="pct"),
        dict(label="Entries made by carrying the ball", sea=v("sea", "carry_into_final_third_share", 100),
             opp=v("opp", "carry_into_final_third_share", 100), fmt=pct, bar="pct", no_bold=True),
        dict(header="Into the box"),
        dict(label="Box entries", sea=v("sea", "entries_into_box"), opp=v("opp", "entries_into_box"), fmt=num, bar="share"),
        dict(label="Box entries per 10 min of possession", sea=v("sea", "entries_into_box_per_10"), opp=v("opp", "entries_into_box_per_10"), fmt=dec, bar="share"),
        dict(label="Passes into the box completed", sea=v("sea", "passes_into_box_completion_pct", 100),
             opp=v("opp", "passes_into_box_completion_pct", 100), fmt=pct, bar="pct"),
    ]
    ax2 = fig.add_axes([0.12, 0.07, 0.76, top - 0.02 - h - 0.05 - 0.07])
    ax2.set_facecolor(surface)
    plot_stat_comparison(ax2, rows, sea_color=sea_color, sea_label=card["Team"], opp_label=card["Opponent"])

    tab_footer(fig, "Open play only  ·  lanes from each attacker's point of view: an opponent's left is the SEA team's right")
    return fig

funnel_rows = [("reached_opp_half", "Reached the opponent's half"),
               ("reached_final_third", "Reached the final third"),
               ("reached_box", "Reached the box"),
               ("shot", "Ended with a shot")]
end_type_order = ["Shot", "Won a set piece", "Kept (other)", "Lost, opponent's restart", "Lost in play", "End of half"]
end_type_colors = {"Shot": "#6250d6", "Won a set piece": "#128a62", "Kept (other)": "#3cbf8f",
                   "Lost, opponent's restart": "#e8875a", "Lost in play": "#c2461b", "End of half": "#d9d8d2"}
end_type_dark = {"Shot", "Won a set piece", "Lost in play"}     # segments that get a white label

def _section_title(fig, y, title, x0=0.08, x1=0.92):
    """Small uppercase section label with a thin rule under it, in figure coordinates."""
    fig.text(x0, y, title.upper(), ha="left", va="bottom", fontsize=8.5, weight="bold", color=muted)
    fig.add_artist(Line2D([x0, x1], [y - 0.005, y - 0.005], color=grid, lw=0.8, transform=fig.transFigure))


def plot_possession_funnel(ax, sea, opp, sea_color, opp_color=opp_color):
    """Dumbbell rows: share of each side's possessions that got this far (sea / opp are dicts of fractions)."""
    n = len(funnel_rows)
    ax.set_xlim(-46, 104)
    ax.set_ylim(n - 0.3, -0.7)
    ax.axis("off")
    for i, (col, label) in enumerate(funnel_rows):
        s, o = sea[col] * 100, opp[col] * 100
        ax.text(-45, i, label, ha="left", va="center", fontsize=10.5, color=text_2)
        ax.plot([0, 100], [i, i], color=track, lw=6, solid_capstyle="round", zorder=1)
        ax.plot([min(s, o), max(s, o)], [i, i], color="#c9c8c1", lw=2.5, zorder=2)
        ax.scatter([o], [i], s=150, color=opp_color, edgecolor=surface, linewidth=2, zorder=3)
        ax.scatter([s], [i], s=150, color=sea_color, edgecolor=surface, linewidth=2, zorder=4)
        lo_is_sea = s <= o
        ax.text(min(s, o) - 2.5, i, f"{(s if lo_is_sea else o):.0f}%", ha="right", va="center", fontsize=10.5,
                color=text, weight="bold" if not lo_is_sea else "normal")
        ax.text(max(s, o) + 2.5, i, f"{(o if lo_is_sea else s):.0f}%", ha="left", va="center", fontsize=10.5,
                color=text, weight="bold" if lo_is_sea else "normal")
    for x in (0, 50, 100):
        ax.text(x, n - 0.45, f"{x}%", ha="center", va="center", fontsize=8, color=muted)
    return ax


def plot_end_type_bars(ax, sea_shares, opp_shares, sea_label, opp_label, sea_color, opp_color=opp_color):
    """One 100% bar per side, split by how the possessions ended, with a legend underneath."""
    ax.set_xlim(0, 1)
    ax.set_ylim(3.2, -0.1)
    ax.axis("off")
    for y, shares, label, color in [(0.55, sea_shares, sea_label, sea_color), (1.65, opp_shares, opp_label, opp_color)]:
        ax.scatter([0.006], [y - 0.42], s=40, color=color, zorder=3)
        ax.text(0.018, y - 0.42, label, ha="left", va="center", fontsize=10, color=text)
        left = 0
        for end in end_type_order:
            w = shares.get(end, 0)
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=0.42, color=end_type_colors[end], edgecolor=surface, linewidth=2)
            if w >= 0.06:
                ax.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=9.5,
                        color=surface if end in end_type_dark else text, weight="bold")
            left += w
    for k, end in enumerate(end_type_order):                       # legend, two rows of three
        x, y = (k % 3) * 0.34, 2.45 + (k // 3) * 0.42
        ax.scatter([x + 0.008], [y], s=55, marker="s", color=end_type_colors[end])
        ax.text(x + 0.022, y, end, ha="left", va="center", fontsize=9, color=text_2)
    return ax

def plot_possessions_tab(card, possessions, team_stats):
    """Tab 9: how far possessions got, how they ended, where they started and how long they lasted."""
    sea_id = card["team_id"]
    sea_color = team_colors[sea_id]
    p = possessions[possessions["match_id"] == card["match_id"]]
    p_sea, p_opp = p[p["side"] == "sea"], p[p["side"] == "opp"]
    st = match_rows(team_stats, card)

    unknown = sorted(set(p["end_type"].dropna()) - set(end_type_order))
    if unknown:
        print(f"plot_possessions_tab: end types not in end_type_order (left out of the bars): {unknown}")

    funnel = lambda d: {col: d[col].mean() for col, _ in funnel_rows}
    ends = lambda d: d["end_type"].value_counts(normalize=True).to_dict()

    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "Possessions", sea_color)

    # 1. how far they got
    _section_title(fig, top - 0.01, "How far possessions got")
    h1 = 0.17
    ax1 = fig.add_axes([0.08, top - 0.025 - h1, 0.84, h1])
    plot_possession_funnel(ax1, funnel(p_sea), funnel(p_opp), sea_color)
    own = p_sea[p_sea["start_zone"] == "Own third"]
    fig.text(0.5, top - 0.03 - h1, f"{card['Team']}: {len(own)} possessions started in their own third. "
             f"{int(own['reached_final_third'].sum())} reached the final third, {int(own['shot'].sum())} ended with a shot",
             ha="center", va="top", fontsize=9.5, color=text_2)

    # 2. how they ended
    y2 = top - 0.07 - h1
    _section_title(fig, y2, "How possessions ended")
    h2 = 0.125
    ax2 = fig.add_axes([0.08, y2 - 0.012 - h2, 0.84, h2])
    plot_end_type_bars(ax2, ends(p_sea), ends(p_opp), card["Team"], card["Opponent"], sea_color)

    # 3. share of the ball, where they started and how long they lasted
    y3 = y2 - 0.02 - h2
    pct = lambda x: f"{x:.0f}%"
    num = lambda x: f"{x:.0f}"
    rows = [
        dict(header="Share of the ball"),
        dict(label="Possession", sea=st.loc["sea", "possession_pct"] * 100, opp=st.loc["opp", "possession_pct"] * 100,
             fmt=pct, bar="share"),
        dict(label="Possessions", sea=len(p_sea), opp=len(p_opp), fmt=num, bar="share", no_bold=True),
        dict(header="Where they started"),
        dict(label="Started in their own third", sea=(p_sea["start_zone"] == "Own third").mean() * 100,
             opp=(p_opp["start_zone"] == "Own third").mean() * 100, fmt=pct, bar="pct", no_bold=True),
        dict(header="How long they lasted"),
        dict(label="Median length", sea=p_sea["duration_secs"].median(), opp=p_opp["duration_secs"].median(),
             fmt=lambda x: f"{x:.0f}s", bar=None, no_bold=True),
        dict(label="Lasted under 10 seconds", sea=(p_sea["duration_secs"] < 10).mean() * 100,
             opp=(p_opp["duration_secs"] < 10).mean() * 100, fmt=pct, bar="pct", no_bold=True),
        dict(label="Median completed passes", sea=p_sea["passes_completed"].median(),
             opp=p_opp["passes_completed"].median(), fmt=lambda x: f"{x:g}", bar=None, no_bold=True),
        dict(label="3 or more completed passes", sea=(p_sea["passes_completed"] >= 3).mean() * 100,
             opp=(p_opp["passes_completed"] >= 3).mean() * 100, fmt=pct, bar="pct", no_bold=True),
    ]
    ax3 = fig.add_axes([0.12, 0.05, 0.76, y3 - 0.06])
    ax3.set_facecolor(surface)
    plot_stat_comparison(ax3, rows, sea_color=sea_color, sea_label=card["Team"], opp_label=card["Opponent"])
    tab_footer(fig, "StatsBomb possession sequences  ·  reached = an on-ball action there by the team in possession  ·  "
                    "length = first to last in-play event")
    return fig

state_order = ["leading", "level", "trailing by 1", "trailing by 2", "trailing by 3+"]
before_color, after_color = "#b5b4ae", "#52514e"

def _dot_legend(fig, x_right, y, items, fontsize=8):
    """Right-aligned row of coloured dots with labels, in figure coordinates. items = [(label, color), ...]"""
    x = x_right
    for label, color in reversed(items):
        fig.text(x, y, label, ha="right", va="center", fontsize=fontsize, color=text_2)
        x -= _text_width(fig, label, fontsize=fontsize) + 0.008
        fig.add_artist(Line2D([x], [y], marker="o", markersize=5.5, color=color, lw=0, transform=fig.transFigure))
        x -= 0.03


def plot_state_strip(ax, spells, goals, match_len, half_time, sea_color, opp_color=opp_color):
    """The match as a strip of score states, with a dot above it for every goal."""
    for _, sp in spells.iterrows():
        color = to_rgba(sea_color, 0.35) if sp["game_state"] == "leading" else state_colors[sp["game_state"]]
        ax.axvspan(sp["start"], sp["end"], ymin=0, ymax=0.45, color=color, lw=0)
        if sp["end"] - sp["start"] > 7:
            ax.text((sp["start"] + sp["end"]) / 2, 0.225, state_labels[sp["game_state"]], ha="center", va="center",
                    fontsize=8.5, color=text)
    for _, g in goals.iterrows():
        ax.scatter([g["goal_min"]], [0.68], s=34, color=opp_color if g["goal_against"] else sea_color,
                   edgecolor=surface, linewidth=1, zorder=3)
    ax.axvline(half_time, ymin=0, ymax=0.45, color=surface, lw=2)
    for x, lab in [(0, "0'"), (half_time, "HT"), (match_len, "FT")]:
        ax.text(x, -0.2, lab, ha="center", va="top", fontsize=8, color=muted)
    ax.set_xlim(0, match_len)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def plot_state_table(ax, states, sea_color, match_len, opp_color=opp_color, min_minutes=15, min_rows=3):
    """One row per score state: minutes, possession, npxG per 15 for / against, shots, goals, PPDA.

    states: list of dicts with game_state, minutes, possession, sea_npxg_p15, opp_npxg_p15, sea_shots, opp_shots,
            goals_for, goals_against, sea_ppda, opp_ppda
    """
    n = len(states)
    ax.set_xlim(0, 100)
    ax.set_ylim(max(n, min_rows) - 0.45, -1.05)          # same row height whatever the number of states
    ax.axis("off")
    head = dict(fontsize=8, color=muted, weight="bold", va="center")
    for x, lab, ha in [(0, "STATE", "left"), (13.5, "MINUTES", "left"), (34, "POSSESSION", "left"),
                       (61.5, "NPXG PER 15 MIN", "center"), (80, "SHOTS", "center"), (87, "GOALS", "center"),
                       (95, "PPDA", "center")]:
        ax.text(x, -0.72, lab, ha=ha, **head)
    ax.plot([0, 100], [-0.45, -0.45], color=grid, lw=0.8)

    rates = [v for s in states if s["minutes"] >= min_minutes for v in (s["sea_npxg_p15"], s["opp_npxg_p15"]) if pd.notna(v)]
    scale = 7 / max(max(rates, default=0), 0.05)           # the longest npxG bar is 7 units
    ppda = lambda v: "–" if pd.isna(v) else f"{v:.1f}"
    for i, s in enumerate(states):
        chip = to_rgba(sea_color, 0.35) if s["game_state"] == "leading" else state_colors[s["game_state"]]
        ax.barh(i, 11, left=0, height=0.5, color=chip)
        ax.text(5.5, i, state_labels[s["game_state"]], ha="center", va="center", fontsize=9, color=text)

        ax.barh(i, 11, left=13.5, height=0.16, color=track)
        ax.barh(i, 11 * s["minutes"] / match_len, left=13.5, height=0.16, color=text_2)
        ax.text(25.8, i, f"{s['minutes']:.0f} min", ha="left", va="center", fontsize=10, color=text)

        ax.text(34, i, f"{s['possession']:.0%}", ha="left", va="center", fontsize=10, color=text)
        ax.barh(i, 8, left=38.5, height=0.16, color=opp_color)
        ax.barh(i, 8 * s["possession"], left=38.5, height=0.16, color=sea_color)

        if s["minutes"] >= min_minutes:
            ws, wo = s["sea_npxg_p15"] * scale, s["opp_npxg_p15"] * scale
            ax.barh(i, -ws, left=61.2, height=0.3, color=sea_color)
            ax.barh(i, wo, left=61.8, height=0.3, color=opp_color)
            ax.text(61.2 - ws - 0.8, i, f"{s['sea_npxg_p15']:.2f}", ha="right", va="center", fontsize=10, color=text)
            ax.text(61.8 + wo + 0.8, i, f"{s['opp_npxg_p15']:.2f}", ha="left", va="center", fontsize=10, color=text)
            ax.text(95, i, f"{ppda(s['sea_ppda'])} · {ppda(s['opp_ppda'])}", ha="center", va="center",
                    fontsize=10, color=text)
        else:
            ax.text(61.5, i, f"under {min_minutes} min", ha="center", va="center", fontsize=9, color=muted, style="italic")
            ax.text(95, i, "–", ha="center", va="center", fontsize=10, color=muted)
        ax.text(80, i, f"{s['sea_shots']:.0f}–{s['opp_shots']:.0f}", ha="center", va="center", fontsize=10, color=text)
        ax.text(87, i, f"{s['goals_for']:.0f}–{s['goals_against']:.0f}", ha="center", va="center", fontsize=10, color=text)
    return ax

def plot_game_state_tab(card, state_summary_match, ppda_state, goal_windows, state_intervals, match_end, half_time):
    """Tab 10: how the SEA team played in each score state, and what changed around the goals."""
    sea_id = card["team_id"]
    sea_color = team_colors[sea_id]
    mid = card["match_id"]
    me, ht = match_end[mid], half_time[mid]

    ss = state_summary_match[state_summary_match["match_id"] == mid].set_index("game_state")
    pp = ppda_state[ppda_state["match_id"] == mid].set_index(["game_state", "side"])["ppda"]
    gw = goal_windows[goal_windows["match_id"] == mid]
    goals_in = gw.groupby(["state_before", "goal_against"]).size()
    states = [dict(game_state=g, minutes=ss.loc[g, "minutes"], possession=ss.loc[g, "sea_possession_pct"],
                   sea_npxg_p15=ss.loc[g, "sea_npxg_p15"], opp_npxg_p15=ss.loc[g, "opp_npxg_p15"],
                   sea_shots=ss.loc[g, "sea_shots"], opp_shots=ss.loc[g, "opp_shots"],
                   goals_for=goals_in.get((g, False), 0), goals_against=goals_in.get((g, True), 0),
                   sea_ppda=pp.get((g, "sea"), np.nan), opp_ppda=pp.get((g, "opp"), np.nan))
              for g in state_order if g in ss.index and ss.loc[g, "minutes"] > 0]

    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "Game state", sea_color)

    # 1. the match as a strip of score states, then one row per state
    _section_title(fig, top - 0.01, f"Each score state for {card['Team']}")
    fig.text(0.92, top - 0.01, f"pairs: {card['Team']} first, then {card['Opponent']}", ha="right", va="bottom",
             fontsize=8.5, color=muted)
    ax_s = fig.add_axes([0.08, top - 0.07, 0.84, 0.04])
    plot_state_strip(ax_s, merge_state_spells(state_intervals[state_intervals["match_id"] == mid]), gw, me, ht, sea_color)
    _dot_legend(fig, 0.92, top - 0.026, [(f"goal for {card['Team']}", sea_color),
                                         (f"goal for {card['Opponent']}", opp_color)])
    h1 = 0.05 + 0.045 * max(len(states), 3)
    ax1 = fig.add_axes([0.08, top - 0.10 - h1, 0.84, h1])
    plot_state_table(ax1, states, sea_color, me)

    # 2. 10 minutes either side of the goals conceded (or scored, if none were conceded)
    conceded = gw[gw["goal_against"]]
    w, what = (conceded, "conceded") if len(conceded) else (gw[~gw["goal_against"]], "scored")
    goal_word = "goal" if len(w) == 1 else "goals"
    y2 = top - 0.14 - h1
    _section_title(fig, y2, f"10 minutes before and after the {goal_word} {what}")
    fig.text(0.92, y2, f"{len(w)} {goal_word}  ·  windows cut at the previous or next goal and at half-time",
             ha="right", va="bottom", fontsize=8.5, color=muted)

    tot = w.sum(numeric_only=True)
    per10 = lambda col, win: tot[f"{col}_{win}"] / tot[f"{win}_min"] * 10 if tot[f"{win}_min"] > 0 else np.nan
    poss_min = lambda win: tot[f"sea_possession_min_{win}"] + tot[f"opp_possession_min_{win}"]
    poss = lambda win: tot[f"sea_possession_min_{win}"] / poss_min(win) * 100 if poss_min(win) > 0 else np.nan
    one = lambda x: "–" if pd.isna(x) else f"{x:.1f}"
    two = lambda x: "–" if pd.isna(x) else f"{x:.2f}"
    pct = lambda x: "–" if pd.isna(x) else f"{x:.0f}%"
    rows = [
        dict(label="Minutes covered", sea=tot["before_min"], opp=tot["after_min"], fmt=lambda x: f"{x:.0f}",
             bar=None, no_bold=True),
        dict(label=f"{card['Opponent']} shots per 10 min", sea=per10("opp_shots", "before"),
             opp=per10("opp_shots", "after"), fmt=one, bar="share", no_bold=True),
        dict(label=f"{card['Opponent']} npxG per 10 min", sea=per10("opp_npxg", "before"),
             opp=per10("opp_npxg", "after"), fmt=two, bar="share", no_bold=True),
        dict(label=f"{card['Team']} shots per 10 min", sea=per10("sea_shots", "before"),
             opp=per10("sea_shots", "after"), fmt=one, bar="share", no_bold=True),
        dict(label=f"{card['Team']} npxG per 10 min", sea=per10("sea_npxg", "before"),
             opp=per10("sea_npxg", "after"), fmt=two, bar="share", no_bold=True),
        dict(label=f"{card['Team']} possession", sea=poss("before"), opp=poss("after"), fmt=pct, bar="pct",
             no_bold=True),
    ]
    h2 = min(y2 - 0.09, 0.36)
    ax2 = fig.add_axes([0.12, y2 - 0.025 - h2, 0.76, h2])
    ax2.set_facecolor(surface)
    plot_stat_comparison(ax2, rows, sea_color=before_color, opp_color=after_color,
                         sea_label="10 min before", opp_label="10 min after")
    tab_footer(fig, "npxG = non-penalty xG  ·  per-15 rates and PPDA only for states lasting 15+ minutes  ·  "
                    "goal windows pooled across the goals")
    return fig

tab_blurbs = {
    "Stats": "The headline numbers, side by side",
    "xG race": "How the chances added up over the match, with the score state",
    "Shots": "Where the shots came from, and the goals by origin",
    "Goals": "Every goal conceded: how it came about and what kind of goal it was",
    "Losses": "Where the ball was lost, why, and what followed",
    "Regains": "Where the ball was won back, and how high each team defended",
    "Build-up": "Goal kicks, goalkeeper passes and passing style",
    "Attack": "Entries into the final third and the box, by lane",
    "Possessions": "How far possessions got, how they ended and how long they lasted",
    "Game state": "How the team played in each score state, and around the goals",
}
home_card_fields = ["Table at kick-off", "Opponent's other results", "Avg margin elsewhere", "3rd goal conceded"]

def save_tab(fig, card, tab_label):
    i = tab_names.index(tab_label)
    return save_figure(fig, match_folder(card), f"{i:02d}_{tabs[i][0]}.png")

def plot_home_tab(card, notes=None):
    """Tab 0: the match card and a clickable list of the other tabs. notes = optional list of key points."""
    sea_color = team_colors[card["team_id"]]
    fig = plt.figure(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(surface)
    top = draw_match_header(fig, card["Team"], card["FT"], card["Opponent"], card["context_line"],
                            card["stakes_line"], card["scorers_line"], "Home", sea_color)
    y = top - 0.01

    if notes:
        _section_title(fig, y, "Key points")
        y -= 0.035
        for note in notes:
            lines = textwrap.wrap(note, 105)
            fig.text(0.08, y, "•", ha="left", va="top", fontsize=11, color=sea_color)
            fig.text(0.10, y, "\n".join(lines), ha="left", va="top", fontsize=11, color=text, linespacing=1.4)
            y -= 0.022 * len(lines) + 0.012
        y -= 0.02

    rows = [(f, card.get(f)) for f in home_card_fields if pd.notna(card.get(f)) and str(card.get(f)) not in ("", "–")]
    if rows:
        _section_title(fig, y, "Match card")
        y -= 0.035
        for field, value in rows:
            lines = textwrap.wrap(str(value), 80)
            fig.text(0.08, y, field, ha="left", va="top", fontsize=10.5, color=muted)
            fig.text(0.32, y, "\n".join(lines), ha="left", va="top", fontsize=10.5, color=text, linespacing=1.4)
            y -= 0.021 * len(lines) + 0.012
        y -= 0.02

    _section_title(fig, y, "In this report")
    fig.text(0.92, y, "click a tab name, here or in the bar above, to jump to it", ha="right", va="bottom",
             fontsize=8.5, color=muted)
    y -= 0.04
    for i, name in enumerate(tab_names[1:], start=1):
        fig.text(0.08, y, f"{i:02d}", ha="left", va="center", fontsize=10, color=muted)
        fig.text(0.13, y, name, ha="left", va="center", fontsize=12.5, weight="bold", color=sea_color).set_gid(f"tab:{name}")
        fig.text(0.33, y, tab_blurbs[name], ha="left", va="center", fontsize=10.5, color=text_2).set_gid(f"tab:{name}")
        fig.add_artist(Line2D([0.08, 0.92], [y - 0.022, y - 0.022], color=grid, lw=0.6, transform=fig.transFigure))
        y -= 0.046
    tab_footer(fig, "Match report  ·  one page per tab")
    return fig

def _tab_links(fig, pad=3):
    """Click areas of every text tagged 'tab:<name>' on a figure, as (target page, rect in PDF points)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    s = 72 / fig.dpi                                        # screen pixels -> PDF points
    links = []
    for t in fig.findobj(lambda a: isinstance(a, Text) and str(a.get_gid() or "").startswith("tab:")):
        bb = t.get_window_extent(renderer)
        links.append((tab_names.index(t.get_gid()[4:]),
                      (bb.x0 * s - pad, bb.y0 * s - pad, bb.x1 * s + pad, bb.y1 * s + pad)))
    return links

def _keep_zoom(writer, target):
    """Destination: top of the target page, keeping the reader's current zoom (/XYZ with a null zoom)."""
    page = writer.pages[target]
    return ArrayObject([page.indirect_reference, NameObject("/XYZ"), NullObject(),
                        FloatObject(float(page.mediabox.top)), NullObject()])

def save_match_pdf(card, tab_makers):
    """Every tab of one match in one PDF: one page per tab, clickable tab bar and Home list, a bookmark per tab."""
    buf = io.BytesIO()
    page_links = []
    with PdfPages(buf) as pdf:
        for name in tab_names:
            fig = tab_makers[name](card)
            page_links.append(_tab_links(fig))
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
    buf.seek(0)
    writer = PdfWriter(clone_from=PdfReader(buf))
    for page, links in enumerate(page_links):
        for target, rect in links:
            if target != page:
                link = writer.add_annotation(page_number=page, annotation=Link(rect=rect, target_page_index=target))
                # point the link at the page itself (pypdf writes a bare page number, which some viewers ignore)
                link[NameObject("/Dest")] = _keep_zoom(writer, target)            # was [..., NameObject("/Fit")]
    for i, name in enumerate(tab_names):
        writer.add_outline_item(name, i, fit=Fit.xyz(top=float(writer.pages[i].mediabox.top)))   # was add_outline_item(name, i)
    folder = match_folder(card)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{folder_name(card['Team'])}_{folder.name}.pdf"
    with open(path, "wb") as f:
        writer.write(f)
    return path