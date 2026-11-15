<p align="right"><img alt="Hudl StatsBomb logo" src="https://raw.githubusercontent.com/statsbomb/logos/main/HudlStatsbomb_Python.svg" height="28"></p>

# SEA-Teams-FWWC

How did South East Asian teams perform in FIFA Women's World Cups, 2019 and 2023 editions?

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data).

The key question: **How could heavy defeats have been avoided?**

Each goal conceded is split into what the team had a hand in (restarts and turnovers), low-quality chances, and the gap in quality, and the findings end with lessons for the Philippines ahead of the 2027 World Cup.

## Key findings

- **Most goals came from restarts and turnovers, but they made defeats heavier rather than deciding them.** Among 40 goals, 22 came from set pieces, throw-ins, penalties or losing the ball in their own half, but 12 of those came when the deficits were already 3 or more. The opening goals came mostly from the gap in quality.
- **Losing the ball in their own half was the costliest habit,** behind 12 goals. Thailand and Vietnam did it around 30 times per match, while the Philippines about a third as often.
- **Luck inflated the margins, not the results.** In the 3 heaviest defeats, far more went in than the chances suggested.
- **The shared weakness was in SEA teams' attack.** Each team only had 10 to 13 shots in their own 3 matches. Even though the Philippines reached the final third more than either Thailand or Vietnam, they turned these entries into shots the least often.

The full cross-team findings, including the lessons for the Philippines, are in [findings/all_teams.md](findings/all_teams.md).

## Goals conceded, match by match

| Team | Match | Restarts & turnovers | Low-chance | Quality gap |
| --- | --- | --- | --- | --- |
| Thailand | 0–13 United States | 6 | 3 | 4 |
| Thailand | 1–5 Sweden | 4 | 1 | 0 |
| Thailand | 0–2 Chile | 1 | 0 | 1 |
| Vietnam | 0–3 United States | 1 | 0 | 2 |
| Vietnam | 0–2 Portugal | 2 | 0 | 0 |
| Vietnam | 0–7 Netherlands | 4 | 1 | 2 |
| Philippines | 0–2 Switzerland | 1 | 0 | 1 |
| Philippines | 1–0 New Zealand | 0 | 0 | 0 |
| Philippines | 0–6 Norway | 3 | 2 | 1 |
| **Total** | | **22** | **7** | **11** |

- **Restarts & turnovers:** set piece or throw-in, penalty, after losing the ball in their own half, or within 30 seconds of losing their own goal kick. Own goals are judged by how the move started.
- **Low-chance:** non-penalty xG below 0.10.
- **Quality gap:** everything else.

## Match reports

Each match has a ten-page PDF report (stats, xG race, shots, goals, losses, regains, build-up, attack, possessions and game state) and a written analysis. The tab bar in each PDF is clickable when opened in a PDF reader.

| Team | Match | Report | Analysis |
| --- | --- | --- | --- |
| Thailand | 0–13 United States | [PDF](plots/thailand/md1_united_states/thailand_md1_united_states.pdf) | [md](findings/thailand/md1_united_states.md) |
| Thailand | 1–5 Sweden | [PDF](plots/thailand/md2_sweden/thailand_md2_sweden.pdf) | [md](findings/thailand/md2_sweden.md) |
| Thailand | 0–2 Chile | [PDF](plots/thailand/md3_chile/thailand_md3_chile.pdf) | [md](findings/thailand/md3_chile.md) |
| Vietnam | 0–3 United States | [PDF](plots/vietnam/md1_united_states/vietnam_md1_united_states.pdf) | [md](findings/vietnam/md1_united_states.md) |
| Vietnam | 0–2 Portugal | [PDF](plots/vietnam/md2_portugal/vietnam_md2_portugal.pdf) | [md](findings/vietnam/md2_portugal.md) |
| Vietnam | 0–7 Netherlands | [PDF](plots/vietnam/md3_netherlands/vietnam_md3_netherlands.pdf) | [md](findings/vietnam/md3_netherlands.md) |
| Philippines | 0–2 Switzerland | [PDF](plots/philippines/md1_switzerland/philippines_md1_switzerland.pdf) | [md](findings/philippines/md1_switzerland.md) |
| Philippines | 1–0 New Zealand | [PDF](plots/philippines/md2_new_zealand/philippines_md2_new_zealand.pdf) | [md](findings/philippines/md2_new_zealand.md) |
| Philippines | 0–6 Norway | [PDF](plots/philippines/md3_norway/philippines_md3_norway.pdf) | [md](findings/philippines/md3_norway.md) |

Team summaries: [Thailand](findings/thailand/all_games.md) · [Vietnam](findings/vietnam/all_games.md) · [Philippines](findings/philippines/all_games.md) · [All three teams](findings/all_teams.md)

## Repository structure

```
SEA-Teams-FWWC/
├── README.md
├── code/
│   ├── functions.py           tables, analysis and plotting functions
│   └── sea_teams_fwwc.ipynb   the full pipeline, from loading data to saving every tab and PDF
├── findings/
│   ├── all_teams.md           the cross-team findings and lessons for the Philippines
│   └── <team>/
│       ├── all_games.md       the team's tournament
│       └── mdN_<opponent>.md  one analysis per match
└── plots/
    └── <team>/mdN_<opponent>/ the ten tabs as PNGs, plus the match PDF
```

The data is downloaded from StatsBomb open data through `statsbombpy` when the notebook runs, so no data files are stored in the repo. Running the notebook rebuilds every tab in `plots/` and every match PDF.
