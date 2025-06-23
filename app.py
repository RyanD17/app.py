from shiny import App, ui, render, reactive, run_app
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.lines as mlines
import pandas as pd
import numpy as np

def draw_nba_court(ax=None, color='black', lw=2):
    if ax is None:
        ax = plt.gca()
    hoop = Circle((0, 0), radius=0.75, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-3, -0.75), 6, -0.1, linewidth=lw, color=color)
    outer_box = Rectangle((-8, -0.75), 16, 19, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-6, -0.75), 12, 19, linewidth=lw, color=color, fill=False)
    top_free_throw = Arc((0, 19), 12, 12, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 19), 12, 12, theta1=180, theta2=0, linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((0, 0), 8, 8, theta1=0, theta2=180, linewidth=lw, color=color)
    corner_three_a = plt.Line2D([-22, -22], [-0.75, 14], linewidth=lw, color=color)
    corner_three_b = plt.Line2D([22, 22], [-0.75, 14], linewidth=lw, color=color)
    three_arc = Arc((0, 0), 23.75*2, 23.75*2, theta1=22, theta2=158, linewidth=lw, color=color)
    for element in [hoop, backboard, outer_box, inner_box, top_free_throw, bottom_free_throw, restricted, three_arc]:
        ax.add_patch(element)
    ax.add_line(corner_three_a)
    ax.add_line(corner_three_b)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-5, 47)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax

def classify_shot_type(x, y):
    distance = np.sqrt(x**2 + y**2)
    if abs(x) >= 22 - 1e-6 and y <= 14:
        return '3PT'
    if distance >= 23.75 - 1e-6:
        return '3PT'
    return '2PT'

def calculate_shot_summary(player_data):
    player_data = player_data.copy()
    player_data['SHOT_TYPE'] = player_data.apply(
        lambda row: classify_shot_type(row['COURT_X'], row['COURT_Y']), axis=1
    )
    attempts_3pt = player_data[player_data['SHOT_TYPE'] == '3PT'].shape[0]
    makes_3pt = player_data[(player_data['SHOT_TYPE'] == '3PT') & (player_data['SHOT_MADE_FLAG'] == 1)].shape[0]
    attempts_2pt = player_data[player_data['SHOT_TYPE'] == '2PT'].shape[0]
    makes_2pt = player_data[(player_data['SHOT_TYPE'] == '2PT') & (player_data['SHOT_MADE_FLAG'] == 1)].shape[0]
    accuracy_3pt = round((makes_3pt / attempts_3pt * 100), 1) if attempts_3pt > 0 else 0.0
    accuracy_2pt = round((makes_2pt / attempts_2pt * 100), 1) if attempts_2pt > 0 else 0.0
    return {
        'Player': player_data['PLAYER'].iloc[0],
        'Team': player_data['TEAM'].iloc[0],
        '3PA': attempts_3pt,
        '3P%': accuracy_3pt,
        '2PA': attempts_2pt,
        '2P%': accuracy_2pt
    }

# --- Load and Prepare Data ---
data = pd.read_csv("NBA_2025_Shots.csv")
data.columns = [col.strip().upper().replace(" ", "_") for col in data.columns]
rename_map = {
    "TEAM_NAME": "TEAM",
    "PLAYER_NAME": "PLAYER",
    "LOC_X": "COURT_X",
    "LOC_Y": "COURT_Y",
    "SHOT_MADE": "SHOT_MADE_FLAG"
}
data = data.rename(columns=rename_map)
required_cols = ["TEAM", "PLAYER", "COURT_X", "COURT_Y", "SHOT_MADE_FLAG"]
missing = [col for col in required_cols if col not in data.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
player_choices = sorted(data['PLAYER'].unique().tolist())

# --- UI ---
app_ui = ui.page_fluid(
    ui.panel_title("NBA Shot Chart: Compare Two Players"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Primary Player"),
            ui.input_selectize("player1", "Select or Type Player 1:", choices=player_choices, selected=None),
            ui.hr(),
            ui.h4("Secondary Player"),
            ui.input_selectize("player2", "Select or Type Player 2:", choices=player_choices, selected=None),
            ui.hr(),
            ui.h4("Shot Filters"),
            ui.input_radio_buttons(
                "shot_filter",
                "Show shots:",
                choices={
                    "all": "All shots",
                    "made": "Made shots only",
                    "missed": "Missed shots only"
                },
                selected="all"
            ),
        ),
        ui.output_plot("shot_plot"),
        ui.output_table("shot_summary")
    )
)

def server(input, output, session):
    def get_player_shots(player):
        player_data = data[data["PLAYER"] == player]
        if input.shot_filter() == "made":
            player_data = player_data[player_data["SHOT_MADE_FLAG"] == 1]
        elif input.shot_filter() == "missed":
            player_data = player_data[player_data["SHOT_MADE_FLAG"] == 0]
        return player_data

    @output
    @render.plot
    def shot_plot():
        fig, ax = plt.subplots(figsize=(12, 11))
        draw_nba_court(ax)
        players = []
        colors = ['blue', 'red']
        seen_players = set()
        for idx, player_input in enumerate([input.player1(), input.player2()]):
            if player_input and player_input not in seen_players:
                df = get_player_shots(player_input)
                if not df.empty:
                    color = colors[idx]
                    ax.scatter(df["COURT_X"], df["COURT_Y"], s=120, color=color, edgecolor='black', alpha=0.8, label=player_input)
                    players.append((player_input, df["TEAM"].iloc[0], color))
                    seen_players.add(player_input)
        # Legend: just colored markers, no logos
        handles = []
        legend_labels = []
        for player_name, team, color in players:
            fallback = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                     markersize=10, label=f"{player_name} ({team})")
            handles.append(fallback)
            legend_labels.append(f"{player_name} ({team})")
        if handles:
            ax.legend(handles=handles, labels=legend_labels,
                      loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=2)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-5, 47)
        ax.set_title("NBA Shot Chart: Two Players", fontsize=16, pad=20)
        ax.text(0.5, -0.08, "Origin (0,0) at hoop | Circles=Made, X=Missed",
                ha='center', fontsize=10, style='italic', transform=ax.transAxes)
        return fig

    @output
    @render.table
    def shot_summary():
        summaries = []
        seen_players = set()
        for player in [input.player1(), input.player2()]:
            if player and player not in seen_players:
                df = get_player_shots(player)
                if not df.empty:
                    summaries.append(calculate_shot_summary(df))
                    seen_players.add(player)
        if not summaries:
            return None
        summary_df = pd.DataFrame(summaries)
        return summary_df[['Player', 'Team', '3PA', '3P%', '2PA', '2P%']]

app = App(app_ui, server)
run_app(app)
