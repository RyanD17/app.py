from shiny import App, ui, render, reactive, run_app
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase
import pandas as pd
import numpy as np
import requests
from io import BytesIO

# --- NBA Team Logos (official NBA CDN PNGs, as of 2024-25) ---
TEAM_LOGO_URLS = {
    "Atlanta Hawks":        "https://cdn.nba.com/logos/nba/1610612737/primary/L/logo.png",
    "Boston Celtics":       "https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.png",
    "Brooklyn Nets":        "https://cdn.nba.com/logos/nba/1610612751/primary/L/logo.png",
    "Charlotte Hornets":    "https://cdn.nba.com/logos/nba/1610612766/primary/L/logo.png",
    "Chicago Bulls":        "https://cdn.nba.com/logos/nba/1610612741/primary/L/logo.png",
    "Cleveland Cavaliers":  "https://cdn.nba.com/logos/nba/1610612739/primary/L/logo.png",
    "Dallas Mavericks":     "https://cdn.nba.com/logos/nba/1610612742/primary/L/logo.png",
    "Denver Nuggets":       "https://cdn.nba.com/logos/nba/1610612743/primary/L/logo.png",
    "Detroit Pistons":      "https://cdn.nba.com/logos/nba/1610612765/primary/L/logo.png",
    "Golden State Warriors":"https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.png",
    "Houston Rockets":      "https://cdn.nba.com/logos/nba/1610612745/primary/L/logo.png",
    "Indiana Pacers":       "https://cdn.nba.com/logos/nba/1610612754/primary/L/logo.png",
    "Los Angeles Clippers": "https://cdn.nba.com/logos/nba/1610612746/primary/L/logo.png",
    "Los Angeles Lakers":   "https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.png",
    "Memphis Grizzlies":    "https://cdn.nba.com/logos/nba/1610612763/primary/L/logo.png",
    "Miami Heat":           "https://cdn.nba.com/logos/nba/1610612748/primary/L/logo.png",
    "Milwaukee Bucks":      "https://cdn.nba.com/logos/nba/1610612749/primary/L/logo.png",
    "Minnesota Timberwolves":"https://cdn.nba.com/logos/nba/1610612750/primary/L/logo.png",
    "New Orleans Pelicans": "https://cdn.nba.com/logos/nba/1610612740/primary/L/logo.png",
    "New York Knicks":      "https://cdn.nba.com/logos/nba/1610612752/primary/L/logo.png",
    "Oklahoma City Thunder":"https://cdn.nba.com/logos/nba/1610612760/primary/L/logo.png",
    "Orlando Magic":        "https://cdn.nba.com/logos/nba/1610612753/primary/L/logo.png",
    "Philadelphia 76ers":   "https://cdn.nba.com/logos/nba/1610612755/primary/L/logo.png",
    "Phoenix Suns":         "https://cdn.nba.com/logos/nba/1610612756/primary/L/logo.png",
    "Portland Trail Blazers":"https://cdn.nba.com/logos/nba/1610612757/primary/L/logo.png",
    "Sacramento Kings":     "https://cdn.nba.com/logos/nba/1610612758/primary/L/logo.png",
    "San Antonio Spurs":    "https://cdn.nba.com/logos/nba/1610612759/primary/L/logo.png",
    "Toronto Raptors":      "https://cdn.nba.com/logos/nba/1610612761/primary/L/logo.png",
    "Utah Jazz":            "https://cdn.nba.com/logos/nba/1610612762/primary/L/logo.png",
    "Washington Wizards":   "https://cdn.nba.com/logos/nba/1610612764/primary/L/logo.png"
}

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

def get_team_logo(team_name):
    url = TEAM_LOGO_URLS.get(team_name)
    if url is None:
        return None
    try:
        response = requests.get(url)
        img = plt.imread(BytesIO(response.content))
        return img
    except Exception:
        return None

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
        for idx, player_input in enumerate([input.player1(), input.player2()]):
            if player_input:
                df = get_player_shots(player_input)
                if not df.empty:
                    color = colors[idx]
                    ax.scatter(df["COURT_X"], df["COURT_Y"], s=120, color=color, edgecolor='black', alpha=0.8, label=player_input)
                    players.append((player_input, df["TEAM"].iloc[0], color))
        # Custom legend with logos or fallback markers
        handles = []
        legend_labels = []
        for player_name, team, color in players:
            logo_img = get_team_logo(team)
            if logo_img is not None:
                imagebox = OffsetImage(logo_img, zoom=0.08)
                ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
                handles.append(ab)
                legend_labels.append(f"{player_name} ({team})")
            else:
                fallback = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                         markersize=10, label=f"{player_name} ({team})")
                handles.append(fallback)
                legend_labels.append(f"{player_name} ({team})")

        class HandlerLogo(HandlerBase):
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                if isinstance(orig_handle, AnnotationBbox):
                    ab = orig_handle
                    ab.xybox = (width/2, height/2)
                    ab.set_transform(trans)
                    return [ab]
                return [orig_handle]

        if handles:
            ax.legend(handles=handles, labels=legend_labels,
                      handler_map={AnnotationBbox: HandlerLogo()},
                      loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-5, 47)
        ax.set_title("NBA Shot Chart: Two Players", fontsize=16, pad=20)
        ax.text(0, -3, "Origin (0,0) at hoop | Circles=Made, X=Missed", ha='center', fontsize=10, style='italic')
        return fig

    @output
    @render.table
    def shot_summary():
        summaries = []
        for player in [input.player1(), input.player2()]:
            if player:
                df = get_player_shots(player)
                if not df.empty:
                    summaries.append(calculate_shot_summary(df))
        if not summaries:
            return None
        summary_df = pd.DataFrame(summaries)
        return summary_df[['Player', 'Team', '3PA', '3P%', '2PA', '2P%']]

app = App(app_ui, server)
run_app(app)
