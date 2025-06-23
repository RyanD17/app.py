from shiny import App, ui, render, reactive, run_app
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import pandas as pd
import numpy as np
import io
import base64
from functools import lru_cache


# --- Optimized court image creation ---
def create_court_image():
    """Create optimized court image once at startup"""
    fig, ax = plt.subplots(figsize=(12, 11), dpi=75)  # Lower DPI for faster rendering
    # Simplified court drawing with fewer objects
    court_elements = [
        Circle((0, 0), radius=0.75, linewidth=1.5, color='black', fill=False),
        Rectangle((-3, -0.75), 6, -0.1, linewidth=1.5, color='black'),
        Rectangle((-8, -0.75), 16, 19, linewidth=1.5, color='black', fill=False),
        Rectangle((-6, -0.75), 12, 19, linewidth=1.5, color='black', fill=False),
        Arc((0, 19), 12, 12, theta1=0, theta2=180, linewidth=1.5, color='black', fill=False),
        Arc((0, 19), 12, 12, theta1=180, theta2=0, linewidth=1.5, color='black', linestyle='dashed'),
        Arc((0, 0), 8, 8, theta1=0, theta2=180, linewidth=1.5, color='black'),
        Arc((0, 0), 47.5, 47.5, theta1=22, theta2=158, linewidth=1.5, color='black')
    ]
    for element in court_elements:
        ax.add_patch(element)
    # Add corner three-point lines
    ax.plot([-22, -22], [-0.75, 14], 'k-', linewidth=1.5)
    ax.plot([22, 22], [-0.75, 14], 'k-', linewidth=1.5)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-5, 47)
    ax.set_aspect('equal')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=75, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# Precompute court image
COURT_IMAGE = create_court_image()


# --- Ultra-fast vectorized operations ---
@lru_cache(maxsize=1)
def load_and_process_data():
    """Load and preprocess data with memory optimizations"""
    data = pd.read_csv("NBA_2025_Shots.csv")
    data.columns = [col.strip().upper().replace(" ", "_") for col in data.columns]

    # Rename columns
    rename_map = {
        "TEAM_NAME": "TEAM", "PLAYER_NAME": "PLAYER",
        "LOC_X": "COURT_X", "LOC_Y": "COURT_Y", "SHOT_MADE": "SHOT_MADE_FLAG"
    }
    data = data.rename(columns=rename_map)

    # Optimize data types to reduce memory
    data['PLAYER'] = data['PLAYER'].astype('category')
    data['TEAM'] = data['TEAM'].astype('category')
    data['SHOT_MADE_FLAG'] = data['SHOT_MADE_FLAG'].astype('int8')
    data['COURT_X'] = data['COURT_X'].astype('float32')
    data['COURT_Y'] = data['COURT_Y'].astype('float32')

    # Vectorized shot type classification
    distance = np.sqrt(data['COURT_X'] ** 2 + data['COURT_Y'] ** 2).astype('float32')
    conditions = [
        (np.abs(data['COURT_X']) >= 22 - 1e-6) & (data['COURT_Y'] <= 14),
        distance >= 23.75 - 1e-6
    ]
    data['SHOT_TYPE'] = pd.Categorical(
        np.select(conditions, ['3PT', '3PT'], default='2PT')
    )

    return data


# Load data once
DATA = load_and_process_data()
PLAYER_CHOICES = sorted(DATA['PLAYER'].cat.categories.tolist())


# --- Optimized calculation functions ---
def calculate_shot_summary_fast(player_data):
    """Ultra-fast summary calculation using vectorized operations"""
    if player_data.empty:
        return None

    # Vectorized calculations
    is_3pt = player_data['SHOT_TYPE'] == '3PT'
    is_made = player_data['SHOT_MADE_FLAG'] == 1

    attempts_3pt = is_3pt.sum()
    makes_3pt = (is_3pt & is_made).sum()
    attempts_2pt = (~is_3pt).sum()
    makes_2pt = (~is_3pt & is_made).sum()

    return {
        'Player': player_data['PLAYER'].iloc[0],
        'Team': player_data['TEAM'].iloc[0],
        '3PA': int(attempts_3pt),
        '3P%': round(makes_3pt / attempts_3pt * 100, 1) if attempts_3pt > 0 else 0.0,
        '2PA': int(attempts_2pt),
        '2P%': round(makes_2pt / attempts_2pt * 100, 1) if attempts_2pt > 0 else 0.0
    }


# --- UI (unchanged) ---
app_ui = ui.page_fluid(
    ui.panel_title("NBA Shot Chart: Compare Two Players"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Primary Player"),
            ui.input_selectize("player1", "Select or Type Player 1:",
                               choices=PLAYER_CHOICES, selected=None),
            ui.hr(),
            ui.h4("Secondary Player"),
            ui.input_selectize("player2", "Select or Type Player 2:",
                               choices=PLAYER_CHOICES, selected=None),
            ui.hr(),
            ui.h4("Shot Filters"),
            ui.input_radio_buttons(
                "shot_filter", "Show shots:",
                choices={"all": "All shots", "made": "Made shots only", "missed": "Missed shots only"},
                selected="all"
            ),
        ),
        ui.output_plot("shot_plot"),
        ui.output_table("shot_summary")
    )
)


def server(input, output, session):
    @reactive.Calc
    def get_filtered_data():
        """Single reactive calculation for both players with optimized filtering"""
        player1, player2 = input.player1(), input.player2()
        filter_type = input.shot_filter()

        if not player1 and not player2:
            return {}, {}

        # Single-pass filtering for both players
        player1_mask = (DATA["PLAYER"] == player1) if player1 else pd.Series([False] * len(DATA))
        player2_mask = (DATA["PLAYER"] == player2) if player2 else pd.Series([False] * len(DATA))

        # Apply shot filter once
        if filter_type == "made":
            shot_mask = DATA["SHOT_MADE_FLAG"] == 1
        elif filter_type == "missed":
            shot_mask = DATA["SHOT_MADE_FLAG"] == 0
        else:
            shot_mask = pd.Series([True] * len(DATA))

        # Extract data without copying
        p1_data = DATA.loc[player1_mask & shot_mask] if player1 else pd.DataFrame()
        p2_data = DATA.loc[player2_mask & shot_mask] if player2 else pd.DataFrame()

        return p1_data, p2_data

    @output
    @render.plot
    def shot_plot():
        # Reuse existing figure approach but with optimizations
        fig, ax = plt.subplots(figsize=(12, 11), dpi=75)

        # Load precomputed court image
        court_img = plt.imread(io.BytesIO(base64.b64decode(COURT_IMAGE)))
        ax.imshow(court_img, extent=[-25, 25, -5, 47], aspect='auto')

        p1_data, p2_data = get_filtered_data()
        colors = ['#1f77b4', '#ff7f0e']  # Use hex colors for better performance
        players_info = []

        # Plot both players efficiently
        datasets = [(input.player1(), p1_data, 0), (input.player2(), p2_data, 1)]

        for player_name, df, idx in datasets:
            if player_name and not df.empty:
                # Single scatter call with optimized parameters
                ax.scatter(
                    df["COURT_X"].values, df["COURT_Y"].values,
                    s=100, c=colors[idx], edgecolors='black', alpha=0.7,
                    linewidths=0.5, label=f"{player_name} ({df['TEAM'].iloc[0]})"
                )
                players_info.append(player_name)

        # Add legend only if there are players
        if players_info:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)

        ax.set_title("NBA Shot Chart: Two Players", fontsize=16, pad=20)
        ax.text(0.5, -0.08, "Origin (0,0) at hoop",
                ha='center', fontsize=10, style='italic', transform=ax.transAxes)

        return fig

    @output
    @render.table
    def shot_summary():
        p1_data, p2_data = get_filtered_data()
        summaries = []

        # Process both players efficiently
        for player_name, df in [(input.player1(), p1_data), (input.player2(), p2_data)]:
            if player_name and not df.empty:
                summary = calculate_shot_summary_fast(df)
                if summary:
                    summaries.append(summary)

        if not summaries:
            return None

        return pd.DataFrame(summaries)[['Player', 'Team', '3PA', '3P%', '2PA', '2P%']]


app = App(app_ui, server)
run_app(app)
