import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from city import create_city, GRID_SIZE, ROAD, BUILDING, HIGHWAY, TRAFFIC_LIGHT
from cars import create_cars, NUM_CARS
from pathfinding import (build_congestion_map, compute_all_paths,
                         move_cars_with_paths)


# ── Settings ───────────────────────────────────────────────────────────────────
MAX_TICKS    = 100
REPATH_EVERY = 5
RUSH_HOUR    = 20
INTERVAL_MS  = 200   # milliseconds between frames — lower = faster animation


# ══════════════════════════════════════════════════════════════════════════════
# BUILD STATIC CITY BACKGROUND IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def build_city_image(grid):
    """
    Pre-render the city grid as an RGB image.
    We draw this once and reuse it every frame — much faster than
    redrawing the grid from scratch each tick.
    """
    color_map = {
        ROAD:          "#d3d3d3",
        BUILDING:      "#2c2c2c",
        HIGHWAY:       "#f5a623",
        TRAFFIC_LIGHT: "#e74c3c",
    }

    h, w = grid.shape
    image = np.zeros((h, w, 3))

    for cell_type, hex_color in color_map.items():
        rgb = mcolors.to_rgb(hex_color)
        image[grid == cell_type] = rgb

    return image


# ══════════════════════════════════════════════════════════════════════════════
# SET UP THE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def build_figure():
    """
    Create the full figure layout:
    - Left  : city map with cars and congestion overlay
    - Right : live stats panel (line charts updating each tick)
    """
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')   # dark navy background

    gs = GridSpec(3, 2, figure=fig,
                  width_ratios=[2, 1],
                  hspace=0.5, wspace=0.3,
                  left=0.05, right=0.97,
                  top=0.92, bottom=0.08)

    # City map — spans all 3 rows on the left
    ax_city  = fig.add_subplot(gs[:, 0])

    # Stats panels on the right
    ax_cars  = fig.add_subplot(gs[0, 1])   # moving vs arrived count
    ax_cong  = fig.add_subplot(gs[1, 1])   # congestion over time
    ax_info  = fig.add_subplot(gs[2, 1])   # text info box

    # Style all axes dark
    for ax in [ax_city, ax_cars, ax_cong]:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

    ax_info.set_facecolor('#16213e')
    ax_info.axis('off')

    fig.suptitle("City Traffic Simulator",
                 color='white', fontsize=16, fontweight='bold')

    return fig, ax_city, ax_cars, ax_cong, ax_info


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

def run_animation():

    # ── Init simulation state ──────────────────────────────────────────────────
    grid       = create_city()
    cars       = create_cars(grid)
    city_image = build_city_image(grid)
    paths      = {}

    # History buffers for live charts
    ticks_log    = []
    moving_log   = []
    arrived_log  = []
    cong_log     = []

    rush_triggered = False

    # ── Build figure ───────────────────────────────────────────────────────────
    fig, ax_city, ax_cars, ax_cong, ax_info = build_figure()

    # ── City map base layer ────────────────────────────────────────────────────
    ax_city.imshow(city_image, interpolation='nearest', zorder=1)

    # Congestion overlay — starts as zeros, updates every frame
    cong_overlay = ax_city.imshow(
        np.zeros((GRID_SIZE, GRID_SIZE)),
        cmap='Reds', alpha=0.45, zorder=2,
        vmin=0, vmax=8,
        interpolation='nearest'
    )

    # Grid lines
    for x in range(GRID_SIZE + 1):
        ax_city.axvline(x - 0.5, color='white', linewidth=0.3, zorder=3)
    for y in range(GRID_SIZE + 1):
        ax_city.axhline(y - 0.5, color='white', linewidth=0.3, zorder=3)

    # Car scatter plots — moving (blue) and arrived (green)
    scat_moving  = ax_city.scatter([], [], c='#3498db', s=40,
                                    zorder=5, label='Moving')
    scat_arrived = ax_city.scatter([], [], c='#2ecc71', s=40,
                                    zorder=5, label='Arrived')

    # Rush hour cars get a special orange color
    scat_rush = ax_city.scatter([], [], c='#f39c12', s=40,
                                 zorder=5, label='Rush hour')

    # Legend
    legend_elements = [
        Patch(facecolor='#d3d3d3', label='Road'),
        Patch(facecolor='#2c2c2c', label='Building'),
        Patch(facecolor='#f5a623', label='Highway'),
        Patch(facecolor='#e74c3c', label='Traffic light'),
        Patch(facecolor='#3498db', label='Moving car'),
        Patch(facecolor='#2ecc71', label='Arrived car'),
        Patch(facecolor='#f39c12', label='Rush hour car'),
    ]
    ax_city.legend(handles=legend_elements, loc='lower right',
                   fontsize=7, framealpha=0.85,
                   facecolor='#1a1a2e', labelcolor='white')

    ax_city.set_title("Live City Map", color='white',
                       fontweight='bold', fontsize=12)
    ax_city.set_xlabel("X", color='white')
    ax_city.set_ylabel("Y", color='white')

    # ── Stats charts ───────────────────────────────────────────────────────────
    line_moving,  = ax_cars.plot([], [], color='#3498db',
                                  linewidth=2, label='Moving')
    line_arrived, = ax_cars.plot([], [], color='#2ecc71',
                                  linewidth=2, label='Arrived')
    ax_cars.set_xlim(0, MAX_TICKS)
    ax_cars.set_ylim(0, NUM_CARS * 1.6)
    ax_cars.set_title("Cars Over Time", fontsize=9, fontweight='bold')
    ax_cars.set_xlabel("Tick")
    ax_cars.set_ylabel("Count")
    ax_cars.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white')
    ax_cars.axvspan(RUSH_HOUR, RUSH_HOUR + 20, alpha=0.15,
                    color='orange', label='Rush hour')

    line_cong, = ax_cong.plot([], [], color='#e74c3c',
                               linewidth=2, label='Peak')
    line_mean, = ax_cong.plot([], [], color='#f39c12',
                               linewidth=2, linestyle='--', label='Mean')
    ax_cong.set_xlim(0, MAX_TICKS)
    ax_cong.set_ylim(0, 12)
    ax_cong.set_title("Congestion Over Time", fontsize=9, fontweight='bold')
    ax_cong.set_xlabel("Tick")
    ax_cong.set_ylabel("Cars/cell")
    ax_cong.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white')
    ax_cong.axvspan(RUSH_HOUR, RUSH_HOUR + 20, alpha=0.15,
                    color='orange')

    # Text info box
    info_text = ax_info.text(
        0.05, 0.95, '', transform=ax_info.transAxes,
        color='white', fontsize=9, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#0f3460',
                  edgecolor='#444466', alpha=0.9)
    )

    # ── Track rush hour car IDs ────────────────────────────────────────────────
    rush_car_ids = set()

    # ── Animation update function ─────────────────────────────────────────────
    # This function is called once per frame by FuncAnimation
    def update(tick):
        nonlocal cars, paths, rush_triggered, rush_car_ids

        # ── Rush hour ─────────────────────────────────────────────────────────
        if tick == RUSH_HOUR and not rush_triggered:
            rush_triggered = True
            from cars import create_cars as _create_cars
            new_cars = _create_cars(grid, num_cars=30)
            new_cars['car_id'] = new_cars['car_id'] + cars['car_id'].max() + 1
            rush_car_ids = set(new_cars['car_id'].values)
            cars = pd.concat([cars, new_cars], ignore_index=True)
            print(f"  ⚡ Rush hour triggered at tick {tick}")

        # ── Repath ────────────────────────────────────────────────────────────
        if tick % REPATH_EVERY == 0:
            congestion_map = build_congestion_map(cars, GRID_SIZE)
            paths = compute_all_paths(cars, grid, congestion_map)

        # ── Move ──────────────────────────────────────────────────────────────
        congestion_map = build_congestion_map(cars, GRID_SIZE)
        cars = move_cars_with_paths(cars, paths, grid)

        # ── Update congestion overlay ─────────────────────────────────────────
        cong_overlay.set_data(congestion_map)

        # ── Update car scatter plots ───────────────────────────────────────────
        moving_cars  = cars[(cars['status'] == 'moving') &
                            (~cars['car_id'].isin(rush_car_ids))]
        arrived_cars = cars[cars['status'] == 'arrived']
        rush_moving  = cars[(cars['status'] == 'moving') &
                             (cars['car_id'].isin(rush_car_ids))]

        scat_moving.set_offsets(
            moving_cars[['x', 'y']].values if len(moving_cars) > 0
            else np.empty((0, 2))
        )
        scat_arrived.set_offsets(
            arrived_cars[['x', 'y']].values if len(arrived_cars) > 0
            else np.empty((0, 2))
        )
        scat_rush.set_offsets(
            rush_moving[['x', 'y']].values if len(rush_moving) > 0
            else np.empty((0, 2))
        )

        # ── Log stats ─────────────────────────────────────────────────────────
        ticks_log.append(tick)
        moving_log.append((cars['status'] == 'moving').sum())
        arrived_log.append((cars['status'] == 'arrived').sum())
        cong_log.append(congestion_map.max())

        # ── Update line charts ────────────────────────────────────────────────
        line_moving.set_data(ticks_log, moving_log)
        line_arrived.set_data(ticks_log, arrived_log)
        line_cong.set_data(ticks_log, cong_log)
        line_mean.set_data(ticks_log,
                           [build_congestion_map(cars, GRID_SIZE).mean()
                            for _ in [0]])   # current mean only

        # ── Update info box ───────────────────────────────────────────────────
        moving_n  = (cars['status'] == 'moving').sum()
        arrived_n = (cars['status'] == 'arrived').sum()
        total_n   = cars['car_id'].nunique()
        pct       = 100 * arrived_n / total_n if total_n > 0 else 0
        peak_cong = congestion_map.max()
        rush_str  = "YES 🚨" if tick >= RUSH_HOUR else "no"

        info_text.set_text(
            f"  TICK        : {tick:03d} / {MAX_TICKS}\n"
            f"  TOTAL CARS  : {total_n}\n"
            f"  MOVING      : {moving_n}\n"
            f"  ARRIVED     : {arrived_n} ({pct:.1f}%)\n"
            f"  PEAK CONG.  : {peak_cong:.1f} cars/cell\n"
            f"  RUSH HOUR   : {rush_str}"
        )

        return (cong_overlay, scat_moving, scat_arrived,
                scat_rush, line_moving, line_arrived,
                line_cong, line_mean, info_text)

    # ── Run the animation ──────────────────────────────────────────────────────
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=MAX_TICKS,
        interval=INTERVAL_MS,
        blit=False,     # blit=False is safer across platforms
        repeat=False
    )

    plt.show()

    # ── Optionally save as gif ─────────────────────────────────────────────────
    save = input("\nSave animation as GIF? (y/n): ").strip().lower()
    if save == 'y':
        print("Saving... this may take 30–60 seconds...")
        ani.save('traffic_simulation.gif',
                 writer='pillow', fps=8, dpi=80)
        print("Saved to traffic_simulation.gif")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_animation()