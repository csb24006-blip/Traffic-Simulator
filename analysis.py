import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from city import create_city, GRID_SIZE

def load_history(path='simulation_history.csv'):
    """
    Load the simulation history CSV and do basic type cleanup.
    """
    history = pd.read_csv(path)

    history['tick']             = history['tick'].astype(int)
    history['car_id']           = history['car_id'].astype(int)
    history['x']                = history['x'].astype(int)
    history['y']                = history['y'].astype(int)
    history['local_congestion'] = history['local_congestion'].astype(float)

    print(f"Loaded history: {history.shape[0]:,} rows × {history.shape[1]} columns")
    print(f"Ticks: {history['tick'].min()} → {history['tick'].max()}")
    print(f"Unique cars: {history['car_id'].nunique()}")
    print()
    return history

def analyze_congestion_over_time(history):
    """
    For each tick, compute:
    - Mean congestion across all cells
    - Peak congestion (max on any single cell)
    - Number of still-moving cars

    Returns a per-tick summary DataFrame.
    """
    per_tick = (history
        .groupby('tick')
        .agg(
            mean_congestion = ('local_congestion', 'mean'),
            peak_congestion = ('local_congestion', 'max'),
            moving_cars     = ('status', lambda x: (x == 'moving').sum()),
            arrived_cars    = ('status', lambda x: (x == 'arrived').sum()),
        )
        .reset_index()
    )

    print("── Congestion Over Time (sample) ──")
    print(per_tick.head(10).to_string(index=False))
    print()
    return per_tick

def analyze_hotspots(history, top_n=10):
    """
    Find the most congested cells across the entire simulation.
    Groups by (x, y) and computes average congestion at each cell.
    """
    hotspots = (history
        .groupby(['x', 'y'])
        .agg(
            avg_congestion  = ('local_congestion', 'mean'),
            peak_congestion = ('local_congestion', 'max'),
            total_visits    = ('car_id', 'count'),
        )
        .reset_index()
        .sort_values('avg_congestion', ascending=False)
    )

    print(f"── Top {top_n} Congestion Hotspots ──")
    print(hotspots.head(top_n).to_string(index=False))
    print()
    return hotspots

def analyze_trip_durations(history):
    """
    For each car that arrived, find its total trip duration.
    Compare pre-rush-hour cars vs rush hour cars.
    """

    final_states = (history
        .sort_values('tick')
        .groupby('car_id')
        .last()
        .reset_index()
    )

    arrived = final_states[final_states['status'] == 'arrived'].copy()

    id_cutoff = arrived['car_id'].quantile(0.70)  
    arrived['cohort'] = np.where(
        arrived['car_id'] <= id_cutoff,
        'Original cars',
        'Rush hour cars'
    )

    print("── Trip Duration by Cohort ──")
    cohort_summary = (arrived
        .groupby('cohort')['ticks_traveled']
        .agg(['mean', 'median', 'min', 'max', 'count'])
        .round(1)
    )
    print(cohort_summary.to_string())
    print()
    return arrived

def visualize_analysis(history, per_tick, hotspots, arrived, grid):
    """
    Render a 4-panel analysis dashboard:
    - Panel 1: Congestion & active cars over time
    - Panel 2: Spatial hotspot heatmap
    - Panel 3: Trip duration histogram by cohort
    - Panel 4: Top 10 bottleneck cells ranked
    """

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Traffic Simulator — Analysis Dashboard",
                 fontsize=18, fontweight='bold', y=0.98)

    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])

    ax1.plot(per_tick['tick'], per_tick['mean_congestion'],
             color='#3498db', linewidth=2, label='Mean congestion')
    ax1.plot(per_tick['tick'], per_tick['peak_congestion'],
             color='#e74c3c', linewidth=2, linestyle='--', label='Peak congestion')

    rush_start, rush_end = 20, 40
    ax1.axvspan(rush_start, rush_end, alpha=0.15, color='orange', label='Rush hour')

    ax1.set_title("Congestion Over Time", fontweight='bold')
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("Cars per cell")
    ax1.legend(fontsize=8)

    ax1b = ax1.twinx()
    ax1b.fill_between(per_tick['tick'], per_tick['moving_cars'],
                      alpha=0.15, color='#2ecc71', label='Moving cars')
    ax1b.set_ylabel("Moving cars", color='#2ecc71')
    ax1b.tick_params(axis='y', labelcolor='#2ecc71')

    ax2 = fig.add_subplot(gs[0, 1])

    heatmap = np.zeros((GRID_SIZE, GRID_SIZE))
    for _, row in hotspots.iterrows():
        heatmap[int(row['y']), int(row['x'])] = row['avg_congestion']

    building_mask = grid == 1   

    im = ax2.imshow(heatmap, cmap='YlOrRd', interpolation='nearest',
                    vmin=0, vmax=hotspots['avg_congestion'].quantile(0.95))

    building_overlay = np.zeros((*grid.shape, 4))   
    building_overlay[building_mask] = [0.2, 0.2, 0.2, 0.7]
    ax2.imshow(building_overlay, interpolation='nearest')

    plt.colorbar(im, ax=ax2, label='Avg congestion', shrink=0.8)
    ax2.set_title("Congestion Heatmap", fontweight='bold')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    ax3 = fig.add_subplot(gs[1, 0])

    cohorts = arrived['cohort'].unique()
    colors  = {'Original cars': '#3498db', 'Rush hour cars': '#e74c3c'}

    for cohort in cohorts:
        subset = arrived[arrived['cohort'] == cohort]['ticks_traveled']
        ax3.hist(subset, bins=20, alpha=0.6,
                 color=colors.get(cohort, 'grey'),
                 label=f"{cohort} (n={len(subset)})")

    ax3.axvline(arrived['ticks_traveled'].mean(), color='black',
                linestyle='--', linewidth=1.5, label='Overall mean')

    ax3.set_title("Trip Duration Distribution", fontweight='bold')
    ax3.set_xlabel("Ticks to arrive")
    ax3.set_ylabel("Number of cars")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])

    top10 = hotspots.head(10).copy()
    top10['cell_label'] = top10.apply(
        lambda r: f"({int(r['x'])},{int(r['y'])})", axis=1
    )

    bars = ax4.barh(top10['cell_label'], top10['avg_congestion'],
                    color='#e74c3c', alpha=0.8, edgecolor='white')

    for bar, val in zip(bars, top10['avg_congestion']):
        ax4.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=8)

    ax4.set_title("Top 10 Bottleneck Cells", fontweight='bold')
    ax4.set_xlabel("Avg congestion (cars/cell)")
    ax4.invert_yaxis()   

    plt.savefig('analysis_dashboard.png', dpi=150, bbox_inches='tight')
    print("Dashboard saved to analysis_dashboard.png")
    plt.show()

if __name__ == "__main__":
    grid    = create_city()
    history = load_history()

    per_tick = analyze_congestion_over_time(history)
    hotspots = analyze_hotspots(history)
    arrived  = analyze_trip_durations(history)

    visualize_analysis(history, per_tick, hotspots, arrived, grid)
