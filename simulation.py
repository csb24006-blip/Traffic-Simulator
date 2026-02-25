import numpy as np
import pandas as pd

from city import create_city, GRID_SIZE
from cars import create_cars, NUM_CARS
from pathfinding import (build_congestion_map, compute_all_paths,
                         move_cars_with_paths)

MAX_TICKS        = 100   # how long the simulation runs
REPATH_EVERY     = 5     # recompute paths every N ticks
RUSH_HOUR_TICKS  = (20, 40)   # extra cars spawn during this tick range
RUSH_HOUR_EXTRA  = 30         # how many extra cars spawn at rush hour


def log_snapshot(cars, congestion_map, tick):
    """
    Capture the current state of the simulation as a single Pandas DataFrame row
    per car. This builds up our full history for analysis in Stage 5.
    """
    snapshot = cars[['car_id', 'x', 'y', 'status', 'ticks_traveled',
                      'distance_to_dest', 'speed']].copy()
    snapshot['tick'] = tick

    # Attach the congestion level at each car's current cell
    xs = snapshot['x'].astype(int).values
    ys = snapshot['y'].astype(int).values
    snapshot['local_congestion'] = congestion_map[ys, xs]

    return snapshot


def spawn_rush_hour_cars(cars, grid, num_extra=RUSH_HOUR_EXTRA):
    """
    Inject extra cars into the simulation to simulate rush hour.
    These are appended as new rows to the existing cars DataFrame.
    """
    from cars import create_cars

    print(f"  ⚡ Rush hour! Spawning {num_extra} extra cars...")
    new_cars = create_cars(grid, num_cars=num_extra)

    # Give new cars unique IDs continuing from existing ones
    new_cars['car_id'] = new_cars['car_id'] + cars['car_id'].max() + 1

    combined = pd.concat([cars, new_cars], ignore_index=True)
    return combined


def run_simulation(grid, cars, max_ticks=MAX_TICKS):
    """
    Main simulation engine. Runs for max_ticks steps and returns
    the full history log as a Pandas DataFrame.

    Parameters:
        grid    : 2D NumPy city grid
        cars    : initial cars DataFrame
        max_ticks: number of ticks to simulate

    Returns:
        history : DataFrame with one row per car per tick
        cars    : final state of all cars
        grid    : city grid (unchanged, returned for convenience)
    """

    history_frames = []   # we'll concatenate these at the end
    paths = {}            # path cache: {car_id: [(row,col), ...]}

    print(f"Starting simulation: {len(cars)} cars, {max_ticks} ticks\n")

    for tick in range(max_ticks):

        congestion_map = build_congestion_map(cars, GRID_SIZE)

        if tick == RUSH_HOUR_TICKS[0]:
            cars = spawn_rush_hour_cars(cars, grid)

        if tick % REPATH_EVERY == 0:
            paths = compute_all_paths(cars, grid, congestion_map)

        cars = move_cars_with_paths(cars, paths, grid)

        snapshot = log_snapshot(cars, congestion_map, tick)
        history_frames.append(snapshot)

        if tick % 10 == 0 or tick == max_ticks - 1:
            moving  = (cars['status'] == 'moving').sum()
            arrived = (cars['status'] == 'arrived').sum()
            max_cong = congestion_map.max()
            print(f"  Tick {tick:03d} | Moving: {moving:3d} | "
                  f"Arrived: {arrived:3d} | "
                  f"Peak congestion: {max_cong:.0f} cars/cell")

        if (cars['status'] == 'moving').sum() == 0:
            print(f"\n  All cars arrived by tick {tick}!")
            break

    print("\nBuilding history DataFrame...")
    history = pd.concat(history_frames, ignore_index=True)
    print(f"History shape: {history.shape} "
          f"({history.shape[0]:,} rows × {history.shape[1]} columns)")

    return history, cars, grid


def summarize_simulation(history, cars):
    """
    Print a clean summary of what happened during the simulation.
    Uses Pandas groupby and aggregation — a preview of Stage 5.
    """

    print("\n" + "="*55)
    print("         SIMULATION SUMMARY")
    print("="*55)

    total_cars  = cars['car_id'].nunique()
    arrived     = (cars['status'] == 'arrived').sum()
    still_moving = (cars['status'] == 'moving').sum()

    print(f"  Total cars       : {total_cars}")
    print(f"  Arrived          : {arrived} "
          f"({100*arrived/total_cars:.1f}%)")
    print(f"  Still moving     : {still_moving}")

    # Average trip duration for cars that completed their journey
    completed = cars[cars['status'] == 'arrived']
    if len(completed) > 0:
        avg_ticks = completed['ticks_traveled'].mean()
        max_ticks = completed['ticks_traveled'].max()
        min_ticks = completed['ticks_traveled'].min()
        print(f"\n  Avg trip length  : {avg_ticks:.1f} ticks")
        print(f"  Shortest trip    : {min_ticks} ticks")
        print(f"  Longest trip     : {max_ticks} ticks")

    # Peak congestion tick
    peak_tick = (history.groupby('tick')['local_congestion']
                        .mean()
                        .idxmax())
    print(f"\n  Busiest tick     : Tick {peak_tick}")

    # Most congested area
    hotspot = (history.groupby(['x', 'y'])['local_congestion']
                      .mean()
                      .idxmax())
    print(f"  Congestion hotspot: cell {hotspot}")
    print("="*55)

if __name__ == "__main__":
    grid = create_city()
    cars = create_cars(grid)

    history, final_cars, grid = run_simulation(grid, cars)
    summarize_simulation(history, final_cars)

    # Save history to CSV so Stage 5 can load it instantly
    history.to_csv('simulation_history.csv', index=False)
    print("\nHistory saved to simulation_history.csv")

    # Quick peek at the history DataFrame
    print("\nSample rows from history:")
    print(history.head(10).to_string())
