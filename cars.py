import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from city import create_city, ROAD, HIGHWAY, BUILDING, TRAFFIC_LIGHT, GRID_SIZE


# ── Constants ──────────────────────────────────────────────────────────────────
NUM_CARS = 100


def get_valid_road_cells(grid):
    """
    Return all (row, col) positions in the grid that are NOT buildings.
    Cars can only spawn and travel on roads, highways, and traffic lights.
    """
    # np.argwhere returns the coordinates of all cells matching the condition
    valid_cells = np.argwhere(grid != BUILDING)
    return valid_cells   # shape: (N, 2) — N valid cells, each with [row, col]


def create_cars(grid, num_cars=NUM_CARS):
    """
    Spawn all cars at once as a Pandas DataFrame.
    Each car gets a random start and destination on a valid road cell.
    """

    valid_cells = get_valid_road_cells(grid)

    # ── Pick random start positions ────────────────────────────────────────────
    # We sample num_cars indices from the valid cells array
    start_indices = np.random.choice(len(valid_cells), size=num_cars, replace=True)
    start_positions = valid_cells[start_indices]   # shape: (num_cars, 2)

    # ── Pick random destination positions ─────────────────────────────────────
    dest_indices = np.random.choice(len(valid_cells), size=num_cars, replace=True)
    dest_positions = valid_cells[dest_indices]     # shape: (num_cars, 2)

    # ── Assign speeds based on spawn location ─────────────────────────────────
    # Cars that spawn on a highway start faster
    spawn_types = grid[start_positions[:, 0], start_positions[:, 1]]
    speeds = np.where(spawn_types == HIGHWAY, 2, 1)

    # ── Build the DataFrame ───────────────────────────────────────────────────
    cars = pd.DataFrame({
        'car_id'  : np.arange(num_cars),
        'x'       : start_positions[:, 1],    # col = x (horizontal)
        'y'       : start_positions[:, 0],    # row = y (vertical)
        'dest_x'  : dest_positions[:, 1],
        'dest_y'  : dest_positions[:, 0],
        'speed'   : speeds,
        'status'  : 'moving',                 # 'moving' or 'arrived'
        'ticks_traveled' : 0,                 # how many ticks this car has lived
        'distance_to_dest': 0.0,              # we'll compute this next
    })

    # ── Compute initial Manhattan distance to destination ─────────────────────
    # Manhattan distance = |x2 - x1| + |y2 - y1|
    # This is how far each car is from its goal right now
    cars['distance_to_dest'] = (
        (cars['dest_x'] - cars['x']).abs() +
        (cars['dest_y'] - cars['y']).abs()
    )

    # ── Drop cars that spawned exactly on their destination ───────────────────
    cars = cars[cars['distance_to_dest'] > 0].reset_index(drop=True)
    cars['car_id'] = cars.index   # re-assign clean IDs

    print(f"Spawned {len(cars)} cars successfully.")
    print(cars.head(10))          # preview the first 10 rows
    return cars

def move_cars(cars, grid):
    """
    Advance every 'moving' car one step toward its destination.
    """

    moving = cars['status'] == 'moving'

    if moving.sum() == 0:
        return cars

    dx = cars.loc[moving, 'dest_x'] - cars.loc[moving, 'x']
    dy = cars.loc[moving, 'dest_y'] - cars.loc[moving, 'y']

    step_x = np.sign(dx).astype(int)
    step_y = np.sign(dy).astype(int)

    move_horizontal = dx.abs() >= dy.abs()

    new_x = cars.loc[moving, 'x'].copy().astype(int)
    new_y = cars.loc[moving, 'y'].copy().astype(int)

    # ── Primary move ──────────────────────────────────────────────────────────
    new_x[move_horizontal]  += step_x[move_horizontal]
    new_y[~move_horizontal] += step_y[~move_horizontal]

    # Clamp to grid
    new_x = new_x.clip(0, grid.shape[1] - 1)
    new_y = new_y.clip(0, grid.shape[0] - 1)

    # ── Building check: try the OTHER axis as fallback ─────────────────────── 
    is_building = grid[new_y, new_x] == BUILDING

    # Reset blocked cars to original position
    original_x = cars.loc[moving, 'x'].astype(int)
    original_y = cars.loc[moving, 'y'].astype(int)

    new_x[is_building] = original_x[is_building]
    new_y[is_building] = original_y[is_building]

    # Now try the OTHER axis (not the primary one)
    new_x[is_building & move_horizontal]  = (original_x + step_x)[is_building & move_horizontal]
    new_y[is_building & move_horizontal]  = original_y[is_building & move_horizontal]
    new_x[is_building & ~move_horizontal] = original_x[is_building & ~move_horizontal]
    new_y[is_building & ~move_horizontal] = (original_y + step_y)[is_building & ~move_horizontal]

    # Clamp again after fallback
    new_x = new_x.clip(0, grid.shape[1] - 1)
    new_y = new_y.clip(0, grid.shape[0] - 1)

    # ── If fallback is ALSO a building, just stay put this tick ──────────────
    still_blocked = grid[new_y, new_x] == BUILDING
    new_x[still_blocked] = original_x[still_blocked]
    new_y[still_blocked] = original_y[still_blocked]

    # ── Highway speed boost ───────────────────────────────────────────────────
    on_highway = grid[new_y, new_x] == HIGHWAY

    boosted_x = (new_x + step_x * on_highway).clip(0, grid.shape[1] - 1)
    boosted_y = (new_y + step_y * on_highway).clip(0, grid.shape[0] - 1)

    # Only apply boost if the boosted cell isn't a building
    boost_blocked = grid[boosted_y, boosted_x] == BUILDING
    new_x[on_highway & ~boost_blocked] = boosted_x[on_highway & ~boost_blocked]
    new_y[on_highway & ~boost_blocked] = boosted_y[on_highway & ~boost_blocked]

    # ── Write back ────────────────────────────────────────────────────────────
    cars.loc[moving, 'x'] = new_x.values
    cars.loc[moving, 'y'] = new_y.values
    cars.loc[moving, 'ticks_traveled'] += 1

    # ── Update distance ───────────────────────────────────────────────────────
    cars.loc[moving, 'distance_to_dest'] = (
        (cars.loc[moving, 'dest_x'] - cars.loc[moving, 'x']).abs() +
        (cars.loc[moving, 'dest_y'] - cars.loc[moving, 'y']).abs()
    )

    # ── Mark arrived ──────────────────────────────────────────────────────────
    just_arrived = moving & (cars['distance_to_dest'] == 0)
    cars.loc[just_arrived, 'status'] = 'arrived'

    return cars

def visualize_cars(grid, cars, tick=0):
    """
    Draw the city grid with all active cars overlaid as colored dots.
    Moving cars = blue, arrived cars = green.
    """

    color_map = {
        ROAD:          "#d3d3d3",
        BUILDING:      "#2c2c2c",
        HIGHWAY:       "#f5a623",
        TRAFFIC_LIGHT: "#e74c3c",
    }

    height, width = grid.shape
    image = np.zeros((height, width, 3))
    for cell_type, hex_color in color_map.items():
        rgb = mcolors.to_rgb(hex_color)
        image[grid == cell_type] = rgb

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image, interpolation='nearest')

    # Draw grid lines
    for x in range(width + 1):
        ax.axvline(x - 0.5, color='white', linewidth=0.4)
    for y in range(height + 1):
        ax.axhline(y - 0.5, color='white', linewidth=0.4)

    # ── Plot moving cars as blue dots ─────────────────────────────────────────
    moving_cars = cars[cars['status'] == 'moving']
    ax.scatter(moving_cars['x'], moving_cars['y'],
               color='#3498db', s=60, zorder=5, label='Moving')

    # ── Plot arrived cars as green dots ───────────────────────────────────────
    arrived_cars = cars[cars['status'] == 'arrived']
    ax.scatter(arrived_cars['x'], arrived_cars['y'],
               color='#2ecc71', s=60, zorder=5, label='Arrived')

    ax.set_title(f"City Traffic — Tick {tick} | "
                 f"Moving: {len(moving_cars)} | Arrived: {len(arrived_cars)}",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.show()


# ── Run it ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    grid = create_city()
    cars = create_cars(grid)

    print("\n--- Tick 0 (spawn) ---")
    visualize_cars(grid, cars, tick=0)

    # Simulate 5 ticks manually so you can see movement
    for tick in range(1, 6):
        cars = move_cars(cars, grid)
        print(f"\n--- Tick {tick} ---")
        print(f"Moving: {(cars['status'] == 'moving').sum()} | "
              f"Arrived: {(cars['status'] == 'arrived').sum()}")

    print("\nFinal car states (first 10):")
    print(cars[['car_id','x','y','dest_x','dest_y',
                'status','ticks_traveled']].head(10))

    visualize_cars(grid, cars, tick=5)