import numpy as np
import pandas as pd
from city import ROAD, BUILDING, HIGHWAY, TRAFFIC_LIGHT

# ── Cell traversal costs ───────────────────────────────────────────────────────
# Lower cost = more attractive to route through
BASE_COSTS = {
    ROAD:          1.0,
    HIGHWAY:       0.5,    # highways are twice as attractive
    TRAFFIC_LIGHT: 1.5,    # slight penalty for intersections
    BUILDING:      np.inf, # impassable
}


def build_cost_grid(grid, congestion_map):
    """
    Build a floating point cost grid by combining:
    - Base terrain cost (road vs highway vs traffic light)
    - Current congestion on each cell (more cars = higher cost)

    Parameters:
        grid          : the city grid (2D NumPy int array)
        congestion_map: 2D NumPy array counting cars per cell

    Returns:
        cost_grid: 2D NumPy float array
    """
    size = grid.shape[0]
    cost_grid = np.ones((size, size), dtype=np.float32)

    # Apply base terrain costs
    for cell_type, base_cost in BASE_COSTS.items():
        cost_grid[grid == cell_type] = base_cost

    # Add congestion penalty — each extra car on a cell adds 0.5 cost
    congestion_penalty = congestion_map * 0.5
    cost_grid += congestion_penalty

    # Buildings stay impassable regardless of congestion
    cost_grid[grid == BUILDING] = np.inf

    return cost_grid


def build_congestion_map(cars, grid_size):
    """
    Count how many moving cars are on each cell right now.
    Returns a 2D NumPy array of shape (grid_size, grid_size).
    """
    congestion = np.zeros((grid_size, grid_size), dtype=np.float32)

    moving_cars = cars[cars['status'] == 'moving']

    if len(moving_cars) == 0:
        return congestion

    # Use np.add.at to count cars per cell without looping
    xs = moving_cars['x'].astype(int).values
    ys = moving_cars['y'].astype(int).values
    np.add.at(congestion, (ys, xs), 1)

    return congestion


def dijkstra(cost_grid, start, end):
    """
    Find the lowest-cost path from start to end on the cost grid.

    Parameters:
        cost_grid : 2D NumPy float array
        start     : (row, col) tuple
        end       : (row, col) tuple

    Returns:
        path: list of (row, col) tuples from start to end,
              or empty list if no path found
    """
    import heapq

    rows, cols = cost_grid.shape
    visited = np.full((rows, cols), False)
    dist = np.full((rows, cols), np.inf, dtype=np.float32)
    prev = np.full((rows, cols, 2), -1, dtype=np.int32)

    dist[start] = 0.0
    # Priority queue: (cost, row, col)
    heap = [(0.0, start[0], start[1])]

    # 4-directional movement (up, down, left, right)
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    while heap:
        current_cost, r, c = heapq.heappop(heap)

        if visited[r, c]:
            continue
        visited[r, c] = True

        # Reached the destination
        if (r, c) == end:
            break

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # Skip out-of-bounds
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            # Skip buildings and already visited
            if visited[nr, nc] or cost_grid[nr, nc] == np.inf:
                continue

            new_cost = current_cost + cost_grid[nr, nc]
            if new_cost < dist[nr, nc]:
                dist[nr, nc] = new_cost
                prev[nr, nc] = [r, c]
                heapq.heappush(heap, (new_cost, nr, nc))

    # ── Reconstruct path by walking backwards through prev ───────────────────
    path = []
    r, c = end

    # If we never reached the end, return empty path
    if dist[end] == np.inf:
        return path

    while (r, c) != start:
        path.append((r, c))
        pr, pc = prev[r, c]
        r, c = pr, pc

    path.append(start)
    path.reverse()
    return path


def get_next_step(path, current_pos):
    """
    Given a full path and current position,
    return the very next (row, col) step to take.
    """
    if len(path) < 2:
        return current_pos   # already at destination

    # path[0] should be current position, path[1] is next step
    # Find where we are in the path and return the next node
    for i, step in enumerate(path):
        if step == current_pos and i + 1 < len(path):
            return path[i + 1]

    # Fallback: just return the second step
    return path[1]

def compute_all_paths(cars, grid, congestion_map):
    """
    Compute and cache a path for every moving car that doesn't have one yet.
    Returns a dictionary: {car_id: [list of (row,col) steps]}
    """
    cost_grid = build_cost_grid(grid, congestion_map)
    paths = {}

    moving_cars = cars[cars['status'] == 'moving']

    for _, car in moving_cars.iterrows():
        start = (int(car['y']), int(car['x']))
        end   = (int(car['dest_y']), int(car['dest_x']))

        if start == end:
            continue

        path = dijkstra(cost_grid, start, end)
        if path:
            paths[car['car_id']] = path

    return paths


def move_cars_with_paths(cars, paths, grid):
    """
    Move each car one step along its cached path.
    If a car has no path, it stays put this tick.
    """
    moving = cars['status'] == 'moving'
    new_x = cars['x'].copy()
    new_y = cars['y'].copy()

    for idx, car in cars[moving].iterrows():
        car_id = car['car_id']

        if car_id not in paths:
            continue

        path = paths[car_id]
        current_pos = (int(car['y']), int(car['x']))
        next_step = get_next_step(path, current_pos)

        new_y.loc[idx] = next_step[0]
        new_x.loc[idx] = next_step[1]

    cars['x'] = new_x
    cars['y'] = new_y
    cars.loc[moving, 'ticks_traveled'] += 1

    # ── Update distance and arrival ───────────────────────────────────────────
    cars.loc[moving, 'distance_to_dest'] = (
        (cars.loc[moving, 'dest_x'] - cars.loc[moving, 'x']).abs() +
        (cars.loc[moving, 'dest_y'] - cars.loc[moving, 'y']).abs()
    )

    just_arrived = moving & (cars['distance_to_dest'] == 0)
    cars.loc[just_arrived, 'status'] = 'arrived'

    return cars

if __name__ == "__main__":
    from city import create_city
    from cars import create_cars

    grid = create_city()
    cars = create_cars(grid)

    print("\nBuilding congestion map...")
    congestion_map = build_congestion_map(cars, grid.shape[0])
    print(f"Max congestion on any cell: {congestion_map.max()}")

    print("\nComputing paths for all cars (this may take a second)...")
    paths = compute_all_paths(cars, grid, congestion_map)
    print(f"Paths computed for {len(paths)} cars")

    # Show a sample path for car 0
    if 0 in paths:
        print(f"\nCar 0 path ({len(paths[0])} steps):")
        print(paths[0][:10], "...")   # first 10 steps

    # Run 10 ticks
    for tick in range(1, 11):
        congestion_map = build_congestion_map(cars, grid.shape[0])
        # Recompute paths every 5 ticks to adapt to congestion changes
        if tick % 5 == 1:
            paths = compute_all_paths(cars, grid, congestion_map)
        cars = move_cars_with_paths(cars, paths, grid)
        arrived = (cars['status'] == 'arrived').sum()
        print(f"Tick {tick:02d} | Arrived: {arrived}/{len(cars)}")