import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Constants ──────────────────────────────────────────────────────────────────
GRID_SIZE    = 20    # 20x20 city
ROAD         = 0
BUILDING     = 1
HIGHWAY      = 2
TRAFFIC_LIGHT = 3

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)   # same city every run until you change this


def create_city(size=GRID_SIZE):
    """
    Build the city grid as a 2D NumPy array.
    Every cell starts as a road, then we carve in
    buildings, highways, and traffic lights.
    """

    # Start with all roads
    grid = np.zeros((size, size), dtype=np.int32)

    # ── Add buildings (randomly scattered) ────────────────────────────────────
    # We'll make ~15% of the grid buildings
    num_buildings = int(size * size * 0.15)
    building_coords = np.random.choice(size * size, num_buildings, replace=False)

    # Convert flat indices to 2D row, col positions
    rows, cols = np.unravel_index(building_coords, (size, size))
    grid[rows, cols] = BUILDING

    # ── Add highways (two straight lines — horizontal and vertical) ───────────
    # A horizontal highway cuts across row 5
    # A vertical highway cuts across col 10
    grid[5, :]  = HIGHWAY
    grid[:, 10] = HIGHWAY

    # Buildings should not overwrite highways, so re-clear highway cells
    # (highways are more important than buildings on those paths)
    grid[5, :]  = HIGHWAY
    grid[:, 10] = HIGHWAY

    # ── Add traffic lights at highway intersections ────────────────────────────
    # Every 4th cell along the highways becomes a traffic light
    grid[5,  ::4] = TRAFFIC_LIGHT
    grid[::4, 10] = TRAFFIC_LIGHT

    # ── Make sure the grid has at least one clear border road ─────────────────
    # The outer edge stays as roads so cars can always enter/exit the city
    grid[0,  :] = ROAD
    grid[-1, :] = ROAD
    grid[:,  0] = ROAD
    grid[:, -1] = ROAD

    return grid

def visualize_city(grid):
    """
    Draw the city grid using Matplotlib.
    Each cell type gets its own color.
    """

    # Define a color for each cell type
    # 0=Road, 1=Building, 2=Highway, 3=Traffic Light
    color_map = {
        ROAD:          "#d3d3d3",   # light grey
        BUILDING:      "#2c2c2c",   # dark charcoal
        HIGHWAY:       "#f5a623",   # amber/orange
        TRAFFIC_LIGHT: "#e74c3c",   # red
    }

    # Build an RGB image array from the grid
    # Each cell becomes an (R, G, B) pixel
    height, width = grid.shape
    image = np.zeros((height, width, 3))   # 3 channels: R, G, B

    for cell_type, hex_color in color_map.items():
        # Convert hex color to RGB values between 0 and 1
        rgb = mcolors.to_rgb(hex_color)
        # Wherever the grid matches this cell type, paint that color
        mask = grid == cell_type
        image[mask] = rgb

    # ── Draw the image ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, interpolation='nearest')

    # Draw grid lines so individual cells are visible
    for x in range(width + 1):
        ax.axvline(x - 0.5, color='white', linewidth=0.5)
    for y in range(height + 1):
        ax.axhline(y - 0.5, color='white', linewidth=0.5)

    # Labels and title
    ax.set_title("City Grid", fontsize=16, fontweight='bold')
    ax.set_xlabel("Column (X)")
    ax.set_ylabel("Row (Y)")

    # Add a simple legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color="#d3d3d3", label="Road"),
        plt.Rectangle((0,0),1,1, color="#2c2c2c", label="Building"),
        plt.Rectangle((0,0),1,1, color="#f5a623", label="Highway"),
        plt.Rectangle((0,0),1,1, color="#e74c3c", label="Traffic Light"),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.show()


# ── Run it ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    grid = create_city()

    print("Grid shape:", grid.shape)
    print("Cell type counts:")
    unique, counts = np.unique(grid, return_counts=True)
    labels = {ROAD: "Road", BUILDING: "Building",
              HIGHWAY: "Highway", TRAFFIC_LIGHT: "Traffic Light"}
    for u, c in zip(unique, counts):
        print(f"  {labels[u]}: {c} cells")

    visualize_city(grid)