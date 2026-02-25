import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

GRID_SIZE    = 20    
ROAD         = 0
BUILDING     = 1
HIGHWAY      = 2
TRAFFIC_LIGHT = 3

np.random.seed(42)   


def create_city(size=GRID_SIZE):
    """
    Build the city grid as a 2D NumPy array.
    Every cell starts as a road, then we carve in
    buildings, highways, and traffic lights.
    """
 
    grid = np.zeros((size, size), dtype=np.int32)

    num_buildings = int(size * size * 0.15)
    building_coords = np.random.choice(size * size, num_buildings, replace=False)

    rows, cols = np.unravel_index(building_coords, (size, size))
    grid[rows, cols] = BUILDING

    grid[5, :]  = HIGHWAY
    grid[:, 10] = HIGHWAY

    grid[5, :]  = HIGHWAY
    grid[:, 10] = HIGHWAY

    grid[5,  ::4] = TRAFFIC_LIGHT
    grid[::4, 10] = TRAFFIC_LIGHT

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
      
        mask = grid == cell_type
        image[mask] = rgb

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, interpolation='nearest')

    for x in range(width + 1):
        ax.axvline(x - 0.5, color='white', linewidth=0.5)
    for y in range(height + 1):
        ax.axhline(y - 0.5, color='white', linewidth=0.5)

    ax.set_title("City Grid", fontsize=16, fontweight='bold')
    ax.set_xlabel("Column (X)")
    ax.set_ylabel("Row (Y)")

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
