import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend for Linux Mint

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, KDTree

# Constants
GRID_SIZE = 200
NUM_TAXIS = 10
NUM_CUSTOMERS = 3
SEED = 67

# Data Generation
np.random.seed(SEED)
taxi_points = np.random.rand(NUM_TAXIS, 2) * GRID_SIZE
customer_points = np.random.rand(NUM_CUSTOMERS, 2) * GRID_SIZE

# Core Logic
vor = Voronoi(taxi_points)
tree = KDTree(taxi_points)

# Query - Find nearest taxi for ALL customers
distances, indices = tree.query(customer_points)
nearest_taxis = taxi_points[indices]  # Array of nearest taxi coordinates

# --- Helper Functions for Finite Voronoi Regions ---
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    polygons.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map between point index and all ridges
    # (point index, point index) -> [ridge index, ridge index, ...]
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all_ridges.get(p1) is None:
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort vertices counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def clip_polygon(subject_polygon, clip_polygon):
    """
    Clip a polygon using Sutherland-Hodgman algorithm.
    subject_polygon: List of (x,y) vertices
    clip_polygon: List of (x,y) vertices of the clipping window (must be convex)
    """
    def inside(p, cp1, cp2):
        return (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def compute_intersection(cp1, cp2, s, e):
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0] 
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    output_list = subject_polygon
    cp1 = clip_polygon[-1]

    for cp2 in clip_polygon:
        input_list = output_list
        output_list = []
        if not input_list:
            break
        s = input_list[-1]

        for e in input_list:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    output_list.append(compute_intersection(cp1, cp2, s, e))
                output_list.append(e)
            elif inside(s, cp1, cp2):
                output_list.append(compute_intersection(cp1, cp2, s, e))
            s = e
        cp1 = cp2
    return output_list

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))

# --- Layer 1: Background Map ---
try:
    map_img = plt.imread('novi_sad_map.png')
    ax.imshow(map_img, extent=[0, GRID_SIZE, 0, GRID_SIZE], alpha=0.8)
except FileNotFoundError:
    print("Upozorenje: 'novi_sad_map.png' nije pronađen. Koristim belu pozadinu.")

# --- Layer 2: Voronoi Regions (Solid & Clipped) ---
colors = plt.get_cmap('tab20').colors
if NUM_TAXIS > 20: 
    colors = colors * (NUM_TAXIS // 20 + 1)

# Compute finite regions
regions, vertices = voronoi_finite_polygons_2d(vor, radius=GRID_SIZE*2)

# Clipping box (Counter-Clockwise)
box = [[0, 0], [GRID_SIZE, 0], [GRID_SIZE, GRID_SIZE], [0, GRID_SIZE]]

for i, region in enumerate(regions):
    polygon = vertices[region]
    
    # Clip to the map size
    clipped_polygon = clip_polygon(polygon.tolist(), box)
    
    if clipped_polygon:
        # Draw Polygon
        poly = Polygon(clipped_polygon, facecolor=colors[i % len(colors)], 
                       edgecolor='blue', linewidth=1, alpha=0.2, linestyle='solid')
        ax.add_patch(poly)

# (Skipped voronoi_plot_2d as we draw manually now)

# 2. Draw Taxis
ax.plot(taxi_points[:, 0], taxi_points[:, 1], 'ko', label='Ostala vozila')

# 3. Draw Customer
# 3. Draw Customers
ax.plot(customer_points[:, 0], customer_points[:, 1], 'rx', markersize=12, markeredgewidth=2, label='Mušterije')

# 4. Highlight Assignments & Draw Paths
# Plot all assigned taxis (using unique to avoid overplotting if multiple customers get same taxi)
unique_assigned_indices = np.unique(indices)
unique_assigned_taxis = taxi_points[unique_assigned_indices]
ax.plot(unique_assigned_taxis[:, 0], unique_assigned_taxis[:, 1], 'go', markersize=12, label='Dodeljena vozila')

# Draw paths for EACH customer
for i in range(NUM_CUSTOMERS):
    cust = customer_points[i]
    taxi = nearest_taxis[i]
    ax.plot([cust[0], taxi[0]], [cust[1], taxi[1]], 'g--')

# Formatting
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_title(f"Voronoj Optimizacija: {NUM_CUSTOMERS} mušterija", fontsize=14)
ax.legend(loc='upper right')
ax.grid(False) # Turn off grid for better map visibility

# Save Result
output_filename = 'rezultat_simulacije.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Slika sačuvana kao: {output_filename}")

# Console Report
print(f"\n--- IZVEŠTAJ SIMULACIJE ---")
print(f"Broj aktivnih vozila: {NUM_TAXIS}")
print(f"Broj mušterija: {NUM_CUSTOMERS}")
print("-" * 30)
for i in range(NUM_CUSTOMERS):
    print(f"Mušterija #{i+1} (Lok: [{customer_points[i,0]:.1f}, {customer_points[i,1]:.1f}]) -> Taksi #{indices[i]} (Dist: {distances[i]:.2f} km)")
print("-" * 30)
print(f"Status: USPEŠNO")
print(f"---------------------------\n")

plt.show()
