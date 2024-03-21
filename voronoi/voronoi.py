import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from skimage import io, filters
from skimage.feature import peak_local_max

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
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

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


image = io.imread('bw_blur.png')

coords = peak_local_max(image, min_distance=10, exclude_border=1)
coords = coords[:, :2]

if len(image.shape) > 2:
    image = np.mean(image, axis=2)

threshold_value = 100

coords_to_check = [(x, y) for x, y in coords]
new_coords = []
for x, y in coords_to_check:
    if image[x, y] > threshold_value:
        new_coords.append((x, y))

plt.imshow(image, cmap='gray')
if new_coords:
    x_vals, y_vals = zip(*new_coords)
    plt.plot(y_vals, x_vals, 'r.')
plt.axis('off')
plt.savefig('bw_peaks.png', bbox_inches='tight', pad_inches=0)
plt.close()
new_coords = [(y, x) for x, y in new_coords]

vor = Voronoi(new_coords)

fig, ax = plt.subplots()

# Clip the Voronoi diagram to the extent of the image
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='red', line_width=1)

# Plot the image
ax.imshow(image, cmap='gray')

# Set limits to match the image extent
min_x, min_y = np.min(vor.points, axis=0)
max_x, max_y = np.max(vor.points, axis=0)
ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])

plt.axis('off')

# Save the plot with Voronoi diagram and image background without axis
plt.savefig('voronoi_background.png', bbox_inches='tight', pad_inches=0)
plt.close()
# # Show the plot
# plt.axis('off')
# plt.show()

# Plot the Voronoi diagram without image background
plt.figure()

# plot
regions, vertices = voronoi_finite_polygons_2d(vor)
print(regions)

# for region in regions:
#     polygon = vertices[region]
#     plt.fill(*zip(*polygon), alpha=0.4)

def polygon_area(vertices):
    x, y = zip(*vertices)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Calculate the area of each region
region_areas = [polygon_area(vertices[region]) for region in regions]

# Normalize the region sizes
max_size = max(region_areas)
min_size = min(region_areas)
normalized_sizes = [(size - min_size) / (max_size - min_size) for size in region_areas]

min_alpha  = 0.4

#rgb
colour = (87, 171, 87)
colour = tuple(x/255 for x in colour)

# Plot and fill the regions with shades of green based on region size
for region, normalized_size in zip(regions, normalized_sizes):
    polygon = vertices[region]
    # Apply exponential scaling to the alpha values based on normalized size
    exponent = 700  # Adjust this value to fine-tune the contrast
    alpha = (min_alpha + (1-min_alpha) * (1 - np.exp(-exponent * normalized_size)))
    plt.fill(*zip(*polygon), alpha=alpha, color=colour, edgecolor='black', linewidth=0.1)

#seed location
# if new_coords:
#     x_vals, y_vals = zip(*new_coords)
#     plt.plot(x_vals, y_vals, 'b.')

#IMPORTANT DONT DELETE
plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

# Remove axis
plt.axis('off')

# Save the plot without image background without axis
plt.savefig('voronoi_green.png', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()