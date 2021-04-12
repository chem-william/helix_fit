import os
from ase.visualize import view
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(
    style="ticks",
    rc={
        "font.family": "Arial",
        "font.size": 40,
        "axes.linewidth": 2,
        "lines.linewidth": 5,
    },
    font_scale=3.5,
    palette=sns.color_palette("Set2")
)
c = ["#007fff", "#ff3616", "#138d75", "#7d3c98", "#fbea6a"]  # Blue, Red, Green, Purple, Yellow


import utilities

from Helix import Helix

import matplotlib
matplotlib.use("Qt5Agg")


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return rho, phi


def center_atoms(atoms, center):
    x = center[0]
    y = center[1]
    z = center[2]
    # Centering atoms around given atom
    for idx, atom in enumerate(atoms):
        atoms[idx].position[0] = atom.position[0] - x
        atoms[idx].position[1] = atom.position[1] - y
        atoms[idx].position[2] = atom.position[2] - z

    return atoms


def print_jmol_str(line_values, center):
    file = "analyzed/diffp_2me_homo-1"
    print("*"*25)
    print(f"Writing to {file}")
    print("*"*25)

    curve_str = f"draw curve1 CURVE curve width 0.3"

    for value in line_values:
        x = value[0] + center[0]
        y = value[1] + center[1]
        z = value[2] + center[2]
        curve_str += f" {{ {x} {y} {z} }}"

    with open(f"{file}/jmol_export.spt", "a") as f:
        f.write(curve_str)

    print(curve_str)


def remove_outlier(ordered):
    # Not elegant, possibly slow, but it works
    temp = []
    for idx, value in enumerate(ordered[:, 2]):
        if idx < len(ordered[:, 2]) - 1:
            temp.append(abs(value - ordered[idx + 1, 2]))
    std = np.std(temp)
    mean = np.mean(temp)

    # It lies much further down the z-axis
    # than the rest of the points
    if not (mean - std) < temp[0] < (mean + std):
        return ordered[1:]

    # If no outliers is found, return the original array
    else:
        return ordered


center_bottom_top = np.array([2, 9, 7])
handedness = None
truncation = [None, None]

file = "./8cum_me_homo_homo/homo.cube"

ax = plt.axes(projection='3d')

radius = 1.4
limits = 3

# Check that the analysis hasn't already been done
names = file.split("/")
folder = "/".join(names[-3:-1])
print(f"foldername: {folder}")
if os.path.exists(folder):
    print(f"Found existing data files in {folder}")
    planes = np.load(folder + "/planes.npy", allow_pickle=True)
    atoms, _, _, center = np.load(
        folder + "/atom_info.npy", allow_pickle=True
    )
    xyz_vec = np.load(folder + "/xyz_vec.npy", allow_pickle=True)
else:
    atoms, all_info, xyz_vec = utilities.read_cube(file)

    # Sort the data after z-value
    all_info = all_info[all_info[:, 2].argsort()]

    # Center of the molecule is chosen to be Ru
    # center = atoms[3].position
    center = atoms[center_bottom_top[0]].position
    all_info[:, :3] = all_info[:, :3] - center
    atoms = center_atoms(atoms, center)

    planes = []
    plane = []
    prev_coord = all_info[0]

    for coordinate in tqdm(all_info, desc="Finding planes.."):
        if np.equal(coordinate[2], prev_coord[2]):
            # we're in the same plane so add the coordinate
            plane.append([coordinate[0],
                          coordinate[1],
                          coordinate[2],
                          coordinate[3]])

        else:
            plane = np.array(plane)
            # Drop coordinates with isovalues == 0.0
            plane = plane[np.where(plane[:, 3] != 0.0)]

            if plane.size != 0:
                planes.append(plane)

            plane = []

        prev_coord = coordinate
    planes = np.array(planes)

mean_z = []
ordered = []
all_r = []
bottom_carbon = atoms[center_bottom_top[1]].position
top_carbon = atoms[center_bottom_top[2]].position
print('Cleaning values..')
for idx, plane in enumerate(planes):
    if top_carbon[2] > plane[0, 2] > bottom_carbon[2]:
        if idx < len(planes) - 1:

            # Uncomment to find points with the most positive isovalue
            # Rare cases there might be the same maximum at two locations
            # That's I just take the first one with [0][0]
            maximum = np.amax(plane[:, 3])
            max_index = np.where(plane[:, 3] == maximum)[0][0]
            next_plane = planes[idx + 1]
            next_maximum = np.amax(next_plane[:, 3])
            next_index = np.where(next_plane[:, 3] == next_maximum)[0][0]

            # Uncomment to find points with the most negative isovalue
            # minimum = np.amin(plane[:, 3])
            # min_index = np.where(plane[:, 3] == minimum)
            # next_plane = planes[idx + 1]
            # next_minimum = np.amin(next_plane[:, 3])
            # next_index = np.where(next_plane[:, 3] == next_minimum)

            current_iso_idx = max_index
            next_iso_idx = next_index

            # Check if point is within certain radius of the helical axis
            if cart2pol(plane[current_iso_idx, 0], plane[current_iso_idx, 1])[0] < radius:
                current_x = plane[current_iso_idx, 0].item()
                current_y = plane[current_iso_idx, 1].item()
                current_z = plane[current_iso_idx, 2].item()
                current_iso = plane[current_iso_idx, 3].item()

                next_x = next_plane[next_index, 0].item()
                next_y = next_plane[next_index, 1].item()
                next_z = next_plane[next_index, 2].item()
                next_iso = next_plane[next_iso_idx, 3].item()

                # Current point is beneath the next point
                if (current_x == next_x) & (current_y == next_y):
                    delta_z = abs(next_z - current_z)

                    # Are they direcly on top of each other?
                    if round(delta_z, 4) <= 2*round(xyz_vec[2], 4):
                        mean_z.append(current_z)

                # They are not directly on top of each other
                else:
                    ax.scatter(
                        plane[current_iso_idx, 0],
                        plane[current_iso_idx, 1],
                        plane[current_iso_idx, 2],
                        # c='purple',
                        c=c[0],
                    )
                    # To be used as an estimate of
                    # the radius when fitting the helix
                    all_r.append(
                        cart2pol(plane[current_iso_idx, 0], plane[current_iso_idx, 1])[0]
                    )

                    mean_z.append(current_z)
                    ordered.append(
                        [current_x, current_y, np.mean(mean_z), current_iso]
                    )
                    mean_z = []

        # TODO: Maybe I'm skipping the last point? Does it even matter?
        # else:
        #     prev_x = current_x
        #     prev_y = current_y
        #     prev_z = current_z
        #     prev_iso = current_iso

        #     current_x = plane[max_index, 0].item()
        #     current_y = plane[max_index, 1].item()
        #     current_z = plane[max_index, 2].item()
        #     current_iso = plane[max_index, 3].item()

        #     if cart2pol(current_x, current_y)[0] < radius:
        #         all_r.append(cart2pol(plane[max_index, 0], plane[max_index, 1])[0])

        #         if (current_x == prev_x) & (current_y == prev_y):
        #             delta_z = abs(prev_z - current_z)

        #             # Are they directly on top of each other?
        #             if round(delta_z, 4) <= 2*round(z_vec, 4):
        #                 mean_z.append(current_z)
        #                 ordered.append([current_x,
        #                                 current_y,
        #                                 np.mean(mean_z),
        #                                 current_iso])

        #         # They are not directly on top of each other
        #         else:

        #             mean_z.append(current_z)
        #             ordered.append([current_x,
        #                             current_y,
        #                             np.mean(mean_z),
        #                             current_iso])
        #             mean_z = []
ordered = np.array(ordered)
mean_radius = np.mean(all_r)

# Check if the first point is an outlier
ordered = remove_outlier(ordered)
# ordered, mean_radius = np.load("orbital_16_helix.npy", allow_pickle=True)
# ax.plot([0, ordered[0, 0]], [0, ordered[0, 1]], [0, 0])
# Line that connects each data point
# ax.plot(
#     ordered[truncation[0]:truncation[1], 0],
#     ordered[truncation[0]:truncation[1], 1],
#     ordered[truncation[0]:truncation[1], 2],
#     color='blue'
# )

print('Fitting datapoints to helix..')

helix = Helix(
    ordered[0:, :3],
    fitting_method='ampgo',
    radius=mean_radius,
    handedness=handedness,
    truncation=truncation,
)
out = helix.fit_helix()
fitted_values = helix.fitted_values

# print_jmol_str(fitted_values, center)

print('RMSD: {}'.format(helix.RMSD))
print(out)
print('handedness: {}'.format(helix.handedness))
delta_z = helix.get_statistics()
print('std: {}'.format(np.std(delta_z)))
print('mean: {}'.format(np.mean(delta_z)))
print(f'p-value: {helix.p_value}')

ax.plot(
    fitted_values[:, 0],
    fitted_values[:, 1],
    fitted_values[:, 2],
)

ax.plot((0, helix.a[0]), (0, helix.a[1]), (0, helix.a[2]))
ax.plot((0, helix.v[0]), (0, helix.v[1]), (0, helix.v[2]))
ax.plot((0, helix.w[0]), (0, helix.w[1]), (0, helix.w[2]), color='black')


print('Plotting atoms..')
for atom in atoms:
    if atom.symbol == 'C':
        ax.scatter(
            atom.position[0],
            atom.position[1],
            atom.position[2],
            c='black'
        )

    if atom.symbol == 'Ru':
        ax.scatter(
            atom.position[0],
            atom.position[1],
            atom.position[2],
            c='turquoise'
        )

    # if atom.symbol == 'P':
    #     ax.scatter3D(atom.position[0],
    #                  atom.position[1],
    #                  atom.position[2],
    #                  c='orange')

ax.set_xlim([-limits, limits])
ax.set_ylim([-limits, limits])

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
