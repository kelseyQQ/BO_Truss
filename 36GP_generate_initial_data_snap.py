import numpy as np
import time
from scipy.stats.qmc import LatinHypercube

import sys
sys.path.append('./pyJive/')

from utils import proputils as pu
import main

import io
import contextlib


def snap_area_to_discrete(area_values):
    AREA_SET = np.arange(2.00, 21.75 + 0.001, 0.25)
    area_values = np.asarray(area_values)
    idx = np.abs(area_values[..., None] - AREA_SET).argmin(axis=-1)
    return AREA_SET[idx]


def toggle_plot(new_value):
    filename = 'cantilever_36GP.pro'
    with open(filename, 'r') as file:
        lines = file.readlines()

    with open(filename, 'w') as file:
        for line in lines:
            if 'enablePlot' in line:
                line = f'  enablePlot = {"True" if new_value else "False"};\n'
            file.write(line)


def update_geom_file(geom_path, coords, node_idx):
    x_coords = coords[0::2]
    y_coords = coords[1::2]

    with open(geom_path, 'r') as file:
        lines = file.readlines()

    j = 0
    for i in range(len(lines)):
        if lines[i][0] in node_idx:
            lines[i] = f'{i-1} {x_coords[j]} {y_coords[j]}\n'
            j += 1
        if lines[i][0] == 'm':
            break

    with open(geom_path, 'w') as file:
        file.writelines(lines)


def parse_geometry_file(file_path, areas, N):
    with open(file_path, "r") as file:
        lines = file.readlines()

    split_index = None
    for i, line in enumerate(lines):
        if line.strip() == "":
            split_index = i
            break

    nodes_data = []
    for line in lines[:split_index]:
        parts = line.strip().split()
        if len(parts) == 3:
            node_index, x, y = map(float, parts)
            nodes_data.append((int(node_index), x, y))

    elements_data = []
    for line in lines[split_index + 1:]:
        parts = line.strip().split()
        if len(parts) == 4:
            node_1, node_2, number, cross_section = map(float, parts)
            elements_data.append((int(node_1), int(node_2), int(number), int(cross_section)))

    nodes_dict = {node[0]: {"x": node[1], "y": node[2]} for node in nodes_data}
    elements_dict = {}

    for i, elem in enumerate(elements_data):
        node_1, node_2, cross_section = elem[0], elem[1], elem[3]

        x1, y1 = nodes_dict[node_1]["x"], nodes_dict[node_1]["y"]
        x2, y2 = nodes_dict[node_2]["x"], nodes_dict[node_2]["y"]

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        area = areas[elem[3]]
        sigma = N[i][0] / area
        sigma_buc = -4 * 1e4 * area / length**2

        elements_dict[i] = {
            "node_1": node_1,
            "node_2": node_2,
            "area": area,
            "length": length,
            "sigma": sigma,
            "sigma_buc": sigma_buc,
        }

    return nodes_dict, elements_dict


def get_weight(elements_dict):
    weight = 0
    for elem_id, elem in elements_dict.items():
        weight += 0.1 * elem['length'] * elem['area']
    return weight


def compute_constraint_1(elements_dict):
    """
    36GP version:
    each member contributes 2 constraints:
    1) tension: sigma - 25
    2) compression/buckling: max(sigma_buc - sigma, -25 - sigma)
    """
    constr_value = []
    for elem_id, elem in elements_dict.items():
        constr_tens = elem['sigma'] - 20
        constr_compr = max(elem['sigma_buc'] - elem['sigma'], -20- elem['sigma'])
        constr_value.append(constr_tens)
        constr_value.append(constr_compr)
    return np.array(constr_value)


def finite_element_solver(x):
    x = np.array(x, dtype=float).copy()
    x[:4] = snap_area_to_discrete(x[:4])

    area = x[:4]
    coords = x[4:]
    node_idx = ['2', '4', '6', '8']

    update_geom_file('cantilever_36GP.geom', coords, node_idx)

    props = pu.parse_file('cantilever_36GP.pro')
    props['model']['truss']['area'] = area

    with contextlib.redirect_stdout(io.StringIO()):
        globdat = main.jive(props)

    N = globdat['tables']['stress'][0].get_all_values()[::2]

    nodes_dict, elements_dict = parse_geometry_file('cantilever_36GP.geom', area, N)

    weight = get_weight(elements_dict)
    constr_value = compute_constraint_1(elements_dict)

    return weight, constr_value


def generate_lhs_samples(num_samples, bounds):
    dimensions = len(bounds)
    sampler = LatinHypercube(d=dimensions)
    lhs_points = sampler.random(n=num_samples)

    bounds = np.array(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    scaled_points = lower_bounds + (upper_bounds - lower_bounds) * lhs_points
    return scaled_points


def generate_lhs_samples_height(num_samples, bounds=[]):
    dimensions = len(bounds)
    assert dimensions >= 2, "Number of dimensions must be at least 2."

    lower_bound, upper_bound = bounds[0][0], bounds[-1][1]

    x1 = np.random.uniform(lower_bound, upper_bound, size=num_samples)
    x5 = np.random.uniform(lower_bound, upper_bound, size=num_samples)

    x1, x5 = np.minimum(x1, x5), np.maximum(x1, x5)

    intermediate_points = np.random.uniform(x1[:, None], x5[:, None], size=(num_samples, dimensions - 2))
    all_points = np.hstack([x1.reshape(-1, 1), intermediate_points, x5.reshape(-1, 1)])
    monotonic_samples = np.sort(all_points, axis=1)

    return monotonic_samples[:, ::-1]


def generate_feasible_increasing_samples(num_samples, bounds):
    samples = []
    dims = len(bounds)
    attempts = 0
    max_attempts = num_samples * 100

    while len(samples) < num_samples and attempts < max_attempts:
        candidate = [np.random.uniform(low, high) for (low, high) in bounds]
        if all(candidate[i] > candidate[i+1] for i in range(dims - 1)):
            samples.append(candidate)
        attempts += 1

    if len(samples) < num_samples:
        raise RuntimeError(f"Only generated {len(samples)} valid samples after {attempts} attempts.")

    return np.array(samples)


def generate_initial_data(n_init=4000, save_path="initial_data_36.npz"):
    bounds_unscaled = np.array([
        [2.00, 21.75],
        [2.00, 21.75],
        [2.00, 21.75],
        [2.00, 21.75],
        [775, 1225],
        [-225, 245],
        [525, 975],
        [-225, 245],
        [275, 725],
        [-225, 245],
        [25, 475],
        [-225, 245],
    ])

    print(f'Generating {n_init} initial samples...')

    X_samples_unscaled = generate_lhs_samples(num_samples=n_init, bounds=bounds_unscaled)

    # Snap the 4 area variables to the discrete area set
    X_samples_unscaled[:, :4] = snap_area_to_discrete(X_samples_unscaled[:, :4])

    # Generate monotonic x positions
    X_samples_x_unscaled = generate_feasible_increasing_samples(
        n_init,
        bounds=[[775, 1225], [525, 975], [275, 725], [25, 475]]
    )

    X_samples_y_unscaled = generate_lhs_samples_height(
        n_init,
        bounds=[[-225, 245], [-225, 245], [-225, 245], [-225, 245]]
    )

    print(X_samples_unscaled.shape)
    print(X_samples_x_unscaled.shape)
    print(X_samples_y_unscaled.shape)

    target_columns_x = [4, 6, 8, 10]
    target_columns_y = [5, 7, 9, 11]

    X_samples_unscaled[:, target_columns_x] = X_samples_x_unscaled
    X_samples_unscaled[:, target_columns_y] = X_samples_y_unscaled

    # Reference starting design
    x_set = [21.75, 21.75, 21.75, 21.75, 1000, 0, 750, 0, 500, 0, 250, 0]
    X_samples_unscaled[0] = np.array(x_set)

    print("Running FEM for all initial samples...")
    toggle_plot(False)
    output = [finite_element_solver(x) for x in X_samples_unscaled]
    toggle_plot(True)

    Y_samples_unscaled, constr_samples_unscaled = zip(*output)
    Y_samples_unscaled = np.array(Y_samples_unscaled)
    constr_samples_unscaled = np.array(constr_samples_unscaled).T

    print("Saving data...")
    np.savez(
        save_path,
        X_samples_unscaled=X_samples_unscaled,
        Y_samples_unscaled=Y_samples_unscaled,
        constr_samples_unscaled=constr_samples_unscaled,
        bounds_unscaled=bounds_unscaled
    )

    print(f"Saved to {save_path}")
    print(f"X shape: {X_samples_unscaled.shape}")
    print(f"Y shape: {Y_samples_unscaled.shape}")
    print(f"constraint shape: {constr_samples_unscaled.shape}")


if __name__ == "__main__":
    start_time = time.time()
    generate_initial_data(n_init=4000, save_path="initial_data_36.npz")
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} s")