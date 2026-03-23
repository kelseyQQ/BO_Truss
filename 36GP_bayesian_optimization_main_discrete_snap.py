import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import sys
sys.path.append('./pyJive/')

from utils import proputils as pu
import main

import io
import contextlib

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.random.seed(23)

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


def feasibility_probability(x, gp, threshold):
    mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
    prob = norm.cdf((threshold - mu) / sigma)
    # print(f"mu: {mu}, sigma: {sigma}, threshold: {threshold}, mu-tr: {mu-threshold}, prob: {prob}")
    return prob


def EI(x, gp_Y, gp_constr, Y_samples, threshold, xi, best_y):
    mu, sigma = gp_Y.predict(x.reshape(1, -1), return_std=True)
    mu, sigma = mu[0], sigma[0]
    best = best_y

    z = (best - mu - xi) / sigma
    ei = (best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)

    prob_list = []
    prob = 1
    for i in range(36):
        with contextlib.redirect_stdout(io.StringIO()):
            p_i = feasibility_probability(x, gp_constr[i], threshold[i])
        prob_list.append(p_i)
        prob *= p_i

    print('')
    print(f"mu: {mu}, sigma: {sigma}, best: {best}, xi: {xi}, z: {z}, ei: {ei}, prob: {prob}, EI: {ei * prob}")

    return -ei * prob**2, prob_list


def adaptive_xi_linear(iteration, max_iterations, xi_max=3, xi_min=3):
    return xi_max - (xi_max - xi_min) * (iteration / max_iterations)


def bayesian_optimization(n_iter=100, improvement_threshold=1e-3, patience=50, data_path="initial_data_36.npz"):
    n_dim = 12

    print('Loading initial data...')
    data = np.load(data_path)

    X_samples_unscaled = data["X_samples_unscaled"]
    Y_samples_unscaled = data["Y_samples_unscaled"]
    constr_samples_unscaled = data["constr_samples_unscaled"]
    bounds_unscaled = data["bounds_unscaled"]

    threshold_unscaled = [0] * 36

    print(X_samples_unscaled.shape)
    print(Y_samples_unscaled.shape)
    print(constr_samples_unscaled.shape)

    valid_idx = np.all(constr_samples_unscaled <= np.array([0] * 36)[:, np.newaxis], axis=0)
    print(f"valid: {np.count_nonzero(valid_idx)}")
    valid_variables = X_samples_unscaled[valid_idx]
    valid_weights = Y_samples_unscaled[valid_idx]
    valid_constraints = constr_samples_unscaled[:, valid_idx]

    print("Valid variables:")
    for i, vars_ in enumerate(valid_variables):
        print(f"\nValid sample {i+1}:")
        print("variables =", vars_)
        print("weight    =", valid_weights[i])
        print("max constr=", np.max(valid_constraints[:, i]))
        print("constraints =", valid_constraints[:, i])

    max_per_sample = np.max(constr_samples_unscaled, axis=0)
    indices_smallest = np.argsort(max_per_sample)[:100]

    X_samples_unscaled = X_samples_unscaled[indices_smallest]
    Y_samples_unscaled = Y_samples_unscaled[indices_smallest]
    constr_samples_unscaled = constr_samples_unscaled[:, indices_smallest]

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    constr_scaler = [StandardScaler() for _ in range(36)]

    X_samples = input_scaler.fit_transform(X_samples_unscaled)
    Y_samples = output_scaler.fit_transform(Y_samples_unscaled.reshape(-1, 1)).flatten()

    constr_samples = []
    for i in range(36):
        constr_samples.append(
            constr_scaler[i].fit_transform(constr_samples_unscaled[i].reshape(-1, 1)).flatten()
        )

    bounds = input_scaler.transform(bounds_unscaled.T).T

    threshold = []
    for i in range(36):
        threshold.append(
            constr_scaler[i].transform(np.array(threshold_unscaled[i]).reshape(-1, 1))[0, 0]
        )

    kernel_Y = Matern(length_scale=[10] * 12, nu=2.5, length_scale_bounds=(0.1, 1000)) + RBF(length_scale=10)
    kernel_constr = [
        Matern(length_scale=[1.0] * 12, length_scale_bounds=(0.1, 100), nu=1.5)
        for _ in range(36)
    ]

    gp_Y = GaussianProcessRegressor(kernel=kernel_Y, alpha=0)
    gp_constr = [GaussianProcessRegressor(kernel=kernel_constr[i], alpha=0) for i in range(36)]

    valid_idx = np.all(constr_samples_unscaled <= 0, axis=0)
    if not np.any(valid_idx):
        raise RuntimeError("No feasible initial sample found.")

    valid_initial_indices = np.where(valid_idx)[0]
    best_init_idx = valid_initial_indices[np.argmin(Y_samples_unscaled[valid_initial_indices])]

    best_feasible_x_unscaled = X_samples_unscaled[best_init_idx].copy()
    best_feasible_y_unscaled = Y_samples_unscaled[best_init_idx].copy()
    best_feasible_constr_unscaled = constr_samples_unscaled[:, best_init_idx].copy()

    best_y = output_scaler.transform(np.array(best_feasible_y_unscaled).reshape(-1, 1))[0, 0]
    best_x = input_scaler.transform(best_feasible_x_unscaled.reshape(1, -1))[0]

    y_next_list = []
    best_y_history = []
    no_improvement_count = 0

    constr_plot = []
    prob_plot = []

    for iteration in range(n_iter):
        print('')
        print('')
        print(f'Iteration {iteration + 1}')
        print('Fitting GPs...')

        if iteration % 5 == 0:
            print("Optimizing hyperparameters...")
            gp_Y.optimizer = 'fmin_l_bfgs_b'
            gp_Y.fit(X_samples, Y_samples)
            for i in range(36):
                gp_constr[i].optimizer = 'fmin_l_bfgs_b'
                gp_constr[i].fit(X_samples, constr_samples[i])
        else:
            optimized_kernel_Y = gp_Y.kernel_
            optimized_kernel_constr = [gp_constr[i].kernel_ for i in range(36)]

            gp_Y = GaussianProcessRegressor(kernel=optimized_kernel_Y, alpha=0, optimizer=None)
            gp_constr = [
                GaussianProcessRegressor(kernel=optimized_kernel_constr[i], alpha=0, optimizer=None)
                for i in range(36)
            ]

            gp_Y.fit(X_samples, Y_samples)
            for i in range(36):
                gp_constr[i].fit(X_samples, constr_samples[i])

            print('Finding next point...')

        xi = adaptive_xi_linear(iteration, n_iter, xi_max=-20, xi_min=-2)

        def acquisition(x):
            return EI(x, gp_Y, gp_constr, Y_samples, threshold, xi, best_y)

        with contextlib.redirect_stdout(io.StringIO()):
            result = differential_evolution(
                lambda x: acquisition(x)[0],
                bounds=bounds,
                strategy='best1bin',
                popsize=5,
                maxiter=10,
                tol=1e-6,
                polish=True
            )

        x_next = result.x
        x_next_unscaled = input_scaler.inverse_transform(x_next.reshape(-1, n_dim))[0]
        x_next_unscaled[:4] = snap_area_to_discrete(x_next_unscaled[:4])

        x_next = input_scaler.transform(x_next_unscaled.reshape(1, -1))[0]

        output_next_unscaled = finite_element_solver(x_next_unscaled)
        y_next_unscaled = output_next_unscaled[0]
        constr_next_unscaled = output_next_unscaled[1].T
        is_feasible = np.all(constr_next_unscaled <= 0)

        if is_feasible and y_next_unscaled < best_feasible_y_unscaled:
            best_feasible_x_unscaled = x_next_unscaled.copy()
            best_feasible_y_unscaled = y_next_unscaled
            best_feasible_constr_unscaled = constr_next_unscaled.copy()
            best_y = output_scaler.transform(np.array(best_feasible_y_unscaled).reshape(-1, 1))[0, 0]
            best_x = input_scaler.transform(best_feasible_x_unscaled.reshape(1, -1))[0]

        prob_plot.append(acquisition(x_next)[1])

        y_next = output_scaler.transform(np.array(y_next_unscaled).reshape(-1, 1))[0]
        constr_next = []
        for i in range(36):
            constr_next.append(
                constr_scaler[i].transform(np.array(constr_next_unscaled[i]).reshape(-1, 1))[0]
            )

        X_samples = np.vstack((X_samples, x_next))
        Y_samples = np.append(Y_samples, y_next)

        for i in range(36):
            constr_samples[i] = np.append(constr_samples[i], constr_next[i])

        valid_idx = np.all(np.array(constr_samples) <= np.array(threshold)[:, np.newaxis], axis=0)
        if not np.any(valid_idx):
            raise RuntimeError("No feasible sample found during optimization.")

        # best_idx_local = np.argmin(Y_samples[valid_idx])
        # best_y = np.min(Y_samples[valid_idx])
        # best_x = X_samples[valid_idx][best_idx_local]

        # y_next_list.append(y_next_unscaled)
        # best_y_unscaled = output_scaler.inverse_transform(np.array(best_y).reshape(-1, 1))[0][0]
        # best_y_history.append(best_y_unscaled)
        y_next_list.append(y_next_unscaled)
        best_y_history.append(best_feasible_y_unscaled)

        # print('')
        # print("Optimized Kernel Y:", gp_Y.kernel_)
        # for i in range(36):
        #     print(f"Kernel {i+1}: {gp_constr[i].kernel_}")
        print('')
        print(f'time: {time.time() - start_time}')
        print(f'variables: {x_next_unscaled}')
        print(f'best feasible weight: {best_feasible_y_unscaled}')
        print(f'weight: {y_next_unscaled}')
        print(f'feasibility: {"feasible" if is_feasible else "NOT"}')
        # print(f'constr: {np.mean(constr_next_unscaled)}')
        # print(f'constr: {constr_next_unscaled}')
        print('')

        constr_plot.append(constr_next_unscaled)
        constr_plot_array = np.array(constr_plot)
        prob_plot_array = np.array(prob_plot)

        # Per-iteration bar plots are intentionally disabled
        # fig, ax = plt.subplots(nrows=2, figsize=(10, 5))
        # ax[0].bar(range(1, 37), np.array(prob_plot_array[-1]).reshape(36))
        # ax[1].bar(range(1, 37), constr_next_unscaled)
        # plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(best_y_history, marker='o', label='Best weight so far')

    # 標出最後一個 best weight 數值
    last_x = len(best_y_history) - 1
    last_y = best_y_history[-1]
    plt.annotate(f'{last_y:.2f}',
                xy=(last_x, last_y),
                xytext=(8, 8),
                textcoords='offset points')

    plt.xlabel('Iteration')
    plt.ylabel('Best weight')
    plt.title('Best Weight vs Iteration')
    plt.legend()
    plt.savefig("best_weight_vs_iteration_36GP.png", dpi=300, bbox_inches="tight")
    plt.close()

    # best_x_unscaled = input_scaler.inverse_transform(best_x.reshape(1, -1))[0]
    # best_x_unscaled[:4] = snap_area_to_discrete(best_x_unscaled[:4])

    # return best_x_unscaled, best_y_unscaled
    print("\nFinal FE check of returned best solution:")
    final_weight_check, final_constr_check = finite_element_solver(best_feasible_x_unscaled)
    print("returned x      =", best_feasible_x_unscaled)
    print("returned weight =", final_weight_check)
    print("feasible        =", np.all(final_constr_check <= 0))
    print("max constraint  =", np.max(final_constr_check))

    return best_feasible_x_unscaled, best_feasible_y_unscaled


for i in range(1):
    print(f'Run: {i+1}')

    start_time = time.time()
    variables, weight = bayesian_optimization(data_path="initial_data_36.npz")
    end_time = time.time()

    print('Best values')
    print(f'time: {end_time - start_time}')
    print(f'variables: {variables}')
    print(f'weight: {weight}')
    print('')