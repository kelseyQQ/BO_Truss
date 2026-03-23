import numpy as np
import sys
import io
import contextlib
from pathlib import Path

sys.path.append('./pyJive/')

from utils import proputils as pu
import main


def snap_area_to_discrete(area_values):
    AREA_SET = np.arange(2.00, 21.75 + 0.001, 0.25)
    area_values = np.asarray(area_values)
    idx = np.abs(area_values[..., None] - AREA_SET).argmin(axis=-1)
    return AREA_SET[idx]


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
        area = areas[cross_section]
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
    for _, elem in elements_dict.items():
        weight += 0.1 * elem['length'] * elem['area']
    return weight


def compute_constraint_1(elements_dict):
    constr_value = []
    for _, elem in elements_dict.items():
        constr_tens = elem['sigma'] - 20
        constr_compr = max(elem['sigma_buc'] - elem['sigma'], -20- elem['sigma'])
        constr_value.append(constr_tens)
        constr_value.append(constr_compr)
    return np.array(constr_value)


def evaluate_variables(
    x,
    geom_path='cantilever_measure_weight.geom',
    pro_path='cantilever_measure_weight.pro',
    node_idx=('2', '4', '6', '8')
):
    x = np.array(x, dtype=float).copy()
    x[:4] = snap_area_to_discrete(x[:4])

    area = x[:4]
    coords = x[4:]

    update_geom_file(geom_path, coords, node_idx)

    props = pu.parse_file(pro_path)
    props['model']['truss']['area'] = area

    with contextlib.redirect_stdout(io.StringIO()):
        globdat = main.jive(props)

    N = globdat['tables']['stress'][0].get_all_values()[::2]

    nodes_dict, elements_dict = parse_geometry_file(geom_path, area, N)
    print("\nMember stresses:")
    for i, elem in elements_dict.items():
        print(f"Member {i}: sigma = {elem['sigma']:.6f}, sigma_buc = {elem['sigma_buc']:.6f}")

    weight = get_weight(elements_dict)
    constr_value = compute_constraint_1(elements_dict)
    feasible = np.all(constr_value <= 0)

    return {
        "variables_used": x,
        "areas_used": area,
        "coords_used": coords,
        "weight": weight,
        "constraints": constr_value,
        "feasible": feasible,
        "nodes_dict": nodes_dict,
        "elements_dict": elements_dict,
    }


if __name__ == "__main__":
    variables = np.array([
       12.5    ,     17.5       ,   5.75   ,      3.75    ,   907.34998641	,
         180.15535433, 637.04070068 ,141.71185048, 408.07633436  ,93.89049003, 198.81458272 , 29.38926524

    ])

    result = evaluate_variables(variables)

    print("variables used =", result["variables_used"])
    print("areas used     =", result["areas_used"])
    print("weight         =", result["weight"])
    print("feasible       =", result["feasible"])
    print("constraints    =", result["constraints"])