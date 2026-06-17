import numpy as np
import csv
import io
import struct


def export_csv(x, t, U):
    output = io.StringIO()
    writer = csv.writer(output)
    nt, nx = U.shape
    writer.writerow(['x', 't', 'u'])
    for i in range(nt):
        for j in range(nx):
            writer.writerow([f'{x[j]:.8f}', f'{t[i]:.8f}', f'{U[i, j]:.8f}'])
    return output.getvalue()


def export_vtk(x, t, U, field_name='u'):
    nx = len(x)
    nt = len(t)
    total_points = nx * nt
    n_cells = (nx - 1) * (nt - 1)

    points = np.zeros((total_points, 3), dtype=np.float32)
    for i in range(nt):
        for j in range(nx):
            idx = i * nx + j
            points[idx, 0] = x[j]
            points[idx, 1] = t[i]
            points[idx, 2] = 0.0

    cells = []
    for i in range(nt - 1):
        for j in range(nx - 1):
            p0 = i * nx + j
            p1 = i * nx + j + 1
            p2 = (i + 1) * nx + j + 1
            p3 = (i + 1) * nx + j
            cells.append((p0, p1, p2, p3))

    scalars = U.flatten().astype(np.float32)

    lines = []
    lines.append('# vtk DataFile Version 3.0')
    lines.append('PINN Solver Output')
    lines.append('ASCII')
    lines.append('DATASET UNSTRUCTURED_GRID')
    lines.append(f'POINTS {total_points} float')

    for p in points:
        lines.append(f'{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}')

    lines.append(f'CELLS {n_cells} {n_cells * 5}')
    for c in cells:
        lines.append(f'4 {c[0]} {c[1]} {c[2]} {c[3]}')

    lines.append(f'CELL_TYPES {n_cells}')
    for _ in range(n_cells):
        lines.append('9')

    lines.append(f'POINT_DATA {total_points}')
    lines.append(f'SCALARS {field_name} float 1')
    lines.append('LOOKUP_TABLE default')
    for s in scalars:
        lines.append(f'{s:.8f}')

    return '\n'.join(lines) + '\n'


def export_vtk_3d(x, t, U, field_name='u'):
    nx = len(x)
    nt = len(t)
    total_points = nx * nt
    n_cells = (nx - 1) * (nt - 1)

    points = np.zeros((total_points, 3), dtype=np.float32)
    for i in range(nt):
        for j in range(nx):
            idx = i * nx + j
            points[idx, 0] = x[j]
            points[idx, 1] = t[i]
            points[idx, 2] = U[i, j]

    cells = []
    for i in range(nt - 1):
        for j in range(nx - 1):
            p0 = i * nx + j
            p1 = i * nx + j + 1
            p2 = (i + 1) * nx + j + 1
            p3 = (i + 1) * nx + j
            cells.append((p0, p1, p2, p3))

    scalars = U.flatten().astype(np.float32)

    lines = []
    lines.append('# vtk DataFile Version 3.0')
    lines.append('PINN Solver Output - 3D Surface')
    lines.append('ASCII')
    lines.append('DATASET UNSTRUCTURED_GRID')
    lines.append(f'POINTS {total_points} float')

    for p in points:
        lines.append(f'{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}')

    lines.append(f'CELLS {n_cells} {n_cells * 5}')
    for c in cells:
        lines.append(f'4 {c[0]} {c[1]} {c[2]} {c[3]}')

    lines.append(f'CELL_TYPES {n_cells}')
    for _ in range(n_cells):
        lines.append('9')

    lines.append(f'POINT_DATA {total_points}')
    lines.append(f'SCALARS {field_name} float 1')
    lines.append('LOOKUP_TABLE default')
    for s in scalars:
        lines.append(f'{s:.8f}')

    return '\n'.join(lines) + '\n'
