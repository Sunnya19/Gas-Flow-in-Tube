from pathlib import Path
from typing import List
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None

try:
    import vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    vtk = None


def save_particles_vtp(
    positions: np.ndarray,
    velocities: np.ndarray,
    radius: float,
    output_path: Path,
) -> None:
    if not HAS_PYVISTA:
        raise ImportError(
            "pyvista is required for VTK export."
        )

    # Validate input shapes
    if positions.shape[1] != 2:
        raise ValueError(
            f"positions must have shape (N, 2), got {positions.shape}"
        )
    if velocities.shape != positions.shape:
        raise ValueError(
            f"velocities shape {velocities.shape} must match positions shape {positions.shape}"
        )

    n_particles = positions.shape[0]

    # Create 3D points (z = 0)
    points_3d = np.zeros((n_particles, 3), dtype=np.float64)
    points_3d[:, 0:2] = positions

    # Create 3D velocity vectors (vz = 0)
    velocity_3d = np.zeros((n_particles, 3), dtype=np.float64)
    velocity_3d[:, 0:2] = velocities

    # Compute speed
    speed = np.linalg.norm(velocities, axis=1)

    # Create PolyData
    mesh = pv.PolyData(points_3d)

    # Add scalar and vector fields
    mesh["particle_id"] = np.arange(n_particles, dtype=np.int32)
    mesh["velocity"] = velocity_3d
    mesh["speed"] = speed.astype(np.float64)
    mesh["radius"] = np.full(n_particles, radius, dtype=np.float64)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as .vtp
    mesh.save(str(output_path))
    print(f"Saved VTK particle data to {output_path}")


def save_pvd_file(
    frame_paths: List[Path],
    times: List[float],
    output_path: Path,
) -> None:
    if len(frame_paths) != len(times):
        raise ValueError(
            f"Number of frame paths ({len(frame_paths)}) must match "
            f"number of times ({len(times)})"
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate XML content
    xml_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>'
    ]

    for frame_path, time in zip(frame_paths, times):
        # Use only the filename (relative to .pvd location)
        filename = frame_path.name
        xml_lines.append(
            f'    <DataSet timestep="{time:.6f}" group="" part="0" file="{filename}"/>'
        )

    xml_lines.extend([
        '  </Collection>',
        '</VTKFile>'
    ])

    xml_content = '\n'.join(xml_lines)

    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)

    print(f"Saved PVD collection file to {output_path}")
