"""
Tests for VTK writer module.
"""
import tempfile
import numpy as np
from pathlib import Path

try:
    from src.io.vtk_writer import save_particles_vtp, save_pvd_file
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from src.io.vtk_writer import save_particles_vtp, save_pvd_file


def test_save_particles_vtp_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy particle data
        n_particles = 3
        positions = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 4.0]
        ], dtype=np.float64)
        velocities = np.array([
            [1.0, 0.0],
            [0.0, -1.0],
            [0.5, 0.5]
        ], dtype=np.float64)
        radius = 0.1
        
        output_path = Path(tmpdir) / "test_particles.vtp"
        
        # Call the function
        save_particles_vtp(
            positions=positions,
            velocities=velocities,
            radius=radius,
            output_path=output_path,
        )
        
        # Check that file exists and is not empty
        assert output_path.exists(), f"File {output_path} was not created"
        assert output_path.stat().st_size > 0, f"File {output_path} is empty"
        
        # Optional: verify that the file contains expected strings (VTK header)
        with open(output_path, 'rb') as f:
            content = f.read(100).decode('ascii', errors='ignore')
            assert 'VTK' in content or 'PolyData' in content, \
                f"File does not appear to be a VTK file: {content[:100]}"
        
        print(f"✓ test_save_particles_vtp_creates_file passed")


def test_save_pvd_file_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create dummy frame paths
        frame_paths = [
            tmpdir_path / "particles_000000.vtp",
            tmpdir_path / "particles_000020.vtp",
            tmpdir_path / "particles_000040.vtp",
        ]
        times = [0.0, 0.2, 0.4]
        
        output_path = tmpdir_path / "collection.pvd"
        
        # Call the function
        save_pvd_file(
            frame_paths=frame_paths,
            times=times,
            output_path=output_path,
        )
        
        # Check that file exists and is not empty
        assert output_path.exists(), f"File {output_path} was not created"
        assert output_path.stat().st_size > 0, f"File {output_path} is empty"
        
        # Read file content and verify XML structure
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required XML tags
        assert '<?xml' in content, "Missing XML declaration"
        assert '<VTKFile type="Collection"' in content, "Missing VTKFile Collection tag"
        assert '<DataSet timestep="0.0"' in content, "Missing DataSet tag with timestep"
        assert 'particles_000000.vtp' in content, "Missing expected filename"
        
        print(f"✓ test_save_pvd_file_creates_file passed")


def test_save_particles_vtp_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.vtp"
        
        # Test mismatched shapes
        positions = np.zeros((3, 2))
        velocities = np.zeros((4, 2))  # different number of particles
        
        try:
            save_particles_vtp(positions, velocities, 0.1, output_path)
            assert False, "Expected ValueError for shape mismatch"
        except ValueError as e:
            assert "shape" in str(e).lower() or "match" in str(e).lower()
            print(f"✓ test_save_particles_vtp_validation passed (shape mismatch)")
        
        # Test incorrect position dimensions
        positions_bad = np.zeros((3, 3))  # should be (N, 2)
        velocities_bad = np.zeros((3, 3))
        
        try:
            save_particles_vtp(positions_bad, velocities_bad, 0.1, output_path)
            assert False, "Expected ValueError for incorrect position shape"
        except ValueError as e:
            assert "shape" in str(e).lower()
            print(f"✓ test_save_particles_vtp_validation passed (dimension mismatch)")


def test_save_pvd_file_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.pvd"
        
        # Different lengths
        frame_paths = [Path("a.vtp"), Path("b.vtp")]
        times = [0.0]  # only one time
        
        try:
            save_pvd_file(frame_paths, times, output_path)
            assert False, "Expected ValueError for length mismatch"
        except ValueError as e:
            assert "length" in str(e).lower() or "match" in str(e).lower()
            print(f"✓ test_save_pvd_file_validation passed")


if __name__ == "__main__":
    # Run tests
    test_save_particles_vtp_creates_file()
    test_save_pvd_file_creates_file()
    test_save_particles_vtp_validation()
    test_save_pvd_file_validation()
    print("\nAll VTK writer tests passed!")