import numpy as np
import matplotlib.pyplot as plt

from .plotutils import figsave, _pop_save_path_aliases, _raise_unrecognized_kwargs


def divergence_plot(
    r_vectors,
    r_center=None,
    v_center=None,
    title='Position Errors Projected onto Velocity Plane',
    show=True,
    save_path=None,
    **save_kwargs,
):
    """
    Plot position errors projected onto a plane defined by velocity vector.
    
    Parameters
    ----------
    r_vectors : np.ndarray
        Array of shape (N, 3) containing position vectors
    r_center : np.ndarray, optional
        Reference position vector (3,). If None, uses median of r_vectors.
    v_center : np.ndarray, optional
        Reference velocity vector (3,) that defines the plane normal.
        If None, raises an error.
    title : str, optional
        Plot title
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "divergence_plot")

    # Use median if r_center not provided
    if r_center is None:
        r_center = np.median(r_vectors, axis=0)
    
    # Calculate errors from center
    errors = r_vectors - r_center
    
    # Check that v_center is provided
    if v_center is None:
        raise ValueError("v_center must be provided")
    
    # Compute NTW coordinate system basis vectors
    t_hat = v_center / np.linalg.norm(v_center)
    w_hat = np.cross(r_center, v_center)
    w_hat = w_hat / np.linalg.norm(w_hat)
    n_hat = np.cross(t_hat, w_hat)
    
    # Project errors onto the plane perpendicular to velocity
    errors_projected = errors - np.outer(np.dot(errors, t_hat), t_hat)
    
    # Calculate tangential (along-track) component
    t_component = np.dot(errors, t_hat)
    
    # Convert tangential error to timing difference
    v_magnitude = np.linalg.norm(v_center)  # orbital velocity magnitude in m/s
    timing_difference = t_component / v_magnitude  # timing difference in seconds
    
    # Convert 3D projected points to 2D coordinates using N and W basis
    x_2d = np.dot(errors_projected, n_hat)
    y_2d = np.dot(errors_projected, w_hat)
    
    # Calculate radial norms for statistics
    r_norm = np.sqrt(x_2d**2 + y_2d**2)
    print(f"\nProjected position offsets | "
          f"mean radius = {np.mean(r_norm):.3f} m, "
          f"std radius  = {np.std(r_norm):.3f} m, "
          f"max radius  = {np.max(r_norm):.3f} m")
    print(f"Tangential offsets | "
          f"mean = {np.mean(t_component):.3f} m, "
          f"std = {np.std(t_component):.3f} m, "
          f"range = [{np.min(t_component):.3f}, {np.max(t_component):.3f}] m")
    print(f"Timing differences | "
          f"mean = {np.mean(timing_difference):.3f} s, "
          f"std = {np.std(timing_difference):.3f} s, "
          f"range = [{np.min(timing_difference):.3f}, {np.max(timing_difference):.3f}] s")
    
    # Create scientific plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use color mapping for timing difference
    scatter = ax.scatter(x_2d, y_2d, c=timing_difference, s=80, 
                        cmap='RdYlBu_r', zorder=100, 
                        edgecolors='black', linewidths=0.8,
                        alpha=0.8)
    
    # Add colorbar with timing units
    cbar = fig.colorbar(scatter, ax=ax, label='Timing difference [seconds]', 
                       shrink=0.8, pad=0.02)
    
    # Plot center point
    ax.scatter(0, 0, c='red', s=150, marker='x', zorder=101, 
              linewidths=3, label='Reference center')
    
    # Add circles for reference (not filled bullseye)
    max_error = max(np.max(np.abs(x_2d)), np.max(np.abs(y_2d)))
    if max_error > 0:
        circle_radii = np.linspace(max_error/4, max_error, 4)
        theta = np.linspace(0, 2 * np.pi, 100)
        for radius in circle_radii:
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            ax.plot(x_circle, y_circle, 'k--', alpha=0.2, linewidth=0.8)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Normal (N) error [m]', fontsize=12)
    ax.set_ylabel('Cross-track (W) error [m]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        figsave(fig, save_path)
    if show:
        plt.show()
    
    return fig

# Usage examples:
# divergence_plot(r_vectors, v_center=velocity_vector)  # Uses median r_center
# divergence_plot(r_vectors, r_center=my_r, v_center=my_v)  # Uses provided centers
