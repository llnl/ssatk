def divergence_gif(
    r_histories,
    times_gps,
    output_path,
    r_nominal_hist=None,
    v_nominal_hist=None,
    fps=1,
    duration=None,
    vmin=-50,
    vmax=50
):
    """
    Create an animated GIF showing position error evolution over time.
    
    Parameters
    ----------
    r_histories : np.ndarray
        Position time series, shape (n_samples, n_snapshots, 3) in meters
    times_gps : np.ndarray
        GPS times for each snapshot, shape (n_snapshots,)
    output_path : str
        Path where GIF will be saved (required)
    r_nominal_hist : np.ndarray, optional
        Nominal position time series, shape (n_snapshots, 3) in meters.
        If None, uses median of r_histories at each time step.
    v_nominal_hist : np.ndarray, optional
        Nominal velocity time series, shape (n_snapshots, 3) in m/s.
        If None, computes velocity from r_nominal_hist using finite differences.
    fps : int or float
        Frames per second (ignored if duration is provided). The default is
        intentionally slow enough for divergence growth to be readable.
    duration : float or None
        Seconds per frame. If provided, overrides fps.
    vmin, vmax : float
        Color scale limits for timing difference [seconds]
    
    Returns
    -------
    str
        Path to the created GIF file
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    import tempfile
    import numpy as np
    from ssatk.plots.write_gifs import write_gif
    
    n_samples, n_snapshots, _ = r_histories.shape
    
    # Use median if nominal history not provided
    if r_nominal_hist is None:
        r_nominal_hist = np.median(r_histories, axis=0)
    
    # Compute velocity from position if not provided
    if v_nominal_hist is None:
        v_nominal_hist = np.zeros_like(r_nominal_hist)
        dt = np.diff(times_gps)
        
        # Forward difference for first point
        v_nominal_hist[0] = (r_nominal_hist[1] - r_nominal_hist[0]) / dt[0]
        
        # Central differences for middle points
        for i in range(1, n_snapshots - 1):
            v_nominal_hist[i] = (r_nominal_hist[i+1] - r_nominal_hist[i-1]) / (dt[i-1] + dt[i])
        
        # Backward difference for last point
        v_nominal_hist[-1] = (r_nominal_hist[-1] - r_nominal_hist[-2]) / dt[-1]
    
    # Starting time is first element of time array
    t0_gps = times_gps[0]
    
    # Calculate global max error for consistent axis limits
    print("Computing global axis limits from projected errors...")
    max_nw_errors = []
    
    for snap_idx in range(n_snapshots):
        r_center = r_nominal_hist[snap_idx]
        v_center = v_nominal_hist[snap_idx]
        errors = r_histories[:, snap_idx, :] - r_center
        
        # Compute NTW coordinate system
        t_hat = v_center / np.linalg.norm(v_center)
        w_hat = np.cross(r_center, v_center)
        w_hat = w_hat / np.linalg.norm(w_hat)
        n_hat = np.cross(t_hat, w_hat)
        
        # Project errors
        errors_projected = errors - np.outer(np.dot(errors, t_hat), t_hat)
        
        # Convert to 2D
        x_2d = np.dot(errors_projected, n_hat)
        y_2d = np.dot(errors_projected, w_hat)
        
        # Track max in N-W plane
        max_nw_errors.append(max(np.max(np.abs(x_2d)), np.max(np.abs(y_2d))))
    
    all_max = max(max_nw_errors)
    print(f"Global max N-W error: {all_max:.2f} m")
    
    # Create temporary directory for frame images
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    # Generate frames for each snapshot
    for snap_idx in range(n_snapshots):
        time_hours = (times_gps[snap_idx] - t0_gps) / 3600.0
        
        # Get positions at this snapshot
        r_snap = r_histories[:, snap_idx, :]
        r_center = r_nominal_hist[snap_idx]
        v_center = v_nominal_hist[snap_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate errors
        errors = r_snap - r_center
        
        # Compute NTW coordinate system
        t_hat = v_center / np.linalg.norm(v_center)
        w_hat = np.cross(r_center, v_center)
        w_hat = w_hat / np.linalg.norm(w_hat)
        n_hat = np.cross(t_hat, w_hat)
        
        # Project errors
        errors_projected = errors - np.outer(np.dot(errors, t_hat), t_hat)
        t_component = np.dot(errors, t_hat)
        
        # Convert to timing difference
        v_magnitude = np.linalg.norm(v_center)
        timing_difference = t_component / v_magnitude
        
        # Convert to 2D
        x_2d = np.dot(errors_projected, n_hat)
        y_2d = np.dot(errors_projected, w_hat)
        
        # Plot
        scatter = ax.scatter(x_2d, y_2d, c=timing_difference, s=80,
                            cmap='RdYlBu_r', zorder=100,
                            edgecolors='black', linewidths=0.8,
                            alpha=0.8, vmin=vmin, vmax=vmax)
        
        cbar = fig.colorbar(scatter, ax=ax, label='Timing difference [seconds]',
                           shrink=0.8, pad=0.02)
        
        ax.scatter(0, 0, c='red', s=150, marker='x', zorder=101,
                  linewidths=3, label='Reference center')
        
        # Add reference circles
        max_error_snap = max(np.max(np.abs(x_2d)), np.max(np.abs(y_2d)))
        if max_error_snap > 0:
            circle_radii = np.linspace(max_error_snap/4, max_error_snap, 4)
            theta = np.linspace(0, 2 * np.pi, 100)
            for radius in circle_radii:
                x_circle = radius * np.cos(theta)
                y_circle = radius * np.sin(theta)
                ax.plot(x_circle, y_circle, 'k--', alpha=0.2, linewidth=0.8)
        
        # Set consistent axis limits
        ax.set_xlim(-all_max * 1.1, all_max * 1.1)
        ax.set_ylim(-all_max * 1.1, all_max * 1.1)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('Normal (N) error [m]', fontsize=12)
        ax.set_ylabel('Cross-track (W) error [m]', fontsize=12)
        ax.set_title(f'Position Errors at T+{time_hours:.1f} hours',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(temp_dir, f'frame_{snap_idx:03d}.png')
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        frame_files.append(frame_path)
        plt.close(fig)
        
        if snap_idx % 10 == 0:
            print(f"  Generated frame {snap_idx+1}/{n_snapshots}")
    
    # Use write_gif from ssatk
    output_path = os.path.expanduser(output_path)
    
    write_gif(
        gif_name=output_path,
        frames=frame_files,
        fps=fps,
        duration=duration,
        loop=0,
        sort_frames=False,
        uniform_size=True,
        bg_color=(255, 255, 255, 0)
    )
    
    print(f"GIF saved to: {output_path}")
    
    # Clean up temporary files
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)
    
    return output_path


# Usage examples:
# Example 1: Minimal - just positions, times, and output path
# gif_path = divergence_gif(r_histories, times_gps, "output.gif")

# Example 2: With explicit nominal trajectory
# gif_path = divergence_gif(r_histories, times_gps, "output.gif",
#                          r_nominal_hist=r_nom, v_nominal_hist=v_nom)

# Example 3: Custom timing
# gif_path = divergence_gif(r_histories, times_gps, "output.gif", duration=1.0)
