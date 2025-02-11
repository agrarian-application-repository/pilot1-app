import numpy as np


def compute_heading(velocity: np.ndarray) -> np.ndarray:
    """
    Compute the heading (orientation) angle from velocity vectors.

    The heading is calculated as the angle between the velocity vector and
    the positive x-axis, using the arctan2 function to account for the correct
    quadrant. The angle is returned in radians.

    Parameters
    ----------
    velocity : np.ndarray
        Array of shape (N, 2, T) representing velocity vectors for N objects over T time steps.
        The first channel (index 0) is the x-component and the second (index 1) is the y-component.

    Returns
    -------
    heading : np.ndarray
        Array of shape (N, T) containing heading angles (in radians) for each object at each time step.
    """
    # Compute heading angle for each object at each time step
    heading = np.arctan2(velocity[:, 1, :], velocity[:, 0, :])
    return heading


def compute_angular_velocity(heading: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute the angular velocity (rate of change of heading) for each object.

    Angular velocity is computed as the difference between consecutive unwrapped
    heading angles divided by the time step dt. Unwrapping helps to avoid issues
    with discontinuities (e.g., jumping from π to -π).

    Parameters
    ----------
    heading : np.ndarray
        Array of shape (N, T) representing heading angles (in radians) for each object over T time steps.
    dt : float, optional
        Time interval between consecutive heading measurements (default is 1.0).

    Returns
    -------
    angular_velocity : np.ndarray
        Array of shape (N, T-1) representing the angular velocity for each object over time.
    """
    # Unwrap the angles along the time axis to remove discontinuities
    unwrapped_heading = np.unwrap(heading, axis=1)
    # Compute the difference between consecutive headings and normalize by dt
    angular_velocity = np.diff(unwrapped_heading, axis=1) / dt
    return angular_velocity


def compute_turning_angle(positions: np.ndarray) -> np.ndarray:
    """
    Compute the turning angle between consecutive displacement vectors for each object.

    The turning angle is a measure of how much the direction changes from one segment
    of the trajectory to the next. It is computed using the arctan2 of the cross product
    and dot product of two consecutive displacement (or velocity) vectors.

    For each object, if v1 is the displacement vector from time t to t+1 and
    v2 is from t+1 to t+2, then the turning angle at time t is:

        turning_angle = arctan2(cross(v1, v2), dot(v1, v2))

    where:
        cross(v1, v2) = v1_x * v2_y - v1_y * v2_x
        dot(v1, v2)   = v1_x * v2_x + v1_y * v2_y

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing the [x, y] positions of N objects over T time steps.

    Returns
    -------
    turning_angle : np.ndarray
        Array of shape (N, T-2) representing the turning angle (in radians) between consecutive displacement vectors.
        For each object, the turning angle is computed for each triple of consecutive positions.
    """
    # Compute displacement vectors between consecutive positions.
    # Resulting shape is (N, 2, T-1)
    displacement = np.diff(positions, axis=2)

    # For consecutive displacement vectors, define:
    # v1 = displacement from time t to t+1 (shape: (N, 2, T-2))
    # v2 = displacement from time t+1 to t+2 (shape: (N, 2, T-2))
    v1 = displacement[:, :, :-1]
    v2 = displacement[:, :, 1:]

    # Compute the dot product and cross product (2D cross product returns a scalar)
    dot_product = v1[:, 0, :] * v2[:, 0, :] + v1[:, 1, :] * v2[:, 1, :]
    cross_product = v1[:, 0, :] * v2[:, 1, :] - v1[:, 1, :] * v2[:, 0, :]

    # Compute turning angle using arctan2
    turning_angle = np.arctan2(cross_product, dot_product)
    return turning_angle


# Example usage:
if __name__ == "__main__":
    # Create a dummy dataset:
    # Assume we have 3 objects, 2D positions (normalized between 0 and 1), and 10 time steps.
    N = 3  # number of objects
    T = 10  # number of time steps
    positions = np.random.rand(N, 2, T)


    # For directional features, we first need velocity vectors.
    # Here’s a helper function to compute velocity from positions:
    def compute_velocity(positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Compute velocity vectors by differencing positions over time.

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (N, 2, T) representing positions.
        dt : float, optional
            Time interval between measurements (default is 1.0).

        Returns
        -------
        velocity : np.ndarray
            Array of shape (N, 2, T-1) representing velocity vectors.
        """
        displacement = np.diff(positions, axis=2)
        velocity = displacement / dt
        return velocity


    # Compute velocity from positions.
    velocity = compute_velocity(positions, dt=1.0)  # shape: (N, 2, T-1)

    # Compute directional features.
    heading = compute_heading(velocity)  # shape: (N, T-1)
    angular_velocity = compute_angular_velocity(heading)  # shape: (N, T-2)
    turning_angle = compute_turning_angle(positions)  # shape: (N, T-2)

    # Display the shapes of the computed features.
    print("Velocity shape:", velocity.shape)  # Expected: (N, 2, T-1)
    print("Heading shape:", heading.shape)  # Expected: (N, T-1)
    print("Angular Velocity shape:", angular_velocity.shape)  # Expected: (N, T-2)
    print("Turning Angle shape:", turning_angle.shape)  # Expected: (N, T-2)
