"""
Membership Functions for Fuzzy Linguistic Variables

This module implements various membership function types used
to map crisp values to fuzzy membership degrees.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Addresses RQ1: Modeling vagueness through explicit membership functions
- Based on: Zadeh (1965), Mamdani (1974)

Supported Function Types:
1. Triangular (trimf)
2. Trapezoidal (trapmf)
3. Gaussian (gaussmf)
4. Sigmoid (sigmf)
"""

import numpy as np
from typing import Union, List, Callable, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MembershipFunction:
    """
    Represents a membership function with its parameters.
    
    Attributes:
        func_type: Type of membership function
        params: Function parameters
        func: Callable that computes membership
    """
    func_type: str
    params: List[float]
    func: Callable[[float], float]
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute membership degree for input value(s)."""
        return self.func(x)


# =============================================================================
# MEMBERSHIP FUNCTION IMPLEMENTATIONS
# =============================================================================

def triangular(x: Union[float, np.ndarray], a: float, b: float, c: float) -> Union[float, np.ndarray]:
    """
    Triangular membership function.
    
    Shape: /\
           a  b  c
    
    Args:
        x: Input value(s)
        a: Left foot (membership = 0)
        b: Peak (membership = 1)
        c: Right foot (membership = 0)
        
    Returns:
        Membership degree(s) in [0, 1]
        
    Example:
        "recent" might be triangular(x, 0, 7, 30)
        - At x=0 days: membership = 0 (just updated)
        - At x=7 days: membership = 1 (peak of "recent")
        - At x=30 days: membership = 0 (no longer recent)
    """
    x = np.asarray(x)

    if a == b and b == c:
        return np.ones_like(x, dtype=float)

    result = np.zeros_like(x, dtype=float)

    if a == b:
        rising = x <= b
        result[rising] = 1.0
        falling = (x > b) & (x < c)
        result[falling] = (c - x[falling]) / (c - b + 1e-10)
        return result

    if b == c:
        rising = (x > a) & (x < b)
        result[rising] = (x[rising] - a) / (b - a + 1e-10)
        result[x >= b] = 1.0
        return result

    rising = (x > a) & (x <= b)
    falling = (x > b) & (x < c)
    result[rising] = (x[rising] - a) / (b - a + 1e-10)
    result[falling] = (c - x[falling]) / (c - b + 1e-10)
    return result


def trapezoidal(x: Union[float, np.ndarray], a: float, b: float, 
                c: float, d: float) -> Union[float, np.ndarray]:
    """
    Trapezoidal membership function.
    
    Shape:  ____
           /    \
          a  b  c  d
    
    Args:
        x: Input value(s)
        a: Left foot
        b: Left shoulder (start of plateau)
        c: Right shoulder (end of plateau)
        d: Right foot
        
    Returns:
        Membership degree(s) in [0, 1]
        
    Example:
        "moderate completeness" might be trapezoidal(x, 0.4, 0.5, 0.7, 0.8)
        - Full membership for completeness 50-70%
        - Declining membership outside that range
    """
    x = np.asarray(x)

    if a == b and c == d:
        return np.where((x >= a) & (x <= d), 1.0, 0.0)

    result = np.zeros_like(x, dtype=float)
    rising = (x > a) & (x < b)
    plateau = (x >= b) & (x <= c)
    falling = (x > c) & (x < d)

    if b != a:
        result[rising] = (x[rising] - a) / (b - a + 1e-10)
    else:
        result[x <= b] = 1.0

    result[plateau] = 1.0

    if d != c:
        result[falling] = (d - x[falling]) / (d - c + 1e-10)
    else:
        result[x >= c] = 1.0

    return np.clip(result, 0.0, 1.0)


def gaussian(x: Union[float, np.ndarray], mean: float, sigma: float) -> Union[float, np.ndarray]:
    """
    Gaussian membership function.
    
    Shape: Bell curve centered at mean
    
    Args:
        x: Input value(s)
        mean: Center of the Gaussian
        sigma: Standard deviation (width)
        
    Returns:
        Membership degree(s) in [0, 1]
        
    Example:
        "around 50% complete" might be gaussian(x, 0.5, 0.1)
    """
    x = np.asarray(x)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def sigmoid(x: Union[float, np.ndarray], a: float, c: float) -> Union[float, np.ndarray]:
    """
    Sigmoid membership function.
    
    Shape: S-curve (increasing or decreasing based on sign of a)
    
    Args:
        x: Input value(s)
        a: Slope parameter (positive = increasing, negative = decreasing)
        c: Inflection point (where membership = 0.5)
        
    Returns:
        Membership degree(s) in [0, 1]
        
    Example:
        "old dataset" might be sigmoid(x, 0.01, 365)
        - Datasets older than 365 days have increasing "old" membership
    """
    x = np.asarray(x)
    return 1 / (1 + np.exp(-a * (x - c)))


def zmf(x: Union[float, np.ndarray], a: float, b: float) -> Union[float, np.ndarray]:
    """
    Z-shaped membership function (spline-based).
    
    Shape: Smooth decrease from 1 to 0
    
    Args:
        x: Input value(s)
        a: Point where membership starts decreasing
        b: Point where membership reaches 0
        
    Returns:
        Membership degree(s) in [0, 1]
    """
    x = np.asarray(x)
    mid = (a + b) / 2
    
    result = np.zeros_like(x, dtype=float)
    
    # x <= a: full membership
    mask1 = x <= a
    result[mask1] = 1
    
    # a < x <= mid: smooth decrease
    mask2 = (x > a) & (x <= mid)
    result[mask2] = 1 - 2 * ((x[mask2] - a) / (b - a)) ** 2
    
    # mid < x < b: continue decrease
    mask3 = (x > mid) & (x < b)
    result[mask3] = 2 * ((x[mask3] - b) / (b - a)) ** 2
    
    # x >= b: zero membership
    # Already initialized to 0
    
    return result


def smf(x: Union[float, np.ndarray], a: float, b: float) -> Union[float, np.ndarray]:
    """
    S-shaped membership function (spline-based).
    
    Shape: Smooth increase from 0 to 1
    
    Args:
        x: Input value(s)
        a: Point where membership starts increasing
        b: Point where membership reaches 1
        
    Returns:
        Membership degree(s) in [0, 1]
    """
    return 1 - zmf(x, a, b)


# =============================================================================
# MEMBERSHIP FUNCTION FACTORY
# =============================================================================

FUNCTION_REGISTRY = {
    "triangular": triangular,
    "trapezoidal": trapezoidal,
    "gaussian": gaussian,
    "sigmoid": sigmoid,
    "zmf": zmf,
    "smf": smf
}


def create_membership_function(func_type: str, params: List[float]) -> MembershipFunction:
    """
    Factory function to create a membership function.
    
    Args:
        func_type: Type of function ("triangular", "trapezoidal", etc.)
        params: Parameters for the function
        
    Returns:
        MembershipFunction instance
        
    Example:
        >>> mf = create_membership_function("triangular", [0, 7, 30])
        >>> mf(7)  # Peak of triangle
        1.0
        >>> mf(15)  # Midpoint on right slope
        0.65...
    """
    if func_type not in FUNCTION_REGISTRY:
        raise ValueError(f"Unknown function type: {func_type}. "
                        f"Available: {list(FUNCTION_REGISTRY.keys())}")
    
    base_func = FUNCTION_REGISTRY[func_type]
    
    def bound_func(x):
        return base_func(x, *params)
    
    return MembershipFunction(
        func_type=func_type,
        params=params,
        func=bound_func
    )


def create_from_variable_definition(term_def: Dict) -> MembershipFunction:
    """
    Create membership function from variable term definition.
    
    Args:
        term_def: Dictionary with 'type' and 'params' keys
        
    Returns:
        MembershipFunction instance
    """
    return create_membership_function(
        func_type=term_def["type"],
        params=term_def["params"]
    )


# =============================================================================
# FUZZY OPERATIONS
# =============================================================================

def fuzzy_and(*memberships: float) -> float:
    """
    Fuzzy AND (intersection) using minimum operator.
    
    Args:
        *memberships: Membership degrees
        
    Returns:
        Minimum membership (standard fuzzy intersection)
    """
    return min(memberships)


def fuzzy_or(*memberships: float) -> float:
    """
    Fuzzy OR (union) using maximum operator.
    
    Args:
        *memberships: Membership degrees
        
    Returns:
        Maximum membership (standard fuzzy union)
    """
    return max(memberships)


def fuzzy_not(membership: float) -> float:
    """
    Fuzzy NOT (complement).
    
    Args:
        membership: Membership degree
        
    Returns:
        1 - membership
    """
    return 1 - membership


def aggregate_memberships(memberships: List[float], method: str = "max") -> float:
    """
    Aggregate multiple membership degrees.
    
    Args:
        memberships: List of membership degrees
        method: Aggregation method ("max", "sum", "mean", "prod")
        
    Returns:
        Aggregated value
    """
    if not memberships:
        return 0.0
    
    if method == "max":
        return max(memberships)
    elif method == "sum":
        return min(1.0, sum(memberships))  # Cap at 1
    elif method == "mean":
        return sum(memberships) / len(memberships)
    elif method == "prod":
        result = 1.0
        for m in memberships:
            result *= m
        return result
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# =============================================================================
# VISUALIZATION HELPER
# =============================================================================

def plot_membership_function(
    mf: MembershipFunction,
    universe: Tuple[float, float],
    title: str = "",
    ax=None
):
    """
    Plot a membership function.
    
    Args:
        mf: MembershipFunction to plot
        universe: (min, max) range for x-axis
        title: Plot title
        ax: Matplotlib axis (creates new figure if None)
    """
    try:
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        
        x = np.linspace(universe[0], universe[1], 500)
        y = mf(x)
        
        ax.plot(x, y, linewidth=2)
        ax.fill_between(x, 0, y, alpha=0.3)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Value")
        ax.set_ylabel("Membership Degree")
        ax.set_title(title or f"{mf.func_type.capitalize()} MF")
        ax.grid(True, alpha=0.3)
        
        return ax
        
    except ImportError:
        print("matplotlib required for plotting")
        return None


if __name__ == "__main__":
    # Demo: Create and test membership functions
    print("=" * 60)
    print("MEMBERSHIP FUNCTION DEMONSTRATION")
    print("=" * 60)
    
    # Create triangular MF for "recent"
    recent_mf = create_membership_function("triangular", [0, 7, 30])
    
    test_values = [0, 3, 7, 15, 30, 60]
    print("\nTriangular MF for 'recent' (params: [0, 7, 30]):")
    for val in test_values:
        print(f"  {val} days -> membership = {recent_mf(val):.3f}")
    
    # Create Gaussian MF for "moderate completeness"
    moderate_mf = create_membership_function("gaussian", [0.5, 0.15])
    
    test_values = [0.2, 0.35, 0.5, 0.65, 0.8]
    print("\nGaussian MF for 'moderate completeness' (params: [0.5, 0.15]):")
    for val in test_values:
        print(f"  {val:.0%} complete -> membership = {moderate_mf(val):.3f}")
    
    # Demonstrate fuzzy operations
    print("\nFuzzy Operations:")
    m1, m2 = 0.7, 0.4
    print(f"  μ1 = {m1}, μ2 = {m2}")
    print(f"  AND(μ1, μ2) = {fuzzy_and(m1, m2)}")
    print(f"  OR(μ1, μ2) = {fuzzy_or(m1, m2)}")
    print(f"  NOT(μ1) = {fuzzy_not(m1)}")
