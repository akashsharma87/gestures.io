"""
SpatialFlow Signal Processing Module
=====================================

Implements industrial-grade signal processing for eliminating jitter and providing
rock-solid tracking stability. This is the PRIORITY ZERO module.

Classes:
    - OneEuroFilter: Adaptive low-pass filter for smooth, responsive tracking
    - SchmittTrigger: Hysteresis-based boolean state machine for gesture detection

The One Euro Filter paper: https://cristal.univ-lille.fr/~casiez/1euro/
"""

import math
import time
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class FilterConfig:
    """Configuration for the One Euro Filter"""
    min_cutoff: float = 1.0      # Minimum cutoff frequency (Hz) - lower = smoother
    beta: float = 0.007          # Speed coefficient - higher = more responsive
    d_cutoff: float = 1.0        # Derivative cutoff frequency (Hz)


class LowPassFilter:
    """
    Simple exponential smoothing low-pass filter.
    
    This is the building block for the One Euro Filter.
    Uses the formula: y[n] = α * x[n] + (1 - α) * y[n-1]
    """
    
    def __init__(self, alpha: float = 0.5):
        self._alpha = alpha
        self._y_prev: Optional[float] = None
        self._initialized = False
    
    @property
    def alpha(self) -> float:
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float):
        self._alpha = max(0.0, min(1.0, value))  # Clamp to [0, 1]
    
    def reset(self):
        """Reset the filter state"""
        self._y_prev = None
        self._initialized = False
    
    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        """
        Apply the low-pass filter to the input value.
        
        Args:
            value: The raw input value
            alpha: Optional override for the smoothing factor
            
        Returns:
            The filtered value
        """
        if alpha is not None:
            self.alpha = alpha
            
        if not self._initialized:
            self._y_prev = value
            self._initialized = True
            return value
        
        # Exponential smoothing formula
        filtered = self._alpha * value + (1.0 - self._alpha) * self._y_prev
        self._y_prev = filtered
        return filtered


class OneEuroFilter:
    """
    The One Euro Filter - Adaptive low-pass filter for noisy signals.
    
    This filter provides excellent noise reduction while minimizing latency.
    It adapts the cutoff frequency based on the signal's rate of change:
    - Slow movements → low cutoff → heavy smoothing (reduces jitter)
    - Fast movements → high cutoff → light smoothing (reduces lag)
    
    The magic formula:
        cutoff = min_cutoff + β * |derivative|
        
    where:
        - min_cutoff: baseline smoothing (lower = smoother at rest)
        - β (beta): responsiveness scaling factor
        
    Recommended starting values for hand tracking:
        - min_cutoff = 1.0 Hz
        - beta = 0.007
        
    Reference: Géry Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter"
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        initial_value: Optional[float] = None
    ):
        """
        Initialize the One Euro Filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency in Hz (default: 1.0)
            beta: Speed coefficient for adaptive cutoff (default: 0.007)
            d_cutoff: Derivative cutoff frequency in Hz (default: 1.0)
            initial_value: Optional initial value to seed the filter
        """
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        
        # Create the two low-pass filters
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        
        # Timing
        self._last_time: Optional[float] = None
        self._initialized = False
        
        if initial_value is not None:
            self.filter(initial_value)
    
    @property
    def config(self) -> FilterConfig:
        """Get the current filter configuration"""
        return FilterConfig(
            min_cutoff=self._min_cutoff,
            beta=self._beta,
            d_cutoff=self._d_cutoff
        )
    
    @config.setter
    def config(self, cfg: FilterConfig):
        """Set the filter configuration"""
        self._min_cutoff = cfg.min_cutoff
        self._beta = cfg.beta
        self._d_cutoff = cfg.d_cutoff
    
    def reset(self):
        """Reset the filter to its initial state"""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None
        self._initialized = False
    
    def _compute_alpha(self, cutoff: float, dt: float) -> float:
        """
        Compute the smoothing factor α from the cutoff frequency.
        
        The formula is derived from the RC time constant:
            τ = 1 / (2π * cutoff)
            α = 1 / (1 + τ/dt)
            
        Simplified:
            α = 1 / (1 + 1/(2π * cutoff * dt))
            
        Args:
            cutoff: The cutoff frequency in Hz
            dt: Time delta in seconds
            
        Returns:
            The smoothing factor α ∈ [0, 1]
        """
        tau = 1.0 / (2.0 * math.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / dt)
        return alpha
    
    def filter(self, value: float, timestamp: Optional[float] = None) -> float:
        """
        Apply the One Euro Filter to the input value.
        
        Args:
            value: The raw noisy input value
            timestamp: Optional timestamp in seconds (uses time.time() if not provided)
            
        Returns:
            The filtered, smooth value
        """
        # Get current time
        current_time = timestamp if timestamp is not None else time.time()
        
        if not self._initialized:
            # First sample - just store and return
            self._last_time = current_time
            self._x_filter.filter(value)
            self._dx_filter.filter(0.0)
            self._initialized = True
            return value
        
        # Calculate time delta
        dt = current_time - self._last_time
        if dt <= 0:
            dt = 1e-6  # Avoid division by zero
        self._last_time = current_time
        
        # Step 1: Estimate the derivative (rate of change)
        # Using the previous filtered value to reduce noise amplification
        prev_filtered = self._x_filter._y_prev if self._x_filter._y_prev is not None else value
        dx = (value - prev_filtered) / dt
        
        # Step 2: Filter the derivative
        dx_alpha = self._compute_alpha(self._d_cutoff, dt)
        dx_filtered = self._dx_filter.filter(dx, dx_alpha)
        
        # Step 3: Calculate the adaptive cutoff frequency
        # cutoff = min_cutoff + β * |derivative|
        cutoff = self._min_cutoff + self._beta * abs(dx_filtered)
        
        # Step 4: Apply the main filter with adaptive cutoff
        x_alpha = self._compute_alpha(cutoff, dt)
        filtered_value = self._x_filter.filter(value, x_alpha)
        
        return filtered_value


class OneEuroFilter2D:
    """
    2D variant of the One Euro Filter for (x, y) coordinate pairs.
    Maintains separate filters for each dimension.
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        self._x_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._y_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
    
    def reset(self):
        """Reset both filters"""
        self._x_filter.reset()
        self._y_filter.reset()
    
    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """
        Filter a 2D coordinate pair.
        
        Args:
            x: Raw x coordinate
            y: Raw y coordinate
            timestamp: Optional timestamp
            
        Returns:
            Tuple of (filtered_x, filtered_y)
        """
        fx = self._x_filter.filter(x, timestamp)
        fy = self._y_filter.filter(y, timestamp)
        return (fx, fy)


class OneEuroFilter3D:
    """
    3D variant of the One Euro Filter for (x, y, z) coordinate triples.
    Maintains separate filters for each dimension.
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        self._x_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._y_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._z_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
    
    def reset(self):
        """Reset all filters"""
        self._x_filter.reset()
        self._y_filter.reset()
        self._z_filter.reset()
    
    def filter(
        self, 
        x: float, 
        y: float, 
        z: float, 
        timestamp: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Filter a 3D coordinate triple.
        
        Args:
            x, y, z: Raw coordinates
            timestamp: Optional timestamp
            
        Returns:
            Tuple of (filtered_x, filtered_y, filtered_z)
        """
        fx = self._x_filter.filter(x, timestamp)
        fy = self._y_filter.filter(y, timestamp)
        fz = self._z_filter.filter(z, timestamp)
        return (fx, fy, fz)


class SchmittTrigger:
    """
    Schmitt Trigger - Hysteresis-based boolean state machine.
    
    Prevents rapid oscillation between states (debouncing) by using
    two separate thresholds for transitions:
    
    - LOW → HIGH: Input must exceed high_threshold
    - HIGH → LOW: Input must fall below low_threshold
    
    The gap between thresholds creates a "dead zone" that prevents
    noise from causing unwanted state changes.
    
    Example for pinch detection:
        - high_threshold = 0.08 (release threshold)
        - low_threshold = 0.05 (pinch threshold)
        
        When fingers are apart (dist > 0.08) → RELEASED
        When fingers come together (dist < 0.05) → PINCHED
        Between 0.05 and 0.08 → stays in current state
    
    ASCII visualization:
    
        PINCHED ─────────────────────LOW_THRESH─────────────▶ (stays PINCHED)
                                          │
        RELEASED ◀───────────────────────┘ (if input rises above HIGH_THRESH)
                                          
        LOW_THRESH                    HIGH_THRESH
            │                              │
            ▼                              ▼
        ────┴──────────────────────────────┴────▶ input value
            0.05                          0.08
    """
    
    # State constants
    STATE_LOW = False   # e.g., PINCHED
    STATE_HIGH = True   # e.g., RELEASED
    
    def __init__(
        self,
        low_threshold: float = 0.05,
        high_threshold: float = 0.08,
        initial_state: bool = True
    ):
        """
        Initialize the Schmitt Trigger.
        
        Args:
            low_threshold: Threshold for HIGH → LOW transition
            high_threshold: Threshold for LOW → HIGH transition
            initial_state: Starting state (True = HIGH, False = LOW)
            
        Raises:
            ValueError: If low_threshold >= high_threshold
        """
        if low_threshold >= high_threshold:
            raise ValueError(
                f"low_threshold ({low_threshold}) must be less than "
                f"high_threshold ({high_threshold})"
            )
        
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold
        self._state = initial_state
        self._last_value: Optional[float] = None
        
        # Transition callbacks
        self._on_rising: Optional[callable] = None
        self._on_falling: Optional[callable] = None
    
    @property
    def state(self) -> bool:
        """Current state (True = HIGH, False = LOW)"""
        return self._state
    
    @property
    def is_high(self) -> bool:
        """Check if currently in HIGH state"""
        return self._state
    
    @property
    def is_low(self) -> bool:
        """Check if currently in LOW state"""
        return not self._state
    
    def on_rising_edge(self, callback: callable):
        """Register callback for LOW → HIGH transitions"""
        self._on_rising = callback
    
    def on_falling_edge(self, callback: callable):
        """Register callback for HIGH → LOW transitions"""
        self._on_falling = callback
    
    def reset(self, state: bool = True):
        """Reset to a specific state"""
        self._state = state
        self._last_value = None
    
    def update(self, value: float) -> bool:
        """
        Update the trigger with a new input value.
        
        Args:
            value: The input signal value
            
        Returns:
            The current state after processing the input
        """
        self._last_value = value
        previous_state = self._state
        
        if self._state:  # Currently HIGH
            if value < self._low_threshold:
                self._state = False  # Transition to LOW
                if self._on_falling:
                    self._on_falling()
        else:  # Currently LOW
            if value > self._high_threshold:
                self._state = True  # Transition to HIGH
                if self._on_rising:
                    self._on_rising()
        
        return self._state
    
    def check_transition(self, value: float) -> Optional[str]:
        """
        Check what transition would occur and update state.
        
        Args:
            value: The input signal value
            
        Returns:
            'rising' for LOW→HIGH, 'falling' for HIGH→LOW, None for no change
        """
        previous_state = self._state
        self.update(value)
        
        if previous_state != self._state:
            return 'rising' if self._state else 'falling'
        return None


class GestureDetector:
    """
    High-level gesture detector combining multiple SchmittTriggers.
    
    Detects:
        - PINCH: Thumb and index finger close together
        - FIST: All fingers curled
        - POINT: Index finger extended, others curled
    """
    
    def __init__(
        self,
        pinch_low: float = 0.04,
        pinch_high: float = 0.07,
        fist_threshold: float = 0.15
    ):
        self.pinch_trigger = SchmittTrigger(
            low_threshold=pinch_low,
            high_threshold=pinch_high,
            initial_state=True  # Start RELEASED
        )
        
        self._fist_threshold = fist_threshold
        self._is_fist = False
        self._is_peace = False
    
    @property
    def is_pinching(self) -> bool:
        """Check if pinch gesture is active"""
        return self.pinch_trigger.is_low
    
    @property
    def is_fist(self) -> bool:
        """Check if fist gesture is active"""
        return self._is_fist

    @property
    def is_peace(self) -> bool:
        """Check if peace gesture is active"""
        return self._is_peace

    
    def update_pinch(self, thumb_index_distance: float) -> bool:
        """
        Update pinch detection with thumb-index distance.
        
        Args:
            thumb_index_distance: Normalized distance between thumb tip and index tip
            
        Returns:
            True if pinching, False otherwise
        """
        self.pinch_trigger.update(thumb_index_distance)
        return self.is_pinching
    
    def update_fist(self, avg_finger_curl: float) -> bool:
        """
        Update fist detection.
        
        Args:
            avg_finger_curl: Average curl ratio of all fingers (0=straight, 1=fully curled)
            
        Returns:
            True if fist detected, False otherwise
        """
        self._is_fist = avg_finger_curl > self._fist_threshold
        return self._is_fist

    def update_peace(self, index_dist: float, middle_dist: float, others_dist: float) -> bool:
        """
        Update peace sign detection.
        
        Args:
            index_dist: Distance of index tip from palm
            middle_dist: Distance of middle tip from palm
            others_dist: Average distance of other finger tips from palm
            
        Returns:
            True if peace sign detected, False otherwise
        """
        # Index and middle must be extended, others must be curled
        self._is_peace = index_dist > 0.15 and middle_dist > 0.15 and others_dist < 0.12
        return self._is_peace



# Utility function for testing
def demo_filters():
    """Demonstrate filter behavior with synthetic noisy data"""
    import random
    
    print("=" * 60)
    print("One Euro Filter Demo")
    print("=" * 60)
    
    # Create a filter
    oef = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    
    # Simulate noisy position data
    true_value = 0.5
    noise_amplitude = 0.05
    
    print("\nFiltering noisy signal around 0.5:")
    print("-" * 40)
    
    for i in range(10):
        noisy = true_value + random.uniform(-noise_amplitude, noise_amplitude)
        filtered = oef.filter(noisy)
        error = abs(filtered - true_value)
        print(f"Raw: {noisy:.4f} → Filtered: {filtered:.4f} | Error: {error:.4f}")
    
    print("\n" + "=" * 60)
    print("Schmitt Trigger Demo")
    print("=" * 60)
    
    trigger = SchmittTrigger(low_threshold=0.3, high_threshold=0.7)
    
    # Simulate signal crossing thresholds
    test_values = [0.2, 0.4, 0.5, 0.6, 0.75, 0.65, 0.5, 0.35, 0.25]
    
    print("\nThresholds: LOW=0.3, HIGH=0.7")
    print("-" * 40)
    
    for val in test_values:
        transition = trigger.check_transition(val)
        state_str = "HIGH" if trigger.state else "LOW"
        trans_str = f" [{transition}]" if transition else ""
        print(f"Input: {val:.2f} → State: {state_str}{trans_str}")


if __name__ == "__main__":
    demo_filters()
