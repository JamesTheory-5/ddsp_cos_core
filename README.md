# ddsp_cos_core

Below is the **completed specification** for `ddsp_cos_osc_core` **followed by the full standalone Python module**, exactly matching your GDSP/JAX/DDSP requirements.

---

# ✅ **DDSP COSINE OSCILLATOR MODULE SPEC**

## MODULE NAME:

**ddsp_cos_osc_core**

---

## DESCRIPTION:

A **pure functional**, **differentiable**, **jit-safe** cosine oscillator driven by a normalized **input phase**.
This oscillator:

* consumes an externally generated phase (e.g., from `ddsp_phasor_osc_core`)
* applies optional:

  * amplitude smoothing
  * phase offset (normalized)
  * soft bandlimiting shape
* uses `jnp.cos(2π·phase)` as the core waveform
* performs **no phase accumulation** internally
* is 100% compatible with the GammaJAX DDSP architecture

It is **stateless except for amplitude smoothing**, making it ideal for:

* additive synthesis
* FM/PM/AM stacks
* multi-oscillator banks
* LFOs
* neural-conditioned DDSP models

---

## INPUTS:

* **phase** : normalized incoming phase in `[0,1)` (or any real; wrapped internally)
* **params.amp_target** : target amplitude value
* **params.amp_smooth_coef** : smoothing coefficient α (0–1)
* **params.phase_offset** : normalized phase offset added before cosine
* **params.bandlimit_flag** : 0.0 = normal cosine, 1.0 = shaped cosine
* **params.shape_amount** : amount of shaping for bandlimiting

All inputs must be JAX arrays or scalars.

---

## OUTPUTS:

* **y** : output cosine waveform sample
* **new_state** : updated smoothing state

---

## STATE VARIABLES:

```python
(
    amp_smooth,    # smoothed amplitude value
)
```

---

## EQUATIONS / MATH:

### Phase wrapping

```
phase_wrapped = phase - floor(phase)
```

### Phase offset

```
p = phase_wrapped + phase_offset
p = p - floor(p)
```

### Cosine core

```
y0 = cos(2π * p)
```

### Optional bandlimited shape (soft)

```
shape_term = 1 - shape_amount * p^2
y_shaped   = y0 * shape_term

y = y0*(1 - bandlimit_flag) + y_shaped*bandlimit_flag
```

### Amplitude smoothing

```
amp_smooth[n+1] = amp_smooth[n] + α * (amp_target[n] - amp_smooth[n])
```

### Final output

```
y_out = amp_smooth * y
```

---

## NOTES:

* No Python branching inside jit: all conditional behavior uses `jnp.where` or arithmetic masks.
* No dynamic allocation inside jit.
* No classes, dicts, or side effects.
* All shapes computed outside jit.
* Per-sample parameters must be broadcastable for `lax.scan`.

---

# ✅ **FULL PYTHON MODULE: `ddsp_cos_osc_core.py`**

```python
"""
ddsp_cos_osc_core.py

GammaJAX DDSP — Cosine Oscillator Core
--------------------------------------

This module implements a fully differentiable, pure-JAX cosine oscillator
in GDSP style. Unlike phasor oscillators, this module does NOT generate
its own phase: it consumes a normalized phase input (from phasor_core).

Features:
- Optional amplitude smoothing
- Optional bandlimit shaping
- Optional normalized phase offset
- JIT-safe, differentiable, shape-stable
- Uses the GDSP interface: init(), update_state(), tick(), process()
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple


# =============================================================================
# INIT
# =============================================================================

def ddsp_cos_osc_core_init(
    initial_amp: float = 1.0,
    amp_smooth_coef: float = 0.0,
    phase_offset: float = 0.0,
    bandlimit_flag: float = 0.0,
    shape_amount: float = 0.0,
    *,
    dtype=jnp.float32,
) -> Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray, ...]]:
    """
    Initialize cosine oscillator.

    Returns:
        state  : (amp_smooth,)
        params : (amp_target, amp_smooth_coef, phase_offset,
                  bandlimit_flag, shape_amount)
    """
    amp_smooth = jnp.asarray(initial_amp, dtype=dtype)

    amp_target_arr = jnp.asarray(initial_amp, dtype=dtype)
    smooth_arr = jnp.asarray(amp_smooth_coef, dtype=dtype)
    offset_arr = jnp.asarray(phase_offset, dtype=dtype)
    bl_arr = jnp.asarray(bandlimit_flag, dtype=dtype)
    shape_arr = jnp.asarray(shape_amount, dtype=dtype)

    state = (amp_smooth,)
    params = (amp_target_arr, smooth_arr, offset_arr, bl_arr, shape_arr)
    return state, params


# =============================================================================
# UPDATE STATE  (noop for this oscillator)
# =============================================================================

def ddsp_cos_osc_core_update_state(state, params):
    """No internal updates outside tick; return state unchanged."""
    del params
    return state


# =============================================================================
# TICK
# =============================================================================

@jax.jit
def ddsp_cos_osc_core_tick(
    phase: jnp.ndarray,
    state: Tuple[jnp.ndarray],
    params: Tuple[jnp.ndarray, ...],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    """
    Single-sample cosine oscillator tick.

    Args:
        phase  : normalized phase input
        state  : (amp_smooth,)
        params : (amp_target, amp_smooth_coef, phase_offset,
                  bandlimit_flag, shape_amount)

    Returns:
        y, new_state
    """
    (amp_smooth,) = state
    amp_target, amp_smooth_coef, phase_offset, bandlimit_flag, shape_amount = params

    # Ensure correct dtype, broadcasting
    phase = jnp.asarray(phase, dtype=amp_smooth.dtype)

    amp_smooth_coef = jnp.clip(jnp.asarray(amp_smooth_coef, dtype=phase.dtype), 0.0, 1.0)
    amp_target = jnp.asarray(amp_target, dtype=phase.dtype)
    phase_offset = jnp.asarray(phase_offset, dtype=phase.dtype)
    bandlimit_flag = jnp.clip(jnp.asarray(bandlimit_flag, dtype=phase.dtype), 0.0, 1.0)
    shape_amount = jnp.asarray(shape_amount, dtype=phase.dtype)

    # 1. Amplitude smoothing
    amp_smooth_next = amp_smooth + amp_smooth_coef * (amp_target - amp_smooth)

    # 2. Phase wrap and offset
    p = phase - jnp.floor(phase)
    p = p + phase_offset
    p = p - jnp.floor(p)

    # 3. Cosine core
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=p.dtype)
    y0 = jnp.cos(two_pi * p)

    # 4. Optional bandlimit shaping (soft quadratic weighting)
    shape_term = 1.0 - shape_amount * (p * p)
    y_shaped = y0 * shape_term
    y = y0 * (1.0 - bandlimit_flag) + y_shaped * bandlimit_flag

    # 5. Amplitude apply
    y_out = amp_smooth_next * y

    new_state = (amp_smooth_next,)
    return y_out, new_state


# =============================================================================
# PROCESS
# =============================================================================

@jax.jit
def ddsp_cos_osc_core_process(
    phase_buf: jnp.ndarray,
    state: Tuple[jnp.ndarray],
    params: Tuple[jnp.ndarray, ...],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    """
    Process a buffer of phases using lax.scan.

    phase_buf: (T,)
    """
    phase_buf = jnp.asarray(phase_buf)
    T = phase_buf.shape[0]

    amp_target, amp_smooth_coef, phase_offset, bandlimit_flag, shape_amount = params

    # Broadcast parameters to per-sample arrays
    amp_target = jnp.broadcast_to(jnp.asarray(amp_target), (T,))
    amp_smooth_coef = jnp.broadcast_to(jnp.asarray(amp_smooth_coef), (T,))
    phase_offset = jnp.broadcast_to(jnp.asarray(phase_offset), (T,))
    bandlimit_flag = jnp.broadcast_to(jnp.asarray(bandlimit_flag), (T,))
    shape_amount = jnp.broadcast_to(jnp.asarray(shape_amount), (T,))

    xs = (phase_buf, amp_target, amp_smooth_coef, phase_offset, bandlimit_flag, shape_amount)

    def body(carry, xs_t):
        st = carry
        (phase_t, at, sc, po, bf, sa) = xs_t
        y_t, st_next = ddsp_cos_osc_core_tick(phase_t, st, (at, sc, po, bf, sa))
        return st_next, y_t

    final_state, y_buf = lax.scan(body, state, xs)
    return y_buf, final_state


# =============================================================================
# DEMO / SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    print("=== ddsp_cos_osc_core: smoke test ===")

    sr = 48000
    dur = 0.01
    T = int(sr * dur)
    freq = 440.0

    t = jnp.linspace(0.0, dur, T, endpoint=False)
    phase_buf = jnp.mod(freq * t, 1.0)

    state, params = ddsp_cos_osc_core_init(
        initial_amp=1.0,
        amp_smooth_coef=0.01,
        phase_offset=0.0,
        bandlimit_flag=0.5,
        shape_amount=0.3,
    )

    y_buf, _ = ddsp_cos_osc_core_process(phase_buf, state, params)
    y_np = onp.asarray(y_buf)

    plt.plot(y_np[:300])
    plt.title("Cosine Oscillator (first 300 samples)")
    plt.show()

    if HAVE_SD:
        print("Playing cosine tone...")
        sd.play(y_np * 0.2, samplerate=sr, blocking=True)
```

---

# 🚀 **Next Oscillator?**

You can now request:

✔ `ddsp_triangle_osc_core.py`
✔ `ddsp_saw_blep_osc_core.py`
✔ `ddsp_pulse_blep_osc_core.py`
✔ `ddsp_fm_osc_core.py`
✔ `ddsp_wavetable_osc_core.py`
✔ or **all oscillators together**

Just say:

**“Generate ddsp_triangle_osc_core.py”** or
**“Generate all oscillator modules now.”**
