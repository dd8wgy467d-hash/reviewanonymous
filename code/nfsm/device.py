"""JAX platform selection and reporting. Importing this module does not import JAX.

Set NFSM_PLATFORM (mps | cuda | gpu | cpu | tpu) to pin the platform; `configure` must then run
before the first `import jax`.
"""

from __future__ import annotations

import os

_VALID = {"mps", "metal", "cuda", "gpu", "cpu", "tpu"}


def configure(prefer: str | None = None) -> str | None:
    """Set JAX_PLATFORMS from `prefer` or NFSM_PLATFORM; return the platform in effect (None = auto)."""
    pin = prefer or os.environ.get("NFSM_PLATFORM")
    if pin:
        pin = pin.strip().lower()
        if pin not in _VALID:
            raise ValueError(f"NFSM_PLATFORM={pin!r} is not one of {sorted(_VALID)}")
        os.environ["JAX_PLATFORMS"] = pin
    return os.environ.get("JAX_PLATFORMS")


def print_devices(tag: str = "", file=None) -> list:
    """Print the JAX version, backend and devices; return the device list."""
    import jax
    devices = jax.devices()
    print(f"{'[' + tag + '] ' if tag else ''}jax {jax.__version__} backend={jax.default_backend()} "
          f"x64={jax.config.read('jax_enable_x64')} devices={[str(d) for d in devices]}",
          flush=True, file=file)
    return devices
