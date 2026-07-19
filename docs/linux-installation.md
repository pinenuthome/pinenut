# Linux installation and GPU verification

This guide covers a clean installation from source on Linux and a real
two-GPU smoke test. Commands are intended to be run from a normal user account;
only operating-system package installation needs `sudo`.

## Requirements

### CPU installation

- Linux on x86_64 or aarch64
- Python 3.8 or newer
- `pip` and `venv`
- NumPy 1.20 or newer

### NVIDIA GPU installation

- Two CUDA-capable NVIDIA GPUs
- A working NVIDIA driver (`nvidia-smi` must succeed)
- Python 3.10 or newer is recommended for current CuPy releases
- Exactly one CuPy package matching the CUDA major version

Current stable CuPy wheels support CUDA 12.x and 13.x. The `[ctk]` dependency
used below installs CUDA runtime components inside the Python environment, so a
system-wide CUDA Toolkit is not required; a compatible NVIDIA driver is still
required. Pinenut's current `DataParallel` implementation does not use NCCL,
so NCCL is not required for this smoke test.

Official references:

- [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
- [NVIDIA CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## 1. Install operating-system prerequisites

Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-pip python3-venv
```

Fedora, RHEL, Rocky Linux, or AlmaLinux:

```bash
sudo dnf install -y git python3 python3-pip
```

## 2. Create an isolated Python environment

```bash
git clone https://github.com/pinenuthome/pinenut.git
cd pinenut
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Activate the same environment with `source .venv/bin/activate` whenever you
return to the project.

## 3. Choose one installation mode

### CPU only

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### NVIDIA GPU with CUDA 12.x components

```bash
python -m pip install -r requirements-gpu-cu12.txt
python -m pip install -e . --no-deps
```

The equivalent one-command project installation is:

```bash
python -m pip install -e ".[gpu-cu12]"
```

### NVIDIA GPU with CUDA 13.x components

```bash
python -m pip install -r requirements-gpu-cu13.txt
python -m pip install -e . --no-deps
```

The equivalent one-command project installation is:

```bash
python -m pip install -e ".[gpu-cu13]"
```

If a matching system-wide CUDA Toolkit is already installed and you do not
want the CUDA component wheels, install `cupy-cuda12x` or `cupy-cuda13x`
without the `[ctk]` extra.

Do not install `cupy`, `cupy-cuda12x`, and `cupy-cuda13x` together. Confirm
that exactly one CuPy distribution is present:

```bash
python -m pip freeze | grep -i '^cupy'
```

## 4. Verify the installation

Verify the CPU path:

```bash
python -c "import pinenut as pn; x=pn.tensor(3.0); y=x**2; y.backward(); print(x.grad.data)"
```

The command should print `6.0`.

Verify the NVIDIA driver and visible GPUs:

```bash
nvidia-smi
nvidia-smi -L
```

Verify CuPy on every visible GPU:

```bash
python - <<'PY'
import cupy as cp

count = cp.cuda.runtime.getDeviceCount()
print('visible GPU count:', count)
for device_id in range(count):
    with cp.cuda.Device(device_id):
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = properties['name']
        if isinstance(name, bytes):
            name = name.decode()
        value = cp.arange(8, dtype=cp.float32).sum()
        print('GPU {}: {}, smoke sum={}'.format(device_id, name, float(value.get())))
PY
```

For the project test below, the visible GPU count must be at least two.

## 5. Run the tests

Run the CPU-compatible unit tests first:

```bash
python -m unittest discover -s test -v
```

Then expose two physical GPUs and run the synthetic multi-GPU smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/test_multigpu.py --devices 0,1
```

`CUDA_VISIBLE_DEVICES` remaps selected physical GPUs to logical IDs beginning
at zero. For example, to test physical GPUs 2 and 3, use:

```bash
CUDA_VISIBLE_DEVICES=2,3 python tools/test_multigpu.py --devices 0,1
```

The script trains a small synthetic classifier, verifies that both GPU devices
own model parameters, checks that all parameters are finite, and confirms that
the replicas have identical weights after synchronization. It exits with a
non-zero status if any check fails and prints `MULTI-GPU SMOKE TEST PASSED` on
success.

To make the test longer:

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/test_multigpu.py \
  --devices 0,1 --samples 4096 --batch-size 256 --epochs 10
```

## Troubleshooting

### `nvidia-smi` fails

Install or repair the NVIDIA driver before installing CuPy. Follow the NVIDIA
driver and CUDA Linux documentation for your distribution; do not mix runfile
and distribution package-manager installations.

### `cudaErrorInsufficientDriver`

The installed NVIDIA driver is too old for the selected CUDA runtime. Upgrade
the driver or use a CuPy package for an older supported CUDA major version.

### CuPy imports but reports missing CUDA libraries

Reinstall the matching package with `[ctk]`, for example:

```bash
python -m pip uninstall -y cupy cupy-cuda12x cupy-cuda13x
python -m pip install "cupy-cuda12x[ctk]"
```

### Only one GPU is visible

Check both the host and the current process:

```bash
nvidia-smi -L
echo "$CUDA_VISIBLE_DEVICES"
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

Containers additionally need both GPUs passed through, for example with
Docker's `--gpus '"device=0,1"'` option.
