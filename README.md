Hybrid multiscale hysteretic model for bipolar II disorder


## Setting up the environment

Set up a virtual environment and install the required packages using the following commands:

```bash
python -m venv .venv
```

Activate the virtual environment:

- On Windows:
```powershell
.venv\Scripts\activate
```

- On macOS and Linux:
```bash
source .venv/bin/activate
```

Install pip-tools

```bash
pip install pip-tools
```

Synchronize the environment with the requirements file:

```bash
pip-sync
```

**Remark** In case of changing the requirements in pyproject.toml, run the following command to update the requirements.txt file:

```bash
pip-compile
```
