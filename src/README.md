# Installation

## Prerequisites

Make sure you have **Python 3.13.5** installed.  
You can download it [here](https://www.python.org/downloads/release/python-3135/).

## Install Trader Workstation (TWS)

Trader Workstation (TWS) is Interactive Brokers’ GUI platform that must be running for the IB API to connect.

**Steps:**
1. Download TWS from the [official IB page](https://www.interactivebrokers.ca/en/trading/tws-updateable-stable.php).
2. Create a paper trading account (paper meaning fake money for simulation purposes)
2. Install and launch TWS.
3. Enable API access:
   - Log in to TWS
   - Go to **File** > **Global Configuration** > **API** > **Settings**
   - Enable:
     - *"Enable ActiveX and Socket Clients"*
     - *"Download open orders on connection"*
     - *"Include FX positions when sending portfolio"*
    - Ensure *"Read-Only API"* is disabled
   - Set **Socket port** to: `7497` (default for paper trading, use 7496 for real account)
   - Click **Apply** and **OK**


## Set up the Python Environment

> It is strongly recommend to use `venv` or `virtualenv` for dependency isolation.

### Using `venv`:

```bash
# Ensure the correct Python version ins installed
python --version # Should return "Python 3.13.5". Might have to use "python3" or "python3.13" instead of "python"

# Clone the repository
git clone https://github.com/sultanoloyede/SPY_API.git


# Create a virtual environment named .venv
python -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate


# Install the repository as a package
pip install -e .

# Install the software dependencies
pip install -r requirements.txt
```

## Install the Interactive Broker's API (IBApi)

Install the api package from the [IBkr GitHub Download Link](https://interactivebrokers.github.io/).

In a terminal instance, activate the `.venv`.

```bash
# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

Once download and installed on your computer, naviguate to the install location in the `\source\pythonclient` folder.

From there, run the following command:

```bash
python setup.py install
```


