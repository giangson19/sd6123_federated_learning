# SD6123: Data Privacy Group Assignment

## Description
This repository contains the implementation for the SD6123 Data Privacy Group Assignment focusing on federated learning.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sd6123_federated_learning.git
    cd sd6123_federated_learning
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Simulation

To run the federated learning simulation:

```bash
python run_simulation.py
```

### Command Line Arguments

You can customize the simulation with various arguments:

```bash
python run_simulation.py --strategy fedprox --proximal-mu 0.1 --num-rounds 20
```

#### Strategy Selection
--strategy Choose the FL strategy:
- `fedavg`: Federated Averaging
- `fedavgm`: FedAvg with momentum
- `fedprox`: FedProx with proximal term
  
Example:
``` bash 
--strategy fedavgm
```

#### Common Strategy Parameters

- `--fraction-fit`: Fraction of clients sampled for training each round (default: 1.0)
- `--fraction-evaluate`: Fraction of clients sampled for evaluation (default: 1.0)
- `--min-fit-clients`: Minimum number of clients selected for training (default: 3)
- `--min-evaluate-clients`: Minimum number of clients selected for evaluation (default: 3)
- `--min-available-clients`: Minimum number of clients required to proceed (default: 3)

Example:
```bash
--fraction-fit 0.8 --min-fit-clients 5
```

#### Strategy-Specific Parameters
**For FedAvgM**:
- `--server-learning-rate`: Server optimizer learning rate (default: 0.1)
- `--server-momentum`: Server optimizer momentum (default: 0.9)

Example: 
```
--strategy fedavgm --server-learning-rate 0.05 --server-momentum 0.8
```

**For FedProx**:
- `--proximal-mu`: Proximal term coefficient (default: 0.1)

Example:
```bash
--strategy fedprox --proximal-mu 0.05
```

#### Simulation Parameters
--num-clients`: Total number of clients in the simulation (default: 3)
--num-rounds`: Number of federated learning rounds (default: 10)

Example:
```bash
--num-clients 10 --num-rounds 50
```

## Project Structure

- `client.py`: Creating Client.
- `data.py`: Loading data (CIFAR-10).
- `model.py`: Defining model architecture.
- `data/`: Folder for data files.