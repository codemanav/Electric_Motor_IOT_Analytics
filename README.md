# Electric Motor IoT Analytics

This project analyzes electric motor telemetry and applies Hidden Markov
Models (HMMs) to model discrete operating states and observations.

## Dataset

The dataset is stored in [`data/df_clean.csv`](./data/df_clean.csv) and
contains 9,183 cleaned observations with these sensor and target fields:

- `coolant` - coolant temperature
- `stator_winding` - stator winding temperature
- `motor_speed` - motor speed
- `i_q` - q-axis current
- `pm` - permanent-magnet temperature
- `torque` - measured motor torque

The data represents the type of telemetry used in industrial IoT monitoring
and motor-condition analysis.

## HMM work

- [`task3 & task3.2.py`](./task3%20%26%20task3.2.py) generates a synthetic
  four-state, three-symbol HMM and trains a categorical HMM.
- [`task3.1.py`](./task3.1.py) contains a Viterbi most-probable-state-path
  implementation.
- [`task3.2.py`](./task3.2.py) trains a four-state multinomial HMM on an
  observation sequence.

The scripts demonstrate transition matrices, emission matrices, initial-state
distributions, sequence generation, parameter learning, and state decoding.

## Setup

Install the Python dependencies:

```bash
pip install numpy pandas hmmlearn
```

Run the scripts from the project root:

```bash
python "task3 & task3.2.py"
python task3.1.py
python task3.2.py
```

Some scripts are assignment fragments and expect variables such as `P`, `B`,
`pi`, or `observations` to be initialized in the surrounding workflow.

## Project context

This is an academic IoT and machine-learning assignment focused on electric
motor telemetry and HMM concepts. It is not a complete production IoT
pipeline: it does not include live sensor ingestion, cloud storage,
monitoring dashboards, or deployment infrastructure.
