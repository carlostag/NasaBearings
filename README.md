# Execution Instructions and Reproducibility Guide

This guide describes how to set up the environment and run the early fault detection pipeline on a Raspberry Pi (Pi 4 or Pi 5) running Raspberry Pi OS, satisfying the reproducibility and submission requirements of the home assignment.

---

## 1. Prerequisites and File Transfer

Before starting, ensure that all project files and the dataset have been transferred to your Raspberry Pi.

### File Structure on the RPi:
Your project directory on the Pi should contain:
```text
project_directory/
│
├── 4th_test/
│   └── txt/                     # Contains the 6,324 raw NASA IMS txt files
│
├── process_bearings.py          # The main python processing pipeline
├── data_exploration.ipynb       # Reorganized Jupyter exploration notebook
├── requirements.txt             # Python dependency list
└── README.md                    # This instruction guide
```

*Note: If you have the dataset as a archive (e.g. `3rd_test.rar`), you can extract it on the Pi using `unrar x 3rd_test.rar` or `unzip` depending on the format.*

---

## 2. Environment Setup

It is highly recommended to run the code in a Python Virtual Environment (`venv`) to avoid dependency conflicts.

1. **Open a terminal** on your Raspberry Pi (or connect via SSH).
2. **Navigate** to your project directory:
   ```bash
   cd path/to/project_directory
   ```
3. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```
4. **Activate the virtual environment**:
   - On Raspberry Pi / Linux:
     ```bash
     source venv/bin/activate
     ```
5. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```
6. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Running the Pipeline

Once the dependencies are installed, you can execute the feature-extraction and entropy computation pipeline.

1. **Run the Python script**:
   ```bash
   python process_bearings.py
   ```
2. **What the script does**:
   - Loads and parses the chronological records of Test Set 3.
   - Sparsely samples every 20th record (total 317 snapshots).
   - Computes Slope Entropy ($m=2$ and $m=3$) and angular sensitivity thresholds for the failing Bearing 3 and healthy control Bearing 1.
   - Profiles execution time (seconds) and memory usage (RSS MB) per snapshot.
   - Runs the $3\sigma$ statistical turning point detection logic.
   - Saves all quantitative indicators to `processed_results.csv`.
   - Saves summary metrics (average latencies, memory footprint, detected turning point) to `performance_metrics.txt`.
   - Generates and saves four analytical figures:
     1. `fig1_raw_signals.png`: Time-domain healthy vs. degraded waveforms.
     2. `fig2_entropy_evolution.png`: Entropy trend and $3\sigma$ threshold trigger.
     3. `fig3_parameter_sensitivity.png`: Sensitivity analyses ($m$ and thresholds).
     4. `fig4_computational_cost.png`: Execution time and memory footprint curves.

---

## 4. Running the Jupyter Notebook on the Pi

To run the interactive analysis notebook (`data_exploration.ipynb`) directly on the Raspberry Pi:

1. **Install Jupyter** inside your activated virtual environment:
   ```bash
   pip install jupyter
   ```
2. **Launch the Jupyter Notebook server**:
   - If running headless (via SSH), launch without a browser:
     ```bash
     jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
     ```
   - Copy the URL token generated in the terminal and open it on your PC's browser (replacing `localhost` or the IP with the Raspberry Pi's network IP, e.g. `http://<rpi_ip>:8888/?token=...`).
3. **Explore the sections**:
   - **7.1 to 7.6** map directly to the technical requirements, including the time/frequency domain plotting, FFT computation, statistical trigger logic, and a quantitative stage comparison table.
   - **Section 10** contains the interactive parameters playground.
