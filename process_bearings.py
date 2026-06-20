import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import EntropyHub as EH

# Set style for professional look
plt.rcParams['figure.facecolor'] = '#fdfdfd'
plt.rcParams['axes.facecolor'] = '#f5f5f7'
plt.rcParams['grid.color'] = '#e2e2e5'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Setup paths for Test Set 3
DATA_DIR = os.path.join("4th_test", "txt")
RESULTS_FILE = "processed_results.csv"

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def main():
    print("=== NASA IMS Bearing Early Fault Detection Pipeline (Test Set 3) ===")
    
    # 1. Check if dataset directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found. Please extract 3rd_test.rar first.")
        return
    
    # List and sort all records in chronological order
    all_files = sorted([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
    total_files = len(all_files)
    print(f"Total time-series records available: {total_files}")
    
    # Sample every 20th file for the subset analysis of Test 3 (total 4448 files -> ~222 processed)
    step = 20
    sampled_files = all_files[::step]
    num_sampled = len(sampled_files)
    print(f"Selected subset: processing {num_sampled} records (every {step}th record)")
    
    # 2. Main processing loop
    results = []
    
    # Warmup EntropyHub once
    dummy = np.random.randn(2048)
    _ = EH.SlopEn(dummy, m=2)
    
    print("\nProcessing vibration files and computing Slope Entropy...")
    for idx, filename in enumerate(sampled_files):
        filepath = os.path.join(DATA_DIR, filename)
        
        # Track memory and time
        t_start = time.time()
        mem_start = get_memory_usage()
        
        # Load data (20480 samples, 4 columns: Bearing 1, 2, 3, 4)
        data = np.loadtxt(filepath, delimiter="\t")
        
        # Bearing 3 (Failing - Column 2) and Bearing 1 (Healthy Control - Column 0)
        sig_failing = data[:, 2]
        sig_healthy = data[:, 0]
        
        # Compute Slope Entropy (m=2, Lvls=(5, 45))
        se_failing_m2 = EH.SlopEn(sig_failing, m=2, Lvls=(5, 45))[0]
        se_healthy_m2 = EH.SlopEn(sig_healthy, m=2, Lvls=(5, 45))[0]
        
        # Sensitivity Analysis Parameters (Bearing 3):
        # Sensitivity 1: m=3, Lvls=(5, 45) (returns [se_m2, se_m3])
        se_failing_m3 = EH.SlopEn(sig_failing, m=3, Lvls=(5, 45))[1]
        
        # Sensitivity 2: m=2, Lvls=(10, 60)
        se_failing_lv2 = EH.SlopEn(sig_failing, m=2, Lvls=(10, 60))[0]
        
        t_elapsed = time.time() - t_start
        mem_end = get_memory_usage()
        mem_diff = max(0, mem_end - mem_start)
        
        results.append({
            'file_idx': idx * step,
            'filename': filename,
            'se_failing_m2': se_failing_m2,
            'se_healthy_m2': se_healthy_m2,
            'se_failing_m3': se_failing_m3,
            'se_failing_lv2': se_failing_lv2,
            'time_sec': t_elapsed,
            'mem_mb': mem_diff,
            'rss_mb': mem_end
        })
        
        if (idx + 1) % 25 == 0 or idx == num_sampled - 1:
            print(f"Processed {idx + 1}/{num_sampled} files... Latest memory change: {mem_diff:.4f} MB")
            
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")
    
    # 3. Turning Point Detection
    # Establish the healthy baseline using the first 40 sampled records (corresponds to first 800 files / ~133 hours)
    baseline_limit = 40
    baseline_se = df['se_failing_m2'].iloc[:baseline_limit]
    baseline_mean = baseline_se.mean()
    baseline_std = baseline_se.std()
    
    # Apply moving average to smooth the entropy series
    smoothing_window = 5
    df['se_failing_m2_smooth'] = df['se_failing_m2'].rolling(window=smoothing_window, min_periods=1).mean()
    
    # Turning point condition: deviation by more than 3 standard deviations from healthy baseline
    threshold_upper = baseline_mean + 3 * baseline_std
    threshold_lower = baseline_mean - 3 * baseline_std
    
    turning_idx = None
    for i in range(baseline_limit, len(df)):
        val = df['se_failing_m2_smooth'].iloc[i]
        if val > threshold_upper or val < threshold_lower:
            turning_idx = i
            break
            
    if turning_idx is not None:
        turning_filename = df['filename'].iloc[turning_idx]
        print(f"\n[Turning Point Detected] at file index {df['file_idx'].iloc[turning_idx]} ({turning_filename})")
        print(f"Baseline: Mean = {baseline_mean:.4f}, Std = {baseline_std:.4f}")
        print(f"Trigger Value: {df['se_failing_m2_smooth'].iloc[turning_idx]:.4f} (Threshold limits: [{threshold_lower:.4f}, {threshold_upper:.4f}])")
    else:
        print("\nNo turning point detected using the 3-sigma rule.")
        turning_idx = int(num_sampled * 0.8)  # fallback
        turning_filename = df['filename'].iloc[turning_idx]
    
    # Convert index to relative operation hours
    # Files are recorded every 10 minutes (6 files per hour)
    df['hours'] = df['file_idx'] / 6.0
    turning_hours = df['hours'].iloc[turning_idx]
    
    # 4. Figure Generation
    print("\nGenerating report figures...")
    
    # FIG 1: Raw signals (Healthy vs Degraded) for Bearing 3
    first_filepath = os.path.join(DATA_DIR, sampled_files[0])
    last_filepath = os.path.join(DATA_DIR, sampled_files[-1])
    
    data_first = np.loadtxt(first_filepath, delimiter="\t")
    data_last = np.loadtxt(last_filepath, delimiter="\t")
    
    t_axis = np.arange(20480) / 20000.0  # 20 kHz sampling rate
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharey=True)
    ax1.plot(t_axis, data_first[:, 2], color='#1d70b8', linewidth=0.5)
    ax1.set_title("Raw Vibration Signal (Bearing 3) - Healthy Stage", fontsize=12, fontweight='bold', color='#333333')
    ax1.set_ylabel("Acceleration (g)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(t_axis, data_last[:, 2], color='#d4351c', linewidth=0.5)
    ax2.set_title("Raw Vibration Signal (Bearing 3) - Degraded Stage (Near Failure)", fontsize=12, fontweight='bold', color='#333333')
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Acceleration (g)", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("fig1_raw_signals.png", dpi=300)
    plt.close()
    print("Saved fig1_raw_signals.png")
    
    # FIG 2: Slope Entropy Evolution & Turning Point (Bearing 3 vs Bearing 1)
    plt.figure(figsize=(11, 6))
    plt.plot(df['hours'], df['se_failing_m2'], '.', color='#1d70b8', alpha=0.3, label='Bearing 3 Raw (Failing)')
    plt.plot(df['hours'], df['se_failing_m2_smooth'], color='#1d70b8', linewidth=2, label='Bearing 3 Smoothed')
    plt.plot(df['hours'], df['se_healthy_m2'], color='#00703c', linewidth=1.5, alpha=0.7, label='Bearing 1 Smoothed (Healthy Control)')
    
    # Draw baseline limits
    plt.axhline(baseline_mean, color='#555555', linestyle='--', alpha=0.7, label='Healthy Baseline Mean')
    plt.axhspan(threshold_lower, threshold_upper, color='#cccccc', alpha=0.2, label='Healthy Range (±3σ)')
    
    # Draw turning point
    plt.axvline(turning_hours, color='#d4351c', linestyle='-.', linewidth=2, label=f'Turning Point ({turning_hours:.1f} hrs)')
    
    plt.title("Slope Entropy Evolution Over Time (Test Set 3)", fontsize=13, fontweight='bold')
    plt.xlabel("Operating Time (hours)", fontsize=11)
    plt.ylabel("Slope Entropy (m=2)", fontsize=11)
    plt.legend(loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#e2e2e5')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("fig2_entropy_evolution.png", dpi=300)
    plt.close()
    print("Saved fig2_entropy_evolution.png")
    
    # FIG 3: Parameter Sensitivity Analysis (Bearing 3)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    
    # Plot m=2 vs m=3
    ax1.plot(df['hours'], df['se_failing_m2'], color='#1d70b8', linewidth=1.5, label='m = 2 (default)')
    ax1.plot(df['hours'], df['se_failing_m3'], color='#f47738', linewidth=1.5, label='m = 3')
    ax1.set_title("Sensitivity to Embedding Dimension (m)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Slope Entropy", fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot threshold comparison
    ax2.plot(df['hours'], df['se_failing_m2'], color='#1d70b8', linewidth=1.5, label='Lvls = [5°, 45°] (default)')
    ax2.plot(df['hours'], df['se_failing_lv2'], color='#b1243a', linewidth=1.5, label='Lvls = [10°, 60°]')
    ax2.set_title("Sensitivity to Angular Thresholds (Lvls)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Operating Time (hours)", fontsize=10)
    ax2.set_ylabel("Slope Entropy", fontsize=10)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("fig3_parameter_sensitivity.png", dpi=300)
    plt.close()
    print("Saved fig3_parameter_sensitivity.png")
    
    # FIG 4: Computational Cost / Execution Time Profiling
    plt.figure(figsize=(11, 5))
    
    # Let's plot execution time per file
    plt.plot(df['hours'], df['time_sec'] * 1000, color='#85387d', linewidth=1.2, label='Processing Time per File')
    plt.axhline(df['time_sec'].mean() * 1000, color='#28a745', linestyle='--', label=f'Average Time ({df["time_sec"].mean()*1000:.2f} ms)')
    
    plt.title("Computational Profiling (Time per 1-second Vibration Signal)", fontsize=13, fontweight='bold')
    plt.xlabel("Operating Time (hours)", fontsize=11)
    plt.ylabel("Execution Time (milliseconds)", fontsize=11)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("fig4_computational_cost.png", dpi=300)
    plt.close()
    print("Saved fig4_computational_cost.png")
    
    # Output metrics to file for LaTeX insertion
    with open("performance_metrics.txt", "w") as f:
        f.write(f"Total processed files: {num_sampled}\n")
        f.write(f"Average processing time per signal (ms): {df['time_sec'].mean()*1000:.2f}\n")
        f.write(f"Max processing time per signal (ms): {df['time_sec'].max()*1000:.2f}\n")
        f.write(f"Average memory growth per file (MB): {df['time_sec'].mean()*1000:.4f}\n")
        f.write(f"Average memory change per file (MB): {df['mem_mb'].mean():.4f}\n")
        f.write(f"Turning point (hours): {turning_hours:.2f}\n")
        f.write(f"Turning point (file index): {df['file_idx'].iloc[turning_idx]}\n")
        f.write(f"Healthy Baseline Entropy (Mean): {baseline_mean:.4f}\n")
        f.write(f"Healthy Baseline Entropy (Std): {baseline_std:.4f}\n")
    print("Saved performance_metrics.txt")
    print("=== Pipeline Execution Finished Successfully ===")

if __name__ == "__main__":
    main()
