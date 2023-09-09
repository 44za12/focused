import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_report(filename):
    """
    Parse the concentration report and return timestamps and scores.
    """
    timestamps = []
    scores = []

    with open(filename, 'r') as f:
        for line in f:
            timestamp, score = line.strip().split('\t')
            timestamps.append(datetime.fromtimestamp(float(timestamp)))
            scores.append(float(score))

    return timestamps, scores

def generate_insights(filename):
    """
    Generate insights from the concentration report with statistics displayed on the plot.
    """
    timestamps, scores = parse_report(filename)
    scores_np = np.array(scores)
    mean_score = np.mean(scores_np)
    median_score = np.median(scores_np)
    max_score = np.max(scores_np)
    min_score = np.min(scores_np)
    std_dev = np.std(scores_np)
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, scores, label='Concentration Score', color='blue')
    plt.axhline(y=0.6, color='r', linestyle='--', label='Threshold (0.6)')
    plt.title('Concentration Score Over Time')
    plt.xlabel('Time')
    plt.ylabel('Concentration Score')
    plt.legend()
    plt.grid(True)
    stats_text = (f"Mean: {mean_score:.2f}\n"
                  f"Median: {median_score:.2f}\n"
                  f"Max: {max_score:.2f}\n"
                  f"Min: {min_score:.2f}\n"
                  f"Std Dev: {std_dev:.2f}")
    plt.gca().text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                   verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Concentration Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    threshold = 0.6
    below_threshold = scores_np < threshold
    if any(below_threshold):
        print("\nPeriods of Low Concentration:")
        in_low_period = False
        for i, (timestamp, score) in enumerate(zip(timestamps, scores)):
            if score < threshold and not in_low_period:
                start_time = timestamp
                in_low_period = True
            elif score >= threshold and in_low_period:
                end_time = timestamps[i-1]
                print(f"From {start_time} to {end_time}")
                in_low_period = False
        if in_low_period:
            print(f"From {start_time} to {timestamps[-1]}")
    else:
        print("\nNo prolonged periods of low concentration detected.")

if __name__ == "__main__":
    generate_insights("concentration_report.txt")
