import os
import random
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from pathlib import Path

# Define constants
DTLD_DIR = "dtld_dataset/cropped_dataset"
LISA_DIR = "lisa_dataset/cropped_dataset"
MERGED_DIR = "merged_dataset"
STATES = ["green", "red", "yellow", "off"]
SAMPLES_PER_STATE = 4  # Reduced to ensure better spacing

def count_samples_per_class(dataset_dir):
    """Count the number of images in each state directory"""
    counts = {}
    total = 0
    
    for state in STATES:
        state_dir = os.path.join(dataset_dir, state)
        if os.path.exists(state_dir):
            image_files = [f for f in os.listdir(state_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            counts[state] = len(image_files)
            total += len(image_files)
        else:
            counts[state] = 0
    
    return counts, total

def load_and_display_samples(dataset_dir, dataset_name, ax_row):
    """Load random samples from each state in a dataset and display them"""
    print(f"\nChecking {dataset_name} dataset...")
    
    # Check if directory exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Directory not found: {dataset_dir}")
        return False
    
    found_samples = False
    for col, state in enumerate(STATES):
        state_dir = os.path.join(dataset_dir, state)
        if os.path.exists(state_dir):
            image_files = [f for f in os.listdir(state_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                found_samples = True
                print(f"✓ Found {len(image_files)} images for state: {state}")
                
                # Select random samples
                samples = random.sample(image_files, min(SAMPLES_PER_STATE, len(image_files)))
                
                # Display each sample
                for i, sample in enumerate(samples):
                    img_path = os.path.join(state_dir, sample)
                    img = cv.imread(img_path)
                    
                    if img is not None:
                        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                        ax = sample_axes[ax_row + i, col]
                        ax.imshow(img)
                        ax.set_title(f"{state}")
                        filename = Path(img_path).name
                        ax.set_xlabel(f"{filename[:8]}..." if len(filename) > 8 else filename, fontsize=8)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        print(f"❌ Failed to load: {img_path}")
            else:
                print(f"❌ No images found for state: {state}")
        else:
            print(f"❌ State directory not found: {state_dir}")
    
    return found_samples

# Create a simple figure structure without using GridSpec
plt.figure(figsize=(16, 16))

# Create sample grid
sample_axes = plt.subplots(2 * SAMPLES_PER_STATE, len(STATES), figsize=(16, 12))[1]

# Add dataset headers
fig = plt.gcf()
for i, state in enumerate(STATES):
    fig.text(0.125 + i * 0.22, 0.96, f"{state}", ha='center', fontsize=16, fontweight='bold')

fig.text(0.5, 0.98, "Dataset Verification", ha='center', fontsize=18, fontweight='bold')
fig.text(0.5, 0.75, "DTLD Dataset", ha='center', fontsize=16, fontweight='bold')
fig.text(0.5, 0.49, "LISA Dataset", ha='center', fontsize=16, fontweight='bold')

# Load and display samples from each dataset
dtld_samples = load_and_display_samples(DTLD_DIR, "DTLD", 0)
lisa_samples = load_and_display_samples(LISA_DIR, "LISA", SAMPLES_PER_STATE)

# Adjust spacing for sample grid
plt.tight_layout(rect=[0.03, 0.25, 0.97, 0.95])
plt.subplots_adjust(wspace=0.3, hspace=0.5)

# Create a separate figure for distribution
plt.figure(figsize=(16, 6))
dist_ax = plt.gca()

# Add class distribution visualization
dtld_counts, dtld_total = count_samples_per_class(DTLD_DIR)
lisa_counts, lisa_total = count_samples_per_class(LISA_DIR)

x = np.arange(len(STATES))  # the label locations
width = 0.35  # the width of the bars

# Create distribution bar chart
rects1 = dist_ax.bar(x - width/2, [dtld_counts.get(state, 0) for state in STATES], width, label='DTLD')
rects2 = dist_ax.bar(x + width/2, [lisa_counts.get(state, 0) for state in STATES], width, label='LISA')

# Add labels and legends
dist_ax.set_title('Class Distribution by Dataset', fontsize=16)
dist_ax.set_ylabel('Number of Images', fontsize=14)
dist_ax.set_xticks(x)
dist_ax.set_xticklabels(STATES, fontsize=12)
dist_ax.legend(fontsize=12)

# Add count numbers on bars
def autolabel(rects):
    """Attach a text label above each bar showing its height"""
    for rect in rects:
        height = rect.get_height()
        dist_ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

# Add percentages in parentheses
for i, rect in enumerate(rects1):
    state = STATES[i]
    pct = dtld_counts.get(state, 0) / dtld_total * 100 if dtld_total > 0 else 0
    dist_ax.annotate(f'({pct:.1f}%)',
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 18),  # 18 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for i, rect in enumerate(rects2):
    state = STATES[i]
    pct = lisa_counts.get(state, 0) / lisa_total * 100 if lisa_total > 0 else 0
    dist_ax.annotate(f'({pct:.1f}%)',
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 18),  # 18 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Summary
print("\n--- Summary ---")
if not dtld_samples and not lisa_samples:
    print("❌ No valid samples found in either dataset")
elif not dtld_samples:
    print("❌ No valid samples found in DTLD dataset")
elif not lisa_samples:
    print("❌ No valid samples found in LISA dataset")
else:
    print("✓ Samples found in both datasets")
    print("\nClass distribution:")
    print("DTLD dataset:")
    for state in STATES:
        pct = dtld_counts.get(state, 0) / dtld_total * 100 if dtld_total > 0 else 0
        print(f"  - {state}: {dtld_counts.get(state, 0)} ({pct:.1f}%)")
    print(f"  - Total: {dtld_total}")
    
    print("\nLISA dataset:")
    for state in STATES:
        pct = lisa_counts.get(state, 0) / lisa_total * 100 if lisa_total > 0 else 0
        print(f"  - {state}: {lisa_counts.get(state, 0)} ({pct:.1f}%)")
    print(f"  - Total: {lisa_total}")