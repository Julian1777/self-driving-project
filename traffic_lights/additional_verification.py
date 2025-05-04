import os
import random
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import tensorflow as tf

# Constants
MERGED_DS = "merged_dataset"
STATES = ["green", "red", "yellow", "off"]
SAMPLES_PER_STATE = 5

def verify_with_focused_regions():
    """Verify images with more sophisticated region analysis"""
    fig, axes = plt.subplots(len(STATES), SAMPLES_PER_STATE, figsize=(15, 12))
    
    for i, state in enumerate(STATES):
        state_dir = os.path.join(MERGED_DS, state)
        if os.path.exists(state_dir):
            files = [f for f in os.listdir(state_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(files) >= SAMPLES_PER_STATE:
                samples = random.sample(files, SAMPLES_PER_STATE)
                
                for j, sample in enumerate(samples):
                    img_path = os.path.join(state_dir, sample)
                    img = cv.imread(img_path)
                    if img is None:
                        axes[i, j].text(0.5, 0.5, "Image load error", ha='center')
                        continue
                        
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    
                    # Show original image
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f"Label: {state}\nFile: {sample[:10]}...")
                    axes[i, j].axis('off')
                    
                    # Overlay color detection masks at 30% opacity
                    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
                    h, s, v = cv.split(hsv)
                    
                    # Create masks for different colors
                    red_mask1 = cv.inRange(hsv, (0, 100, 100), (10, 255, 255))
                    red_mask2 = cv.inRange(hsv, (170, 100, 100), (180, 255, 255))
                    red_mask = cv.bitwise_or(red_mask1, red_mask2)
                    
                    yellow_mask = cv.inRange(hsv, (20, 100, 100), (30, 255, 255))
                    green_mask = cv.inRange(hsv, (40, 100, 100), (80, 255, 255))
                    
                    # Apply masks with overlay
                    red_overlay = np.zeros_like(img)
                    red_overlay[red_mask > 0] = [255, 0, 0]
                    
                    yellow_overlay = np.zeros_like(img)
                    yellow_overlay[yellow_mask > 0] = [255, 255, 0]
                    
                    green_overlay = np.zeros_like(img)
                    green_overlay[green_mask > 0] = [0, 255, 0]
                    
                    # Combine overlays
                    overlay = cv.addWeighted(img, 0.7, red_overlay, 0.3, 0)
                    overlay = cv.addWeighted(overlay, 1.0, yellow_overlay, 0.3, 0)
                    overlay = cv.addWeighted(overlay, 1.0, green_overlay, 0.3, 0)
                    
                    # Display counts in corner
                    red_count = np.count_nonzero(red_mask)
                    yellow_count = np.count_nonzero(yellow_mask)
                    green_count = np.count_nonzero(green_mask)
                    
                    total_colored = red_count + yellow_count + green_count
                    if total_colored > 0:
                        red_pct = red_count / total_colored * 100
                        yellow_pct = yellow_count / total_colored * 100
                        green_pct = green_count / total_colored * 100
                        color_text = f"R:{red_pct:.1f}% Y:{yellow_pct:.1f}% G:{green_pct:.1f}%"
                    else:
                        color_text = "No colors detected"
                        
                    axes[i, j].text(0.02, 0.98, color_text, 
                                   transform=axes[i, j].transAxes,
                                   color='white', fontsize=8,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.suptitle("Traffic Light Visual Verification with Color Detection", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()

if __name__ == "__main__":
    verify_with_focused_regions()