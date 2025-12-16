import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Data from the provided Excel sheet
domains = ['D1D2', 'D1D3', 'D2D1', 'D2D3', 'D3D1', 'D3D2']
labels = ['#0', '#1', '#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9']

# HARL-MOS and HARL-OS data
harl_mos = {
    'D1D2': [79.07, 90.46, 68.97, 96.03, 84.21, 31.82, 82.42, 94.51, 87.27, 94.85],
    'D1D3': [85.04, 83.27, 60.93, 85.02, 48.13, 57.22, 79.93, 78.05, 77.97, 82.59],
    'D2D1': [67.74, 74.08, 32.09, 85.00, 35.62, 2.38, 27.61, 74.63, 63.47, 79.48],
    'D2D3': [80.17, 78.70, 44.55, 91.56, 36.36, 3.25, 66.33, 68.97, 66.04, 72.96],
    'D3D1': [79.75, 89.58, 71.67, 92.27, 55.70, 70.67, 55.49, 65.35, 76.22, 85.11],
    'D3D2': [81.54, 93.01, 58.06, 99.19, 81.82, 60.87, 90.26, 95.79, 84.00, 95.24]
}

harl_os = {
    'D1D2': [70.59, 77.52, 53.33, 95.65, 84.45, 32.56, 85.11, 93.18, 88.46, 82.62],
    'D1D3': [86.14, 79.32, 68.33, 87.98, 35.26, 59.92, 76.31, 74.40, 74.87, 78.51],
    'D2D1': [50.23, 55.41, 2.13, 83.59, 17.24, 0.00, 25.05, 58.74, 49.28, 60.02],
    'D2D3': [69.97, 68.12, 2.26, 85.98, 21.38, 0.00, 57.82, 53.12, 46.29, 64.16],
    'D3D1': [81.33, 85.54, 72.94, 91.18, 62.58, 68.52, 62.09, 64.67, 72.20, 85.71],
    'D3D2': [82.17, 93.98, 51.28, 96.85, 74.70, 63.64, 89.34, 96.30, 86.27, 91.26]
}


# Function to create a radar chart for a given domain
def create_radar_chart(domain, mos_data, os_data, labels):
    # Number of variables
    N = len(labels)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels)
    # Set labels with font size 14
    plt.xticks(angles[:-1], labels, fontsize=24)
    # Draw ylabels
    ax.set_rscale('linear')
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=24)
    plt.ylim(0, 100)

    # Plot HARL-MOS
    mos_values = mos_data[domain] + mos_data[domain][:1]  # Complete the loop
    ax.plot(angles, mos_values, linewidth=2, linestyle='solid', label='HARL-MOS')
    ax.fill(angles, mos_values, 'b', alpha=0.1)

    # Plot HARL-OS
    os_values = os_data[domain] + os_data[domain][:1]  # Complete the loop
    ax.plot(angles, os_values, linewidth=2, linestyle='solid', label='HARL-OS')
    ax.fill(angles, os_values, 'r', alpha=0.1)

    # Add a title
    plt.title(f' {domain}', size=24, color='black', y=1.1)

    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=24)

    # Save the plot
    plt.savefig(f'radar_chart_{domain}.png', bbox_inches='tight')
    plt.close()


# Generate radar charts for each domain
for domain in domains:
    create_radar_chart(domain, harl_mos, harl_os, labels)