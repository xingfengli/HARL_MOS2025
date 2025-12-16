import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Data from the provided Excel sheet
domains = ['S1S2', 'S2S1']
labels = ['AMCR', 'AMRO', 'BLJA', 'ECMC', 'ECMK', 'FFCR']

# HARL-MOS and HARL-OS data
harl_mos = {
    'S1S2': [84.07, 95.06, 65.08, 77.98, 86.69, 96.93],
    'S2S1': [81.92, 84.72, 85.32, 79.84, 90.59, 95.76]
}

harl_os = {
    'S1S2': [89.86, 91.51, 50.39, 71.49, 87.69, 95.99],
    'S2S1': [84.54, 64.74, 74.52, 79.65, 91.28, 81.18]
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