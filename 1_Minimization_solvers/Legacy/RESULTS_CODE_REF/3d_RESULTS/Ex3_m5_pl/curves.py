import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the main directory as the current working directory
main_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(main_directory)

# Define minimization methods and their styles
minimization_methods = {
    "BFGS": {"linestyle": "-", "marker": None, "linewidth": 1.0},
    "LBFGS500": {"linestyle": "--", "marker": None, "linewidth": 1.5},
    "TR": {"linestyle": "", "marker": "o", "linewidth": 1, "markerfacecolor": "none"},
    "TR-icho": {"linestyle": "", "marker": "x", "linewidth": 1}
}

# Initialize lists to store data for plotting
data_fx = {}
data_fy = {}
data_fz = {}
vertical_lines = []

# Loop through each minimization method and read the corresponding CSV file
for method, style in minimization_methods.items():
    # Find folders containing the method substring
    folders = [f for f in os.listdir(main_directory) if os.path.isdir(f) and ("_" + method + "_") in f]
    
    for folder in folders:
        folder_path = os.path.join(main_directory, folder)
        file_path = os.path.join(folder_path, 'sum_fint.csv')
        details_file_path = os.path.join(folder_path, 'ctct0iters_details.csv')
        
        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path, header=None, names=['t', 'Fx', 'Fy', 'Fz'])
            
            # Store the data for plotting
            if method not in data_fx:
                data_fx[method] = (df['t'], df['Fx'])
                data_fy[method] = (df['t'], df['Fy'])
                data_fz[method] = (df['t'], df['Fz'])
            else:
                data_fx[method] = (pd.concat([data_fx[method][0], df['t']]), pd.concat([data_fx[method][1], df['Fx']]))
                data_fy[method] = (pd.concat([data_fy[method][0], df['t']]), pd.concat([data_fy[method][1], df['Fy']]))
                data_fz[method] = (pd.concat([data_fz[method][0], df['t']]), pd.concat([data_fz[method][1], df['Fz']]))
        
        if os.path.exists(details_file_path):
            # Read the details CSV file
            with open(details_file_path, 'r') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    if 'MINIMIZATION' in line:
                        try:
                            vertical_lines.append(float(lines[i + 1].split(',')[0]))
                        except (IndexError, ValueError):
                            continue

case = os.path.basename(main_directory)

# Plot Fx vs t
plt.figure(figsize=(10, 5))
for method, (t, Fx) in data_fx.items():
    style = minimization_methods[method]
    plt.plot(t, Fx, label=method, linestyle=style["linestyle"], marker=style["marker"], linewidth=style["linewidth"], markerfacecolor=style.get("markerfacecolor", None))
if vertical_lines:
    for x in vertical_lines:
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=vertical_lines[0], color='gray', linestyle='--', linewidth=0.5, label='minimization')
plt.xlabel('t')
plt.ylabel('Fx')
plt.title(f'{case} - Fx vs t')
plt.legend()
plt.grid(False)
plt.savefig(f'{case}_Fx_vs_t.png')
plt.savefig(f'{case}_Fx_vs_t.pdf')
plt.show()
plt.close()

# Plot Fy vs t
plt.figure(figsize=(10, 5))
for method, (t, Fy) in data_fy.items():
    style = minimization_methods[method]
    plt.plot(t, Fy, label=method, linestyle=style["linestyle"], marker=style["marker"], linewidth=style["linewidth"], markerfacecolor=style.get("markerfacecolor", None))
if vertical_lines:
    for x in vertical_lines:
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=vertical_lines[0], color='gray', linestyle='--', linewidth=0.5, label='minimization')
plt.xlabel('t')
plt.ylabel('Fy')
plt.title(f'{case} - Fy vs t')
plt.legend()
plt.grid(False)
plt.savefig(f'{case}_Fy_vs_t.png')
plt.savefig(f'{case}_Fy_vs_t.pdf')
plt.close()

# Plot Fz vs t
plt.figure(figsize=(10, 5))
for method, (t, Fz) in data_fz.items():
    style = minimization_methods[method]
    plt.plot(t, Fz, label=method, linestyle=style["linestyle"], marker=style["marker"], linewidth=style["linewidth"], markerfacecolor=style.get("markerfacecolor", None))
if vertical_lines:
    for x in vertical_lines:
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=vertical_lines[0], color='gray', linestyle='--', linewidth=0.5, label='minimization')
plt.xlabel('t')
plt.ylabel('Fz')
plt.title(f'{case} - Fz vs t')
plt.legend()
plt.grid(False)
plt.savefig(f'{case}_Fz_vs_t.png')
plt.savefig(f'{case}_Fz_vs_t.pdf')
plt.close()