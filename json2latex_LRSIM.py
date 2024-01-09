import json
with open('eval_v5.json', 'r') as file:
    data = json.load(file)

categories = {
    'DT x4': {
        'Tri-linear': ['TriLinear_DT_1024_s4_v0'],
        'FNO3D': ['FNO_DT_1024_s4_v0'],
        'ConvLSTM': ['ConvLSTM_DT_1024_s4_v0'],
        'Ours (Euler)': ['PASR_DT_1024_s4_v0_euler'],
        'Ours (RK4)': ['PASR_DT_1024_s4_v0_rk4'],
    },
    'DT x8': {
        'Tri-linear': ['TriLinear_DT_1024_s8_v0'],
        'FNO3D': ['FNO_DT_1024_s8_v0'],
        'ConvLSTM': ['ConvLSTM_DT_1024_s8_v0'],
        'Ours (Euler)': ['PASR_DT_1024_s8_v0_euler'],
        'Ours (RK4)': ['PASR_DT_1024_s8_v0_rk4'],
    },
    'DT x16': {
        'Tri-linear': ['TriLinear_DT_1024_s16_v0'],
        'FNO3D': ['FNO_DT_1024_s16_v0'],
        'ConvLSTM': ['ConvLSTM_DT_1024_s16_v0'],
        'Ours (Euler)': ['PASR_DT_1024_s16_v0_euler'],
        'Ours (RK4)': ['PASR_DT_1024_s16_v0_rk4'],
    },
    # 'Climate': {
    #     'Tri-linear': ['Tri_Climate_None'],
    #     'FNO3D': ['FNO_Climate_None'],
    #     'ConvLSTM': ['ConvLSTM_Climate_None'],
    #     'Ours (Euler)': ['PASR_small_climate_euler'],
    #     'Ours (RK4)': ['PASR_small_climate_rk4'],
    # },
}

latex_code = "\\begin{table}[h!]\n"
latex_code += "    \\centering\n"
latex_code += "    \\begin{tabular}{c|c|ccccccc}\n"
latex_code += "    \\hline\n"
latex_code += "        & & MSE $\\downarrow$ & MAE $\\downarrow$ & RFNE $\\downarrow$  & IN $\\downarrow$ & SSIM $\\uparrow$ & PSNR $\\uparrow$  \\\\\n"
latex_code += "    \\hline\n"
# Tracking the lowest and highest values for each category
track_values = {category: {'lowest_mse': float('inf'), 'lowest_mae': float('inf'),
                           'lowest_rfne': float('inf'), 'lowest_in': float('inf'),
                           'highest_ssim': float('-inf'), 'highest_psnr': float('-inf')}
                for category in categories.keys()}

# First pass to find the lowest and highest values
for category, methods in categories.items():
    for method, entries in methods.items():
        if entries:
            avg_mse = sum(data[entry]['MSE'] for entry in entries) / len(entries)
            avg_mae = sum(data[entry]['MAE'] for entry in entries) / len(entries)
            avg_rfne = sum(data[entry]['RFNE'] for entry in entries) / len(entries)*100
            avg_in = sum(data[entry]['IN'] for entry in entries) / len(entries)
            avg_ssim = sum(data[entry]['SSIM'] for entry in entries) / len(entries)*100
            avg_psnr = sum(data[entry]['PSNR'] for entry in entries) / len(entries)

            # Update tracking values
            track = track_values[category]
            track['lowest_mse'] = min(track['lowest_mse'], avg_mse)
            track['lowest_mae'] = min(track['lowest_mae'], avg_mae)
            track['lowest_rfne'] = min(track['lowest_rfne'], avg_rfne)
            track['lowest_in'] = min(track['lowest_in'], avg_in)
            track['highest_ssim'] = max(track['highest_ssim'], avg_ssim)
            track['highest_psnr'] = max(track['highest_psnr'], avg_psnr)

# Generating LaTeX code with highlighting
for category, methods in categories.items():
    latex_code += f"     \\multirow{{5}}{{*}}{{{category}}} "
    for method, entries in methods.items():
        latex_code += f"& {method} "
        if entries:
            avg_mse = sum(data[entry]['MSE'] for entry in entries) / len(entries)
            avg_mae = sum(data[entry]['MAE'] for entry in entries) / len(entries)
            avg_rfne = sum(data[entry]['RFNE'] for entry in entries) / len(entries)*100
            avg_in = sum(data[entry]['IN'] for entry in entries) / len(entries)
            avg_ssim = sum(data[entry]['SSIM'] for entry in entries) / len(entries)*100
            avg_psnr = sum(data[entry]['PSNR'] for entry in entries) / len(entries)

            # Apply highlighting
            mse_str = f"\\textbf{{{avg_mse:.3f}}}" if avg_mse == track_values[category]['lowest_mse'] else f"{avg_mse:.3f}"
            mae_str = f"\\textbf{{{avg_mae:.3f}}}" if avg_mae == track_values[category]['lowest_mae'] else f"{avg_mae:.3f}"
            rfne_str = f"\\textbf{{{avg_rfne:.3f}}}\\%" if avg_rfne == track_values[category]['lowest_rfne'] else f"{avg_rfne:.3f}\\%"
            in_str = f"\\textbf{{{avg_in:.3f}}}" if avg_in == track_values[category]['lowest_in'] else f"{avg_in:.3f}"
            ssim_str = f"\\textbf{{{avg_ssim:.3f}}}\\%" if avg_ssim == track_values[category]['highest_ssim'] else f"{avg_ssim:.3f}\\%"
            psnr_str = f"\\textbf{{{avg_psnr:.3f}}}" if avg_psnr == track_values[category]['highest_psnr'] else f"{avg_psnr:.3f}"

            latex_code += f"& {mse_str} & {mae_str} & {rfne_str} & {in_str} & {ssim_str} & {psnr_str} \\\\\n"
        else:
            latex_code += "& 0.000 & 0.000 & 0.000\\% & 0.000 & 0.000\\% & 0.000 \\\\\n"
    latex_code += "        \\hline\n"

latex_code += "    \\end{tabular}\n"
latex_code += "    \\caption{Table for standard interpolation results (Data driven)}\n"
latex_code += "    \\label{tab:my_label}\n"
latex_code += "\\end{table}"

print(latex_code)