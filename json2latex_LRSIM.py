import json
with open('eval_v4.json', 'r') as file:
    data = json.load(file)

categories = {
    'DT 1024': {
        'Tri-linear': ['TriLinear_DT_LR_SIM_1024_s4'],
        'FNO3D': ['FNO_DT_LR_SIM_1024_s4'],
        'ConvLSTM': ['ConvLSTM_DT_LR_SIM_1024_s4'],
        'Ours': ['PASR_DT_LR_SIM_1024_s4'],
    },
    'DT 512': {
        'Tri-linear': ['TriLinear_DT_LR_SIM_512_s4'],
        'FNO3D': ['FNO_DT_LR_SIM_512_s4'],
        'ConvLSTM': ['ConvLSTM_DT_LR_SIM_512_s4'],
        'Ours': ['PASR_DT_LR_SIM_512_s4'],
    },
    'DT 256': {
        'Tri-linear': ['TriLinear_DT_LR_SIM_256_s4'],
        'FNO3D': ['FNO_DT_LR_SIM_256_s4'],
        'ConvLSTM': ['ConvLSTM_DT_LR_SIM_256_s4'],
        'Ours': ['PASR_DT_LR_SIM_256_s4'],
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

for category, methods in categories.items():
    latex_code += f"     \\multirow{{4}}{{*}}{{{category}}} "
    for method, entries in methods.items():
        latex_code += f"& {method} "
        if entries:
            avg_mse = sum(data[entry]['MSE'] for entry in entries) / len(entries)
            avg_mae = sum(data[entry]['MAE'] for entry in entries) / len(entries)
            avg_rfne = sum(data[entry]['RFNE'] for entry in entries) / len(entries)*100
            avg_in = sum(data[entry]['IN'] for entry in entries) / len(entries)
            avg_ssim = sum(data[entry]['SSIM'] for entry in entries) / len(entries)*100
            avg_psnr = sum(data[entry]['PSNR'] for entry in entries) / len(entries)
        else:
            avg_mse = avg_mae = avg_rfne = avg_in = avg_ssim = avg_psnr =0.00
        latex_code += f"& {avg_mse:.3f} & {avg_mae:.3f} & {avg_rfne:.3}\\% & {avg_in:.3f} & {avg_ssim:.3}\\% & {avg_psnr:.3f} \\\\\n"
    latex_code += "        \\hline\n"

latex_code += "    \\end{tabular}\n"
latex_code += "    \\caption{Table for standard interpolation results (Data driven)}\n"
latex_code += "    \\label{tab:my_label}\n"
latex_code += "\\end{table}"

print(latex_code)