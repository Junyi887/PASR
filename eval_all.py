import h5py
import numpy as np
import glob
import os
import torch
from src.util import *
from src.models import *
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from src.util.eval_util import *
from torchvision import transforms
DATA_INFO2 = {"decay_turb":"/pscratch/sd/j/junyi012/Decay_Turbulence_small",
                    "rbc": "/pscratch/sd/j/junyi012/RBC_small",
                    "burger2D":"../burger2D_10/*/*.h5"}



def get_normalizer(data_name,in_channels=3,normalization_method="meanstd",normalization="True"):
    stats_loader = DataInfoLoader(DATA_INFO[data_name]+"/*/*.h5")
    if normalization == "True":
        mean, std = stats_loader.get_mean_std()
        min,max = stats_loader.get_min_max()
        if in_channels==1:
            mean,std = mean[0].tolist(),std[0].tolist()
            min,max = min[0].tolist(),max[0].tolist()
        elif in_channels==3:
            mean,std = mean.tolist(),std.tolist()
            min,max = min.tolist(),max.tolist()
        elif in_channels==2:
            mean,std = mean[1:].tolist(),std[1:].tolist()
            min,max = min[1:].tolist(),max[1:].tolist()
        if normalization_method =="minmax":
            return min,max
        if normalization_method =="meanstd":
            return mean,std
    else:
        mean, std = [0], [1]
        mean, std = mean *in_channels, std *in_channels
        return mean,std
mean,std = get_normalizer()


def eval_ConvLSTM(data_name,model_path,in_channels,lr_input_tensor,hr_target_tensor,mean,std,num_snapshots=20,climate_normalization = False):
    steps = num_snapshots + 1 
    effective_step = list(range(0, steps))
    model = PhySR(
        n_feats=32,
        n_layers=[1, 2],  # [n_convlstm, n_resblock]
        upscale_factor=[num_snapshots, 4],  # [t_up, s_up]
        shift_mean_paras=[mean.tolist(), std.tolist()],  
        step=steps,
        in_channels=in_channels,
        effective_step=effective_step
    ).cuda()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    init_state = get_init_state(
        batch_size = [lr_input_tensor.shape[0]], 
        hidden_channels = [32], 
        output_size = [[lr_input_tensor.shape[-2], lr_input_tensor.shape[-1]]], # 32, 32 
        mode = 'random')# define the initial states and initial output for model
    with torch.no_grad():
        lres, hres = lr_input_tensor.permute(2,0,1,3,4), hr_target_tensor.permute(2,0,1,3,4) # (b,c,t,h,w) -> (t,b,c,h,w)
        lres, hres = lres.float().cuda(), hres.float().cuda()
        pred = model(lres,init_state)
        mean = torch.from_numpy(mean).float().cuda()
        std = torch.from_numpy(std).float().cuda()
        if climate_normalization == True:
            pred = (pred - mean)/std
            hres = (hres - mean)/std
        print(f"ConvLSTM pred shape {pred.shape}")
        hres,pred = hres.permute(1,0,2,3,4),pred.permute(1,0,2,3,4)
        RFNE,MAE,MSE,IN = get_metric_RFNE(hres,pred)
        # permute back:
        SSIM = get_ssim(hres,pred).cpu().numpy() # from T,b,C,H,W to b,t,C,H,W
        PSNR = get_psnr(hres,pred).cpu().numpy()
    return pred.cpu().numpy(), RFNE, MSE, MAE,IN,SSIM,PSNR


def eval_FNO(data_name,model_path,in_channels,lr_input_tensor,hr_target_tensor,mean,std,climate_normalization=False):
    fc_dim = 64 # or 40 for climate
    layers = [64, 64, 64, 64, 64]
    modes1 = [8, 8, 8, 8]
    modes2 = [8, 8, 8, 8]
    modes3 = [8, 8, 8, 8]
    # resol = {"decay_turb":(hr_target_tensor.shape[0],3,21,hr_target_tensor.shape[-2],hr_target_tensor.shape[-1]),"rbc":(hr_target_tensor.shape[0],3,21,shape[-2],shape[-1]),"climate_s4_sig1":(shape[0],1,21,shape[-2],shape[-1])}
    target_shape = hr_target_tensor.shape
    model = FNO3D(modes1, modes2, modes3,target_shape,width=16, fc_dim=fc_dim,layers=layers,in_dim=in_channels, out_dim=in_channels, act='gelu',mean=mean,std=std ).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # define the initial states and initial output for model
    with torch.no_grad():
        lres, hres = lr_input_tensor, hr_target_tensor
        lres, hres = lres.float().cuda(), hres.float().cuda()
        pred = model(lres)
        print(f"FNO pred shape{pred.shape}")
        if climate_normalization == True:
            pred = (pred - 278.35330263805355)/20.867389868976833
            hres = (hres - 278.35330263805355)/20.867389868976833
        pred,hres = pred.permute(0,2,1,3,4),hres.permute(0,2,1,3,4) # from B,C,T,H,W to B,T,C,H,W
        RFNE,MAE,MSE,IN = get_metric_RFNE(hres,pred)
        # permute back:
        SSIM = get_ssim(hres,pred).cpu().numpy() 
        PSNR = get_psnr(hres,pred).cpu().numpy()
    return pred.cpu().numpy(), RFNE, MSE, MAE,IN,SSIM,PSNR

def eval_NODE_wrapper(data_name,model_path,in_channels,lr_input_tensor,hr_target_tensor,mean,std,num_snapshots=20,task_dt=1,climate_normalization=False):
    checkpoint = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import argparse
    model_state = checkpoint['model_state_dict']
    config = checkpoint['config']
    args = argparse.Namespace()
    args.__dict__.update(config)
    if args.data =="Decay_turb_small": 
        image = [128,128]
        window_size = 8
    elif args.data =="rbc_small":
        image = [256,64]
        window_size = 8
    elif args.data =="burger2D":
        image = [128,128]
        window_size = 8
    elif args.data =="climate":
        image = [180,360]
        window_size = 9
    model = PASR_ODE(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=window_size, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,num_ode_layers = args.ode_layer,ode_method = args.ode_method,ode_kernel_size = args.ode_kernel,ode_padding = args.ode_padding,aug_dim_t=args.aug_dim_t)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(model_state)
    model.eval()
    with torch.no_grad():
        pred = model(lr_input_tensor.float().cuda(),n_snapshots = num_snapshots,task_dt=task_dt)
        if climate_normalization == True:
            mean = torch.from_numpy(mean).float().cuda()
            std = torch.from_numpy(std).float().cuda()
            pred = (pred - mean)/std
            hr_target_tensor = (hr_target_tensor.float().cuda() - mean)/std
        RFNE,MAE,MSE,IN = get_metric_RFNE(hr_target_tensor.float().cuda(),pred.float().cuda())
    PSNR = get_psnr(pred[:].float().cuda(),hr_target_tensor[:].float().cuda())
    SSIM = get_ssim(pred[:].float().cuda(),hr_target_tensor[:].float().cuda())
    return pred.cpu().numpy(), RFNE, MSE, MAE,IN,SSIM,PSNR


def trilinear_interpolation(lr_input_tensor,hr_target_tensor,climate_normalization=False):
    B,C,T,H,W = hr_target_tensor.shape
    trilinear_pred = F.interpolate(lr_input_tensor, size=(T,H,W), mode='trilinear', align_corners=False)
    print(f"trilinear pred shape {trilinear_pred.shape}")
    if climate_normalization == True:
        trilinear_pred = (trilinear_pred - 278.35330263805355)/20.867389868976833
        hr_target_tensor = (hr_target_tensor - 278.35330263805355)/20.867389868976833
    RFNE = torch.norm((trilinear_pred-hr_target_tensor),dim=(-1,-2))/torch.norm(hr_target_tensor,dim=(-1,-2))
    MSE = torch.mean((trilinear_pred-hr_target_tensor)**2,dim=(-1,-2))
    MAE = torch.mean(torch.abs(trilinear_pred-hr_target_tensor),dim=(-1,-2)) # result in B C T 
    IN = torch.norm((trilinear_pred-hr_target_tensor),dim=(-1,-2),p=np.inf)
    PSNR = get_psnr(hr_target_tensor.permute(0,2,1,3,4),trilinear_pred.permute(0,2,1,3,4))
    SSIM = get_ssim(hr_target_tensor.permute(0,2,1,3,4),trilinear_pred.permute(0,2,1,3,4))
    print("RFNE (Trilinear) ", RFNE.mean())
    return trilinear_pred.permute(0,2,1,3,4).numpy(),RFNE.numpy(),MSE.numpy(),MAE.numpy(),IN.numpy(),SSIM.cpu().numpy(),PSNR.cpu().numpy()    

def dump_json(key, RFNE, MAE, MSE, IN, SSIM, PSNR):
    import json
    #
    magic_batch = 5
    # Check if the results file already exists and load it, otherwise initialize an empty list
    try:
        with open("eval_v2.json", "r") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}
        print("No results file found, initializing a new one.")
    # Create a unique key based on your parameters
    # Check if the key already exists in the dictionary
    if key not in all_results:
        all_results[key] = {
        }
    # Store the results
    all_results[key]["RFNE"] = RFNE[...,1:].mean().item()
    all_results[key]["MAE"] = MAE[...,1:].mean().item()
    all_results[key]["MSE"] = MSE[...,1:].mean().item()
    all_results[key]["IN"] = IN[...,1:].mean().item()
    all_results[key]["SSIM"] = SSIM.mean().item()
    all_results[key]["PSNR"] = PSNR.mean().item()
    with open("eval_v2.json", "w") as f:
        json.dump(all_results, f, indent=4)
        f.close()
    return print("dump json done")

if __name__ == "__main__":
    # data_name_list = ["decay_turb","rbc","climate_s4_sig1"]
    # data_name_list = ["climate_s4_sig1","climate_s4_sig0","climate_s4_sig2","climate_s4_sig4"]
    data_name_list = ["burger2D"]
    # data_name_list = ["climate_s2_sig1","climate_s2_sig0","climate_s2_sig2","climate_s2_sig4"]
    # data_name_list = ["climate_s4_sig1"]
    # model_list = [ConvLSTM, FNO, NODE]
    model_info = {
        "ConvLSTM_rbc": "best_results/ConvLSTM_RBC_4458_checkpoint.pt",
        "ConvLSTM_decay_turb": "best_results/ConvLSTM_Decay_Turb_7538_checkpoint.pt",
        "ConvLSTM_climate_s4_sig1": "best_results/ConvLSTM_Climate_1574_checkpoint.pt",
        "FNO_rbc": "best_results/FNO_data_rbc_FNO_7838.pt",
        "FNO_decay_turb": "best_results/FNO_data_Decay_turb_FNO2179.pt",
        "FNO_climate_s4_sig1": "best_results/FNO_data_climate_sequence_9765.pt",
        "NODE_climate_s4_sig1_rk4":"results/PASR_ODE_small_data_climate_5533.pt",
        "NODE_rbc_euler": "results/PASR_ODE_small_data_rbc_small_1041.pt",
        "NODE_rbc_rk4": "results/PASR_ODE_small_data_rbc_small_5723.pt",
        "NODE_decay_turb_euler": "results/PASR_ODE_small_data_Decay_turb_small_8807.pt",
        "NODE_decay_turb_rk4": "results/PASR_ODE_small_data_Decay_turb_small_6724.pt",
        "NODE_rbc_euler_p0.02": "results/PASR_ODE_small_data_rbc_small_9862.pt",
        "NODE_rbc_rk4_p0.02": "results/PASR_ODE_small_data_rbc_small_8031.pt",
        "Conv_burger2D": "ConvLSTM_Burgers_4822_checkpoint.pt",
        "FNO_burger2D": "results/FNO_data_burger2D_FNO_5379.pt",
        "NODE_burger2D": "results/PASR_ODE_small_data_burger2D_5353.pt",
    }
    apdx = "euler"
    norm_Flag = False
    for data_name in data_name_list:
        for apdx in ["rk4"]:
            if "climate" in data_name:
                in_channel = 1
            else:
                in_channel = 3
            batch = 5 
            mean,std = get_normalization(data_name) # data name should be decay_turb, rbc, climate_s4_sig1
            lr_input,hr_target,lr_input_tensor, hr_target_tensor = load_test_data_IC(data_name,in_channel=in_channel, timescale_factor=4, num_snapshot=20, upscale_factor=4)
            print(f"hr_input_tensor shape {hr_target_tensor.shape}")
            lr_input2, hr_target2, lr_input_tensor2, hr_target_tensor2 = load_test_data_sequence(data_name,in_channel=in_channel, timescale_factor=4, num_snapshot=20, upscale_factor=4)
            pred_tri,RFNE_tri, MSE_tri, MAE_tri,IN_tri,SSIM_tri,PSNR_tri = trilinear_interpolation(lr_input_tensor2,hr_target_tensor2,climate_normalization=norm_Flag)
            pred_conv,RFNE_conv, MSE_conv, MAE_conv,IN_conv,SSIM_conv,PSNR_conv = eval_ConvLSTM(data_name,model_info[f"Conv_{data_name}"],in_channel,lr_input_tensor2,hr_target_tensor2,mean,std,climate_normalization=norm_Flag)
            pred_fno,RFNE_fno, MSE_fno, MAE_fno,IN_fno,SSIM_fno,PSNR_fno = eval_FNO(data_name,model_info[f"FNO_{data_name}"],in_channel,lr_input_tensor2,hr_target_tensor2,mean,std,climate_normalization=norm_Flag)
            pred_node,RFNE_node, MSE_node, MAE_node,IN_node,SSIM_node,PSNR_node = eval_NODE_wrapper(data_name,model_info[f"NODE_{data_name}"],in_channel,lr_input_tensor,hr_target_tensor,mean,std,num_snapshots=20,task_dt=1,climate_normalization=norm_Flag)
            if norm_Flag == True:
                dump_json(f"ConvLSTM_{data_name}_normalized", RFNE_conv.mean(), MAE_conv.mean(), MSE_conv.mean(), IN_conv.mean(), SSIM_conv.mean(), PSNR_conv.mean())
                dump_json(f"FNO_{data_name}_normalized", RFNE_fno.mean(), MAE_fno.mean(), MSE_fno.mean(), IN_fno.mean(), SSIM_fno.mean(), PSNR_fno.mean())
                dump_json(f"NODE_{data_name}_{apdx}_normalized", RFNE_node[batch:].mean(), MAE_node[batch:].mean(), MSE_node[batch:].mean(), IN_node[batch:].mean(), SSIM_node.mean(), PSNR_node.mean())
                dump_json(f"Trilinear_{data_name}_normalized", RFNE_tri.mean(), MAE_tri.mean(), MSE_tri.mean(), IN_tri.mean(), SSIM_tri.mean(), PSNR_tri.mean())
            else:
                dump_json(f"ConvLSTM_{data_name}", RFNE_conv, MAE_conv, MSE_conv, IN_conv, SSIM_conv.mean(), PSNR_conv.mean())
                dump_json(f"FNO_{data_name}", RFNE_fno, MAE_fno, MSE_fno, IN_fno, SSIM_fno.mean(), PSNR_fno.mean())
                dump_json(f"NODE_{data_name}_{apdx}", RFNE_node[batch:], MAE_node[batch:], MSE_node[batch:], IN_node[batch:], SSIM_node.mean(), PSNR_node.mean())
                dump_json(f"Trilinear_{data_name}", RFNE_tri, MAE_tri, MSE_tri, IN_tri, SSIM_tri.mean(), PSNR_tri.mean())
            # np.save(f"pred_{data_name}_ConvLSTM.npy",pred_conv)
            # np.save(f"pred_{data_name}_FNO.npy",pred_fno)
            # np.save(f"pred_{data_name}_{apdx}_NODE.npy",pred_node)
            # np.save(f"pred_{data_name}_trilinear.npy",pred_tri)
            # np.save(f"lr_input_{data_name}.npy",lr_input) 
            # np.save(f"hr_target_{data_name}.npy",hr_target)


    # for extrapolation
    # for data_name in data_name_list:
    #     for apdx in ["euler","rk4"]:
    #         batch = 5 
    #         # mean,std = get_normalization(data_name) # data name should be decay_turb, rbc, climate_s4_sig1
    #         in_channel =3 
    #         lr_input,hr_target,lr_input_tensor, hr_target_tensor = load_test_data_IC(data_name,in_channel=in_channel, timescale_factor=4, num_snapshot=80, upscale_factor=4)
    #         pred_node,RFNE_node, MSE_node, MAE_node,IN_node,SSIM_node,PSNR_node = eval_NODE_wrapper(data_name,model_info[f"NODE_{data_name}_{apdx}"],in_channel,lr_input_tensor,hr_target_tensor,mean,std,num_snapshots=80,task_dt=4)
    #         #np.save(f"Extrapolation_pred_{data_name}_{apdx}_NODE.npy",pred_node)
    #         np.save(f"Extrapolation_hr_target_80_{data_name}.npy",hr_target)
    #         print(f"Extrapolation_pred_{data_name}_{apdx}_NODE.npy saved")

    # for data_name in data_name_list:
    #     for apdx in ["euler","rk4"]:
    #         batch = 5 
    #         mean,std = get_normalization(data_name) # data name should be decay_turb, rbc, climate_s4_sig1
    #         in_channel =3 
    #         if data_name =="rbc":
    #             to_LR = transforms.Resize((int(256/4), int(64/4)), Image.BICUBIC, antialias=False)
    #         elif data_name =="decay_turb":
    #             to_LR = transforms.Resize((int(128/4), int(128/4)), Image.BICUBIC, antialias=False)

    #         lr_input,hr_target,lr_input_tensor, hr_target_tensor = load_test_data_IC(data_name,in_channel=in_channel, timescale_factor=4, num_snapshot=80, upscale_factor=4)
    #         has_pred = False
    #         current_input = lr_input_tensor
    #         for loopback in range(4):
    #             pred_node,RFNE_node, MSE_node, MAE_node,IN_node,SSIM_node,PSNR_node = eval_NODE_wrapper(data_name,model_info[f"NODE_{data_name}_{apdx}"],in_channel,current_input,hr_target_tensor[:,0:21],mean,std,num_snapshots=20,task_dt=1)
    #             if has_pred:
    #                 predictions = np.concatenate((predictions[:, 0:-1], pred_node), axis=1)
    #             else:
    #                 predictions = pred_node
    #                 has_pred = True
    #             # Use the last time snapshot from prediction as the next input

    #             current_input = to_LR(torch.from_numpy(pred_node[:, -1, :, :, :]))

    #         np.save(f"Extrapolation_loop_back_pred_{data_name}_{apdx}_NODE.npy",predictions)
    #         np.save(f"Extrapolation_loop_back_hr_target_80_{data_name}.npy",hr_target)
    #         print(f"Extrapolation_pred_{data_name}_{apdx}_NODE.npy saved")        

    # def forward_predictions(model, lr_input, n_steps=10):
    #     has_pred = False
    #     to_LR = transforms.Resize((int(128/4), int(128/4)), Image.BICUBIC, antialias=False)
        
    #     current_input = lr_input


            
    #         if has_pred:
    #             predictions = torch.cat((predictions[:, 0:-1], pred), dim=1)
    #         else:
    #             predictions = pred
    #             has_pred = True

    #         # Use the last time snapshot from prediction as the next input
    #         current_input = to_LR(pred[:, -1, :, :, :])
        
    #     return predictions




# def plot_energy_specturm_phyLoss(data_name,dic):
#     # pred_trilinear = 
#     realsize_truth, EK_avsphr_truth,result_dict_truth = energy_specturm(u_truth,v_truth)
#     realsize_pred, EK_avsphr_pred,result_dict_pred = energy_specturm(u_pred,v_pred)
#     realsize_pred_p, EK_avsphr_pred_p,result_dict_pred_p = energy_specturm(u_pred_p,v_pred_p)
#     fig= plt.figure(figsize=(5,5))
#     plt.title(f"Kinetic Energy Spectrum -- {data_name}")
#     plt.xlabel(r"k (wavenumber)")
#     plt.ylabel(r"TKE of the k$^{th}$ wavenumber")
#     print(realsize_truth)
#     plt.loglog(np.arange(0,realsize_truth),((EK_avsphr_truth[0:realsize_truth] )),'k',label = "truth")
#     plt.loglog(np.arange(0,realsize_pred),((EK_avsphr_pred[0:realsize_pred] )),'r',label = "pred")
#     plt.loglog(np.arange(0,realsize_pred_p),((EK_avsphr_pred_p[0:realsize_pred_p] )),'b',label = "pred (physics constraint)")
#     plt.legend()
#     fig.savefig(f"{data_name}_energy_specturm.png",dpi=300,bbox_inches='tight')
#     return print("energy specturm plot done")