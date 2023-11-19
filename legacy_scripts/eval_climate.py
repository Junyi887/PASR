import h5py
import numpy as np
import torch
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau 
from torch.nn.utils import weight_norm
from src.util import *
from src.models import *
from src.data_loader_nersc import getData

FLUID_DATA_INFO = {"climate_sequence":["/pscratch/sd/j/junyi012/climate_data/pre-processed_s4_sig1",1]}

steps = 21 # 40 
effective_step = list(range(0, steps))
FLUID_DATA_INFO2 = {"decay_turb":"../Decay_Turbulence_small/*/*.h5",
                    "rbc": "../RBC_small/*/*.h5"}
def get_psnr(true, pred):
    # shape with B,T,C,H,W
    mse = torch.mean((true - pred)**2,dim= (-1,-2))
    if mse.min() <= 1e-16:
        return float('inf')
    B,T,C = mse.shape
    print(mse.shape)
    list = []
    for i in range(T):
        max_value = torch.max(true)
        psnr = 20 * torch.log10(max_value / torch.sqrt(mse[:,i,:])) # return psnr in shape B,C,T
        list.append(psnr)
    return torch.stack(list,dim =1) # return psnr in shape B,C,T

def get_ssim(true,pred):
    from torchmetrics import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure().cuda()
    B,T,C,H,W = true.shape
    true = true.reshape(B*T,C,H,W)
    pred = pred.reshape(B*T,C,H,W)
    list = []
    for i in range(C):
        ssim_score = ssim(pred[:,i:(i+1)],true[:,i:(i+1)])
        list.append(ssim_score) 
    list = torch.stack(list,dim =-1) # return ssim in shape B,C
    return list

def dispatch_FNO(data_name,in_channels,shape,num_snapshots=20):
    if data_name !="climate_sequence":
        normalizer = DataInfoLoader(FLUID_DATA_INFO2[data_name])
        mean,std = normalizer.get_mean_std()
        print('mean of hres is:',mean.tolist())
        print('stf of hres is:', std.tolist())
        fc_dim = 64
    else:
        min = np.array([196.6398630794458])
        max = np.array([318.90588255242176])
        mean =np.array([278.35330263805355])
        std = np.array([20.867389868976833])
        print('mean of hres is:',mean.tolist())
        print('stf of hres is:', std.tolist())
        fc_dim = 40
    layers = [64, 64, 64, 64, 64]
    modes1 = [8, 8, 8, 8]
    modes2 = [8, 8, 8, 8]
    modes3 = [8, 8, 8, 8]
    resol = {"decay_turb":(shape[0],3,21,shape[-2],shape[-1]),"rbc":(shape[0],3,21,shape[-2],shape[-1]),"climate_sequence":(shape[0],1,21,shape[-2],shape[-1])}
    target_shape = resol[data_name]
    model = FNO3D(modes1, modes2, modes3,target_shape,width=16, fc_dim=fc_dim,layers=layers,in_dim=in_channels, out_dim=in_channels, act='gelu',mean=mean,std=std ).cuda()
    return model

def dispatch_ConvLSTM(data_name,in_channels,num_snapshots=20):
    if data_name !="climate_sequence":
        normalizer = DataInfoLoader(FLUID_DATA_INFO2[data_name])
        mean,std = normalizer.get_mean_std()
        print('mean of hres is:',mean.tolist())
        print('stf of hres is:', std.tolist())
    else:
        min = np.array([196.6398630794458])
        max = np.array([318.90588255242176])
        mean =np.array([278.35330263805355])
        std = np.array([20.867389868976833])
        print('mean of hres is:',mean.tolist())
        print('stf of hres is:', std.tolist())
    steps = num_snapshots+1 
    effective_step = list(range(0, steps))
    model = PhySR(
        n_feats = 32,
        n_layers = [1, 2], # [n_convlstm, n_resblock]
        upscale_factor = [num_snapshots, 4], # [t_up, s_up]
        shift_mean_paras = [mean.tolist(), std.tolist()],  
        step = steps,
        in_channels = in_channels,
        effective_step = effective_step).cuda()
    return model

def eval_ConvLSTM(data_name,model_path,lr_input_tensor,hr_target_tensor):

    if data_name !="climate_sequence":
        model = dispatch_ConvLSTM(data_name,3)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        init_state = get_init_state(
            batch_size = [lr_input_tensor.shape[0]], 
            hidden_channels = [32], 
            output_size = [[lr_input_tensor.shape[-2], lr_input_tensor.shape[-1]]], # 32, 32 
            mode = 'random')
        # define the initial states and initial output for model
    else:
        model = dispatch_ConvLSTM(data_name,1)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        init_state = get_init_state(
            batch_size = [lr_input_tensor.shape[0]], 
            hidden_channels = [32], 
            output_size = [[45, 90]], 
            mode = 'random')
        
    with torch.no_grad():
        lres, hres = lr_input_tensor.permute(2,0,1,3,4), hr_target_tensor.permute(2,0,1,3,4) # (b,c,t,h,w) -> (t,b,c,h,w)
        lres, hres = lres.float().cuda(), hres.float().cuda()
        pred = model(lres,init_state)
        # pred = (pred - 278.35330263805355)/20.867389868976833
        # hres = (hres - 278.35330263805355)/20.867389868976833
        print(f"ConvLSTM pred shape{pred.shape}")
        RFNE = torch.norm((pred-hres),dim=(-1,-2))/torch.norm(hres,dim=(-1,-2))
        MSE = torch.mean((pred-hres)**2,dim=(-1,-2))
        MAE = torch.mean(torch.abs(pred-hres),dim=(-1,-2)) 
        IN = torch.norm((pred-hres),dim=(-1,-2),p=np.inf)
        # permute back:
        SSIM = get_ssim(hres.permute(1,0,2,3,4),pred.permute(1,0,2,3,4)).cpu().numpy() # from T,b,C,H,W to b,t,C,H,W
        PSNR = get_psnr(hres.permute(1,0,2,3,4),pred.permute(1,0,2,3,4)).cpu().numpy()
        RFNE = RFNE.permute(1,2,0).cpu().numpy()
        MSE = MSE.permute(1,2,0).cpu().numpy()
        MAE = MAE.permute(1,2,0).cpu().numpy()
        IN = IN.permute(1,2,0).cpu().numpy()
        print("RFNE (ConvLSTM) ", RFNE.mean())
        print(f"SSIM shape {SSIM.shape}")
        print(f"PSNR shape {PSNR.shape}")
        print(f"SSIM (ConvLSTM) {SSIM.mean():.4f} +/- {SSIM.std():.4f}")
        print(f"PSNR (ConvLSTM) {PSNR.mean():.4f} +/- {PSNR.std():.4f}")
        print("channel wise SSIM (ConvLSTM) ", SSIM.tolist())
        print("channel wise PSNR (ConvLSTM) ", PSNR.mean(axis=(0,1)).tolist())

    return pred.permute(1,0,2,3,4).cpu().numpy(), RFNE, MSE, MAE,IN,SSIM,PSNR


def eval_FNO(data_name,model_path,lr_input_tensor,hr_target_tensor):

    if data_name !="climate_sequence":
        model = dispatch_FNO(data_name,3,shape=hr_target_tensor.shape)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        # define the initial states and initial output for model
    else:
        model = dispatch_FNO(data_name,1,shape=hr_target_tensor.shape)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    with torch.no_grad():
        lres, hres = lr_input_tensor, hr_target_tensor
        lres, hres = lres.float().cuda(), hres.float().cuda()
        pred = model(lres)
        print(f"FNO pred shape{pred.shape}")
        # pred = (pred - 278.35330263805355)/20.867389868976833
        # hres = (hres - 278.35330263805355)/20.867389868976833
        RFNE = torch.norm((pred-hres),dim=(-1,-2))/torch.norm(hres,dim=(-1,-2))
        MSE = torch.mean((pred-hres)**2,dim=(-1,-2))
        MAE = torch.mean(torch.abs(pred-hres),dim=(-1,-2)) 
        IN = torch.norm((pred-hres),dim=(-1,-2),p=np.inf)
        # permute back:
        SSIM = get_ssim(hres.permute(0,2,1,3,4),pred.permute(0,2,1,3,4)).cpu().numpy() 
        PSNR = get_psnr(hres.permute(0,2,1,3,4),pred.permute(0,2,1,3,4)).cpu().numpy()
        RFNE = RFNE.cpu().numpy()
        MSE = MSE.cpu().numpy()
        MAE = MAE.cpu().numpy()
        IN = IN.cpu().numpy()
        print("RFNE (FNO) ", RFNE.mean())
        print(f"SSIM shape {SSIM.shape}")
        print(f"PSNR shape {PSNR.shape}")
        print(f"SSIM (FNO) {SSIM.mean():.3f} +/- {SSIM.std():.4f}")
        print(f"PSNR (FNO) {PSNR.mean():.3f} +/- {PSNR.std():.4f}")
        print(f"RFNE (FNO) {RFNE.mean():.3f} +/- {RFNE.std():.4f}")
        print("channel wise SSIM (FNO) ", SSIM.tolist())
        print("channel wise PSNR (FNO) ", PSNR.mean(axis=(0,1)).tolist())

    return pred.cpu().numpy(), RFNE, MSE, MAE,IN,SSIM,PSNR

def load_test_data_squence(data_name,timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4):
    _,_,,test_1,test_2 = getData(upscale_factor = 4, 
                                timescale_factor= timescale_factor,
                                batch_size = 24, 
                                data_path = data_path,
                                num_snapshots = num_snapshot
                                noise_ratio = args.noise_ratio,
                                data_name = "climate_sequence",
                                in_channels=in_channel)

def trilinear_interpolation(lr_input_tensor,hr_target_tensor):
    B,C,T,H,W = hr_target_tensor.shape
    trilinear_pred = F.interpolate(lr_input_tensor, size=(T,H,W), mode='trilinear', align_corners=False)
    print(f"trilinear pred shape {trilinear_pred.shape}")
    # trilinear_pred = (trilinear_pred - 278.35330263805355)/20.867389868976833
    # hr_target_tensor = (hr_target_tensor - 278.35330263805355)/20.867389868976833
    RFNE = torch.norm((trilinear_pred-hr_target_tensor),dim=(-1,-2))/torch.norm(hr_target_tensor,dim=(-1,-2))
    MSE = torch.mean((trilinear_pred-hr_target_tensor)**2,dim=(-1,-2))
    MAE = torch.mean(torch.abs(trilinear_pred-hr_target_tensor),dim=(-1,-2)) # result in B C T 
    IN = torch.norm((trilinear_pred-hr_target_tensor),dim=(-1,-2),p=np.inf)
    PSNR = get_psnr(hr_target_tensor.permute(0,2,1,3,4),trilinear_pred.permute(0,2,1,3,4))
    SSIM = get_ssim(hr_target_tensor.permute(0,2,1,3,4),trilinear_pred.permute(0,2,1,3,4))
    print("RFNE (Trilinear) ", RFNE.mean())
    return trilinear_pred.permute(0,2,1,3,4).numpy(),RFNE.numpy(),MSE.numpy(),MAE.numpy(),IN.numpy(),SSIM.cpu().numpy(),PSNR.cpu().numpy()

if __name__ == "__main__":
    # lr_input,hr_target,lr_input_tensor,hr_target_tensor = load_test_data_squence("decay_turb",timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4)
    # pred_tri,RFNE_tri,MSE_tri,MAE_tri,IN_tri,SSIM_tri,PSNR_tri= trilinear_interpolation(lr_input_tensor,hr_target_tensor)
    # print(f"RFNE (Trilinear): {RFNE_tri.mean():.4f} +/- {RFNE_tri.std():.4f}")
    # print(f"MAE (Trilinear): {MAE_tri.mean():.4f} +/- {MAE_tri.std():.4f}")
    # print(f"IN (Trilinear): {IN_tri.mean():.4f} +/- {IN_tri.std():.4f}")
    # print("channel wise RFNE (Trilinear) ",RFNE_tri.mean(axis=(0,-1)).tolist())
    # print("channel wise MAE (Trilinear) ",MAE_tri.mean(axis=(0,-1)).tolist())
    # print("channel wise IN (Trilinear) ",IN_tri.mean(axis=(0,-1)).tolist())
    # pred_conv,RFNE_conv,MSE_conv,MAE_conv,IN_conv,SSIM_conv,PSNR_conv = eval_ConvLSTM("decay_turb","ConvLSTM_Decay_Turb_7538_checkpoint.pt",lr_input_tensor,hr_target_tensor)
    # pred_FNO,RFNE_FNO,MSE_FNO,MAE_FNO,IN_FNO,SSIM_FNO,PSNR_FNO = eval_FNO("decay_turb","results/FNO_data_Decay_turb_FNO2179.pt",lr_input_tensor,hr_target_tensor)
    # print("hr_target shape",hr_target_tensor.shape)
    # print("pred_conv shape",pred_conv.shape)
    # print("pred_FNO shape",pred_FNO.shape)
    # pred_FNO = pred_FNO.transpose(0,2,1,3,4)
    # hr_target = hr_target.transpose(0,2,1,3,4)
    # lr_input = lr_input.transpose(0,2,1,3,4)
    # np.save("pred_FNO_DT.npy",pred_FNO)
    # np.save("pred_conv_DT.npy",pred_conv)
    # np.save("pred_tri_DT.npy",pred_tri)
    # np.save("hr_target_DT.npy",hr_target)
    # np.save("lr_input_DT.npy",lr_input)
    # print("pred_tri shape",pred_tri.shape)
    # pred_NODE = np.load("pred_NODE_DT.npy")
    # fig,ax = plt.subplots(3,5,figsize=(28,18))
    # import seaborn
    # for batch in [0,1,2,5,8]:
    #     for i in range(3):
    #         ax[i,0].imshow(hr_target[batch,i*10,0],vmin=hr_target.min(),vmax=hr_target.max(),cmap = seaborn.cm.icefire)
    #         ax[i,1].imshow(pred_tri[batch,i*10,0],vmin=hr_target.min(),vmax=hr_target.max(),cmap = seaborn.cm.icefire)
    #         ax[i,2].imshow(pred_FNO[batch,i*10,0],vmin=hr_target.min(),vmax=hr_target.max(),cmap = seaborn.cm.icefire)
    #         ax[i,3].imshow(pred_conv[batch,i*10,0],vmin=hr_target.min(),vmax=hr_target.max(),cmap = seaborn.cm.icefire)
    #         ax[i,4].imshow(pred_NODE[batch,i*10,0],vmin=hr_target.min(),vmax=hr_target.max(),cmap = seaborn.cm.icefire)
    #         ax[i,0].set_axis_off()
    #         ax[i,1].set_axis_off()
    #         ax[i,2].set_axis_off()
    #         ax[i,3].set_axis_off()
    #         ax[i,4].set_axis_off()
    #     ax[0,0].set_title("HR Target",fontsize=18,weight = 'bold')
    #     ax[0,1].set_title("Trilinear",fontsize=18,weight = 'bold')
    #     ax[0,2].set_title("FNO",fontsize=18,weight = 'bold')
    #     ax[0,3].set_title("ConvLSTM",fontsize=18,weight = 'bold')
    #     ax[0,4].set_title("Ours",fontsize=18,weight = 'bold')
    #     ax[0,0].set_ylabel("t = 0",fontsize=18,weight = 'bold')
    #     ax[1,0].set_ylabel("t = 10",fontsize=18,weight = 'bold')
    #     ax[2,0].set_ylabel("t = 20",fontsize=18,weight = 'bold')
    #     fig.savefig(f"Decay_turb_baseline_{batch}.png",bbox_inches='tight')
    
    # print(f"RFNE (ConvLSTM) {RFNE_conv.mean():.4f} +/- {RFNE_conv.std():.4f}")
    # print(f"MAE (ConvLSTM) {MAE_conv.mean():.4f} +/- {MAE_conv.std():.4f}")
    # print(f"IN (ConvLSTM) {IN_conv.mean():.4f} +/- {IN_conv.std():.4f}")
    # print("channel wise RFNE (ConvLSTM) ", RFNE_conv.mean(axis=(0,-1)).tolist())
    # print("channel wise MAE (ConvLSTM) ", MAE_conv.mean(axis=(0,-1)).tolist())
    # print("channel wise IN (ConvLSTM) ", IN_conv.mean(axis=(0,-1)).tolist())
    lr_input,hr_target,lr_input_tensor,hr_target_tensor = load_test_data_squence("rbc",timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4)
    pred_tri,RFNE_tri,MSE_tri,MAE_tri,IN_tri,SSIM_tri,PSNR_tri = trilinear_interpolation(lr_input_tensor,hr_target_tensor)
    pred_conv,RFNE_conv,MSE_conv,MAE_conv,IN_conv,SSIM_conv,PSNR_conv = eval_ConvLSTM("rbc","ConvLSTM_RBC_4458_checkpoint.pt",lr_input_tensor,hr_target_tensor)
    pred_FNO,RFNE_FNO,MSE_FNO,MAE_FNO,IN_FNO,SSIM_FNO,PSNR_FNO = eval_FNO("rbc","results/FNO_data_rbc_FNO_7838.pt",lr_input_tensor,hr_target_tensor)
    pred_FNO = pred_FNO.transpose(0,2,1,3,4)
    hr_target = hr_target.transpose(0,2,1,3,4)
    lr_input = lr_input.transpose(0,2,1,3,4)
    np.save("pred_FNO_RBC.npy",pred_FNO)
    np.save("pred_conv_RBC.npy",pred_conv)
    np.save("pred_tri_RBC.npy",pred_tri)
    np.save("hr_target_RBC.npy",hr_target)
    np.save("lr_input_RBC.npy",lr_input)
    # lr_input,hr_target,lr_input_tensor,hr_target_tensor = load_test_data_squence("climate_sequence",timescale_factor = 4,num_snapshot = 20,in_channel=1,upscale_factor=4)
    # pred_tri,RFNE_tri,MSE_tri,MAE_tri,IN_tri,SSIM_tri,PSNR_tri = trilinear_interpolation(lr_input_tensor,hr_target_tensor)
    # print(f"RFNE (Trilinear): {RFNE_tri.mean():.4f} +/- {RFNE_tri.std():.4f}")
    # print(f"MAE (Trilinear): {MAE_tri.mean():.4f} +/- {MAE_tri.std():.4f}")
    # print(f"IN (Trilinear): {IN_tri.mean():.4f} +/- {IN_tri.std():.4f}")
    # print("channel wise RFNE (Trilinear) ",RFNE_tri.mean(axis=(0,-1)).tolist())
    # print("channel wise MAE (Trilinear) ",MAE_tri.mean(axis=(0,-1)).tolist())
    # print("channel wise IN (Trilinear) ",IN_tri.mean(axis=(0,-1)).tolist())
    # pred_conv,RFNE_conv,MSE_conv,MAE_conv,IN_conv,SSIM_conv,PSNR_conv = eval_ConvLSTM("climate_sequence","ConvLSTM_Climate_1574_checkpoint.pt",lr_input_tensor,hr_target_tensor)
    # pred_FNO,RFNE_FNO,MSE_FNO,MAE_FNO,IN_FNO,SSIM_FNO,PSNR_FNO = eval_FNO("climate_sequence","results/FNO_data_climate_sequence_9765.pt",lr_input_tensor,hr_target_tensor)
    # print(f"RFNE (ConvLSTM) {RFNE_conv.mean():.4f}  +/- {RFNE_conv.std():.4f}")
    # print(f"MAE (ConvLSTM) {MAE_conv.mean():.4f} +/- {MAE_conv.std():.4f} ")
    # print(f"IN (ConvLSTM) {IN_conv.mean():.4f} +/- {IN_conv.std():.4f} ")
    # print("channel wise RFNE (ConvLSTM) ", RFNE_conv.mean(axis=(0,-1)).tolist())
    # print("channel wise MAE (ConvLSTM) ", MAE_conv.mean(axis=(0,-1)).tolist())
    # print("channel wise IN (ConvLSTM) ", IN_conv.mean(axis=(0,-1)).tolist())
    # import json
    # #
    # magic_batch = 5
    # # Check if the results file already exists and load it, otherwise initialize an empty list
    # try:
    #     with open("eval.json", "r") as f:
    #         all_results = json.load(f)
    # except (FileNotFoundError, json.JSONDecodeError):
    #     all_results = {}
    #     print("No results file found, initializing a new one.")
    # Create a unique key based on your parameters
    # key = f"Tri_Decay_turb_small_None"
    # key2 = f"ConvLSTM_Decay_turb_small_None"
    # key3 = f"FNO_Decay_turb_small_None"
    # key = f"Tri_RBC_small_None"
    # key2 = f"ConvLSTM_RBC_small_None"
    # key3 = f"FNO_RBC_small_None"
    # # key = f"Tri_Climate_None"
    # # key2 = f"ConvLSTM_Climate_None"
    # # key3 = f"FNO_Climate_None"
    # # Check if the key already exists in the dictionary
    # if key not in all_results:
    #     all_results[key] = {
    #     }
    # # if key2 not in all_results:
    # #     all_results[key2] = {
    # #     }
    # if key3 not in all_results:
    #     all_results[key3] = {
    #     }
    # # Store the results
    # all_results[key]["RFNE"] = RFNE_tri.mean().item()
    # all_results[key]["MAE"] = MAE_tri.mean().item()
    # all_results[key]["MSE"] = MSE_tri.mean().item()
    # all_results[key]["IN"] = IN_tri.mean().item()
    # all_results[key]["SSIM"] = SSIM_tri.mean().item()
    # all_results[key]["PSNR"] = PSNR_tri.mean().item()
    # all_results[key2]["RFNE"] = RFNE_conv.mean().item()
    # all_results[key2]["MAE"] = MAE_conv.mean().item()
    # all_results[key2]["MSE"] = MSE_conv.mean().item()
    # all_results[key2]["IN"] = IN_conv.mean().item()
    # all_results[key2]["SSIM"] = SSIM_conv.mean().item()
    # all_results[key2]["PSNR"] = PSNR_conv.mean().item()
    # all_results[key3]["RFNE"] = RFNE_FNO.mean().item()
    # all_results[key3]["MAE"] = MAE_FNO.mean().item()
    # all_results[key3]["MSE"] = MSE_FNO.mean().item()
    # all_results[key3]["IN"] = IN_FNO.mean().item()
    # all_results[key3]["SSIM"] = SSIM_FNO.mean().item()
    # all_results[key3]["PSNR"] = PSNR_FNO.mean().item()
    # # Save the results    
    # with open("eval.json", "w") as f:
    #     json.dump(all_results, f, indent=4)
    #     f.close()

    

