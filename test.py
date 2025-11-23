import argparse
import time
from torch.utils.data import DataLoader
import torch
from model.model import vgg19
from datasets.dataset import Crowd
import numpy as np
import os
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--downsample-ratio', default=8, type=int,
                        help='the downsample ratio of the model')
    parser.add_argument('--data-dir', default='',
                        help='the directory of the data')
    parser.add_argument('--model-path', default=r"",
                        help='the path to the model')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='the number of samples in a batch')
    parser.add_argument('--device', default='0',
                        help="assign device")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arg()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    dataset = Crowd(args.data_dir, 224, args.downsample_ratio, method='val')
    dataloader = DataLoader(dataset, 1, shuffle=False, pin_memory=False)
    model = vgg19(25)
    device = torch.device('cuda')
    model.to(device)
    checkpoint = torch.load(args.model_path, device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    file = open('result.txt', 'w')
    step = 0
    epoch_res = []
    epoch_res_bud = []
    epoch_res_bloom = []
    epoch_res_faded = []
    epoch_start = time.time()
    count_of_tree = dict()
    list = []
    for inputs, gt_counts, name in dataloader:
        if name[0][0:1] == "D":
            tree_name = name[0][0:-2]
            if tree_name.endswith("_"):
                tree_name = tree_name[:-1]
        else:
            tree_name = name[0].split('_')[0]

        if tree_name not in list:
            list.append(tree_name)
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            pred_bud = outputs[: ,0:1, :, :]  # 形状为[1, 28, 28]
            pred_bloom = outputs[: ,1:2, :, :]  # 形状为[1, 28, 28]
            pred_faded = outputs[: ,2:3, :, :]  # 形状为[1, 28, 28]

            res = sum(gt_counts) - torch.sum(outputs).item()
            res_bud = gt_counts[0] - torch.sum(pred_bud).item()
            res_bloom = gt_counts[1] - torch.sum(pred_bloom).item()
            res_faded = gt_counts[2] - torch.sum(pred_faded).item()

            epoch_res.append(res)
            epoch_res_bud.append(res_bud)
            epoch_res_bloom.append(res_bloom)
            epoch_res_faded.append(res_faded)
            
    epoch_res = np.array(epoch_res)
    epoch_res_bud = np.array(epoch_res_bud)
    epoch_res_bloom = np.array(epoch_res_bloom)
    epoch_res_faded = np.array(epoch_res_faded)

    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    mse_bud = np.sqrt(np.mean(np.square(epoch_res_bud)))
    mae_bud = np.mean(np.abs(epoch_res_bud))
    mse_bloom = np.sqrt(np.mean(np.square(epoch_res_bloom)))
    mae_bloom = np.mean(np.abs(epoch_res_bloom))
    mse_faded = np.sqrt(np.mean(np.square(epoch_res_faded)))
    mae_faded = np.mean(np.abs(epoch_res_faded))

    print(
        'MAE: {:.2f},MAE_bud: {:.2f},MAE_bloom: {:.2f}, MAE_faded: {:.2f} ,MSE: {:.2f}, MSE_bud: {:.2f},MSE_bloom: {:.2f}, MSE_faded: {:.2f} Cost {:.1f} sec'.format(
            mae, mae_bud, mae_bloom, mae_faded, mse, mse_bud, mse_bloom, mse_faded, (time.time() - epoch_start)))
