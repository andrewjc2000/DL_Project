import torch
from flownet2_pytorch.models import FlowNet2
from flownet2_pytorch.utils import flow_utils
import os
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2
from dataset import Data

with torch.no_grad():
  flownet2 = FlowNet2()
  path = 'FlowNet2_checkpoint.pth.tar'
  pretrained_dict = torch.load(path)['state_dict']
  model_dict = flownet2.state_dict()
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  flownet2.load_state_dict(model_dict)
  flownet2.cuda()
  #print(flownet2)

  PATH = os.getcwd() + "/UCSD_Anomaly_Dataset.v1p2/UCSDped2/"
  of_save_dir = PATH + 'Train/flows'
  of_save_dir_flo_imgs = PATH + 'Train/flows_imgs'
  # dataset = get_dataset(dataset_name='ped2', dir=PATH, context_frame_num=1, mode='train')
  dataset = Data(dataset_name='ped2', root=PATH, train=True, sequence_length=1,
                           bboxes_extractions=True)

  WIDTH, HEIGHT = 1024, 768

  dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

  for idx, (batch, _) in tqdm(enumerate(dataloader), total=len(dataset)):
    cur_img_addr = dataset.frame_addresses[idx]
    cur_img_name = cur_img_addr.split('/')[-1]

    # path to store flows
    video_of_path = os.path.join(of_save_dir, cur_img_addr.split('/')[-2])
    if os.path.exists(video_of_path) is False:
        os.makedirs(video_of_path, exist_ok=True)

    video_of_path_imgs = os.path.join(of_save_dir_flo_imgs, cur_img_addr.split('/')[-2])
    if os.path.exists(video_of_path_imgs) is False:
        os.makedirs(video_of_path_imgs, exist_ok=True)

    # batch [bs,#frames,3,h,w]
    cur_imgs = np.transpose(batch[0].numpy(), [0, 2, 3, 1])  # [#frames,3,h,w] -> [#frames,h,w,3]

    old_size = (cur_imgs.shape[2], cur_imgs.shape[1])  # w,h

    # resize format (w',h')
    im1 = cv2.resize(cur_imgs[0], (WIDTH, HEIGHT))  # the frame before centric
    im2 = cv2.resize(cur_imgs[1], (WIDTH, HEIGHT))  # centric frame
    # [0-255]
    ims = np.array([im1, im2]).astype(np.float32)  # [2,h',w',3]
    ims = torch.from_numpy(ims).unsqueeze(0)
    ims = ims.permute(0, 4, 1, 2, 3).contiguous().cuda()  # [bs,2,H,W,img_channel] -> [bs,img_channel,2,H,W]

    pred_flow = flownet2(ims).cpu().data
    pred_flow = pred_flow[0].numpy().transpose((1, 2, 0))  # [h',w',2]
    
    # # Saves the .flo files for visualization purposes
    # flow_utils.writeFlow(os.path.join(video_of_path_imgs, cur_img_name + '.flo'), pred_flow)
    new_inputs = cv2.resize(pred_flow, old_size)  # [h,w,2]

    print(video_of_path)
    # save new raw inputs
    np.save(os.path.join(video_of_path, cur_img_name + '.npy'), new_inputs)