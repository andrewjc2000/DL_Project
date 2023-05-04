import torch
# import torchvision
import numpy as np
import argparse
import cv2
import os
from tqdm import tqdm
from scipy.ndimage import uniform_filter
from dataset import DataWithFlows
# from models import CLIP

def extract_velocity(flow, magnitude, orientation, orientations=8, motion_threshold=0.):
    orientation *= (180 / np.pi)

    cy, cx = flow.shape[:2]

    orientation_histogram = np.zeros(orientations)
    subsample = np.index_exp[cy // 2:cy:cy, cx // 2:cx:cx]
    for i in range(orientations):

        temp_ori = np.where(orientation < 360 / orientations * (i + 1),
                            orientation, -1)

        temp_ori = np.where(orientation >= 360 / orientations * i,
                            temp_ori, -1)

        cond2 = (temp_ori > -1) * (magnitude >= motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)
        temp_filt = uniform_filter(temp_mag, size=(cy, cx))

        orientation_histogram[i] = temp_filt[subsample]

    return orientation_histogram

thresh = '50'
root = os.getcwd() + "/UCSD_Anomaly_Dataset.v1p2/UCSDped2/"
all_bboxes_train = np.load('/content/drive/MyDrive/DL Proj/Train_bounding_boxes_'+ thresh + '.npy',allow_pickle=True)
all_bboxes_test = np.load('/content/drive/MyDrive/DL Proj/Test_bounding_boxes_' + thresh + '.npy',allow_pickle=True)

# if args.dataset_name == 'shanghaitech': # ShanghaiTech normalization
#     all_bboxes_train_classes = np.load(os.path.join(root, args.dataset_name, '%s_bboxes_train_classes.npy' % args.dataset_name),
#                                allow_pickle=True)
#     all_bboxes_test_classes = np.load(os.path.join(root, args.dataset_name, '%s_bboxes_test_classes.npy' % args.dataset_name),
#                               allow_pickle=True)

# train_dataset = DataWithFlows(dataset_name='ped2', root=root,
#                                       train=True, sequence_length=0, all_bboxes=all_bboxes_train, normalize=True)
test_dataset = DataWithFlows(dataset_name='ped2', root=root,
                                      train=False, sequence_length=0, all_bboxes=all_bboxes_test, normalize=True)

print("test loaded")

bins = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_velocity = []
test_velocity = []

with torch.no_grad():
    print("calculating test velocity")
    # for idx in tqdm(range(len(train_dataset)), total=len(train_dataset)):
    #     batch, batch_flows, _ = train_dataset.__getitem__(idx)
    #     # batch = batch[:, 0].to(device)
    #     batch_flows = batch_flows[:, 0].numpy()
    #     train_sample_velocities = []

    #     frame_bbox = train_dataset.all_bboxes[idx]
    #     length_y = np.ones(1)

    #     for i in range(batch_flows.shape[0]):
    #         img_flow = np.transpose(batch_flows[i], [1, 2, 0])
    #         # convert from cartesian to polar
    #         _, ang = cv2.cartToPolar(img_flow[..., 0], img_flow[..., 1])
    #         mag = np.sqrt(img_flow[..., 0] ** 2) + np.sqrt(img_flow[..., 1] ** 2)   # L1 Magnitudes
    #         velocity_cur = extract_velocity(img_flow, mag, ang, orientations=bins)
    #         train_sample_velocities.append(velocity_cur[None])

    #     train_sample_velocities = np.concatenate(train_sample_velocities, axis=0)

    #     train_velocity.append(train_sample_velocities)
    # train_velocity = np.array(train_velocity)

    # np.save('train_ucsd_velocity.npy', train_velocity)

    for idx in tqdm(range(len(test_dataset)), total=len(test_dataset)):
        batch, batch_flows, _ = test_dataset.__getitem__(idx)
        batch = batch[:, 0].to(device)
        batch_flows = batch_flows[:, 0].numpy()

        test_sample_velocities = []
        frame_bbox = test_dataset.all_bboxes[idx]

        length_y = np.ones(1)

        for i in range(batch_flows.shape[0]):
            img_flow = np.transpose(batch_flows[i], [1, 2, 0])
            # convert from cartesian to polar
            _, ang = cv2.cartToPolar(img_flow[..., 0], img_flow[..., 1])
            mag = np.sqrt(img_flow[..., 0] ** 2) + np.sqrt(img_flow[..., 1] ** 2)
            velocity_cur = extract_velocity(img_flow, mag, ang, orientations=bins)
            test_sample_velocities.append(velocity_cur[None])

        test_sample_velocities = np.concatenate(test_sample_velocities, axis=0)

        test_velocity.append(test_sample_velocities)

    test_velocity = np.array(test_velocity)
    np.save('test_velocity'+thresh+'.npy', test_velocity)
    print('test_velocity'+thresh+'.npy')