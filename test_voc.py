import argparse
import datetime
import os
import random
import sys
import logging
sys.path.append(".")
from collections import OrderedDict
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap
import matplotlib.pyplot as plt
import numpy as np
from utils import evaluate, imutils
import torch
import torch.nn.functional as F
from libs.PAR import PAR
from omegaconf import OmegaConf
from torch import multiprocessing
#from torch.utils.data import DataLoader, dataloader
from tqdm import tqdm
import joblib
from datasets import voc
from utils import evaluate
from libs.model import BMP
import imageio
from utils.camutils import (cam_to_label, propagte_aff_cam_with_bkg, refine_cams_with_par_test, refine_cam_with_bkg_v3_test)

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--work_dir", default="results", type=str, help="work_dir")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=512, type=int, help="resize the long side")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set")
parser.add_argument("--model_path", default="./wetr_iter_18000.pth", type=str, help="model_path")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")

parser.add_argument("--cls_piror", default=1, type=int, help="local_rank")

parser.add_argument("--check_name", default="debug", type=str, help="")

def get_down_size(ori_shape=(512,512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w 
    #_hw = (h + max(dilations)) * (w + max(dilations)) 
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


def validate(model, dataset, test_scales=None):

    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24])

    _preds, _gts, _msc_preds, _cams_preds, _aff_preds = [], [], [], [], []

    ##
    _par_preds, _bmp_preds  = [], []
    ##
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    with torch.no_grad(), torch.cuda.device(0):
        par.cuda(0)
        model.cuda(0)
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            #######
            # resize long side to 512
            _, _, h, w = inputs.shape
            ratio = args.resize_long / max(h,w)
            _h, _w = int(h*ratio), int(w*ratio)
            inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)
            #######
            
            segs_list = []
            inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
            cls, segs_cat, attn_pred, cam_cat = model(inputs_cat, )
            cls = cls[0].unsqueeze(0)
            cls_pred = (cls>0).type(torch.int16)
            segs = segs_cat[0].unsqueeze(0)

            _segs = (segs_cat[0,...] + segs_cat[1,...].flip(-1)) / 2
            segs_list.append(_segs)

            _, _, h, w = segs_cat.shape

            ###
            b, c, _, _ = inputs.shape

            cam_list, aff_mat = [], []
            cam_cat = F.interpolate(cam_cat, size=(h,w), mode='bilinear', align_corners=False)
            cam_cat = torch.max(cam_cat[:b,...], cam_cat[b:,...].flip(-1))
            cam_list = [F.relu(cam_cat)]
            attn_pred = attn_pred[:b]
            ###

            for s in test_scales:
                if s != 1.0:
                    _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                    inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                    _, segs_cat,  _, _cam, = model(inputs_cat, )
                    ##
                    _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                    _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                    cam_list.append(F.relu(_cam))
                    ##

                    _segs_cat = F.interpolate(segs_cat, size=(h, w), mode='bilinear', align_corners=False)
                    _segs = (_segs_cat[0,...] + _segs_cat[1,...].flip(-1)) / 2
                    segs_list.append(_segs)


            if "test" not in args.eval_set:
                ####
                cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
                cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
                cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
                resized_cam = F.interpolate(cam, size=labels.shape[1:], mode='bilinear', align_corners=False)

                ##
                #cls_label = cls_pred
                ##

                cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)

                ###PAR Propagation
                inputs_denorm = imutils.denormalize_img2(inputs.clone())
                par_label = refine_cams_with_par_test(par, inputs_denorm, cams=resized_cam, cls_labels=cls_label, cfg=cfg, img_box=[[0, inputs.shape[-2], 0, inputs.shape[-1]]])
                ##


                ###AFA Propagation
                H, W = get_down_size(ori_shape=(inputs.shape[2], inputs.shape[3]))
                infer_mask = get_mask_by_radius(h=H, w=W, radius=8) #选取特征图临域
                valid_cam_resized = F.interpolate(resized_cam, size=(H,W), mode='bilinear', align_corners=False) #特征图大小
                aff_cam = propagte_aff_cam_with_bkg(valid_cam_resized, aff=attn_pred, mask=infer_mask, cls_labels=cls_label, bkg_score=args.bkg_score) #随机游走
                aff_cam = F.interpolate(aff_cam, size=labels.shape[1:], mode="bilinear", align_corners=False) #上采样回原图大小
                aff_label = aff_cam.argmax(dim=1)

                bkg_cls = torch.ones(size=(1, 1))
                bkg_cls = bkg_cls.to(cam.device)
                _cls_labels = torch.cat((bkg_cls, cls_label), dim=1)

                ###BMP Propagation
                bg_cls = torch.sum(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0).clone()
                bg_cls = bg_cls + F.adaptive_max_pool2d(-bg_cls, (1, 1))
                bg_cls /= F.adaptive_max_pool2d(bg_cls, (1, 1)) + 1e-5
                bg_cam = torch.cat([bg_cls[:, 0, :, :].unsqueeze(1).clone(), cam], dim=1)
                bg_cam = refine_cam_with_bkg_v3_test(par, inputs_denorm, cams=bg_cam, labels=_cls_labels,img_box=[[ 0, inputs_denorm.shape[-2], 0, inputs_denorm.shape[-1]]])
                bg_cam = F.interpolate(bg_cam, size=labels.shape[1:], mode="bilinear", align_corners=False) #上采样回原图大小
                bmp_label = bg_cam.argmax(dim=1)

                _cams_preds += list(cam_label.cpu().numpy().astype(np.int16))
                _aff_preds += list(aff_label.cpu().numpy().astype(np.int16))
                _bmp_preds += list(bmp_label.cpu().numpy().astype(np.int16))
                _par_preds += list(par_label.cpu().numpy().astype(np.int16))

                #_cam_score = evaluate.scores(list(labels.cpu().numpy().astype(np.int16)), cam_label.cpu().numpy().astype(np.int16))["miou"]
                #_par_score = evaluate.scores(list(labels.cpu().numpy().astype(np.int16)), par_label.cpu().numpy().astype(np.int16))["miou"]
                #_aff_score = evaluate.scores(list(labels.cpu().numpy().astype(np.int16)), aff_label.cpu().numpy().astype(np.int16))["miou"]
                #_bmp_score = evaluate.scores(list(labels.cpu().numpy().astype(np.int16)), bmp_label.cpu().numpy().astype(np.int16))["miou"]
                ###
                #vis_aff = encode_cmap(np.squeeze(np.array(aff_label.cpu().data))).astype(np.uint8)
                #vis_cam = encode_cmap(np.squeeze(np.array(cam_label.cpu().data))).astype(np.uint8)
                #vis_par = encode_cmap(np.squeeze(np.array(par_label.cpu().data))).astype(np.uint8)
                #vis_bmp = encode_cmap(np.squeeze(np.array(bmp_label.cpu().data))).astype(np.uint8)


                #plt.imsave(args.work_dir+ '/' + args.check_name + '/afa/' + name[0] + '_%.4f.png'%(_aff_score), vis_aff)
                #plt.imsave(args.work_dir+ '/' + args.check_name + '/bmp/' + name[0] + '_%.4f.png'%(_bmp_score), vis_bmp)
                #plt.imsave(args.work_dir+ '/' + args.check_name + '/cam/' + name[0] + '_%.4f.png'%(_cam_score), vis_cam)
                #plt.imsave(args.work_dir+ '/' + args.check_name + '/par/' + name[0] + '_%.4f.png'%(_par_score), vis_par)
            


            msc_segs = torch.max(torch.stack(segs_list, dim=0), dim=0)[0].unsqueeze(0)

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            seg_preds = torch.argmax(resized_segs, dim=1)

            resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

            _preds += list(seg_preds.cpu().numpy().astype(np.int16))
            _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))

            _gts += list(labels.cpu().numpy().astype(np.int16))

            np.save(args.work_dir+ '/' + args.check_name + '/logit/' + name[0] + '.npy', {"segs":segs.cpu().numpy(), "msc_segs":msc_segs.cpu().numpy()})
            
            
    if "test" not in args.eval_set:
        return _gts, _preds, _msc_preds, _cams_preds, _aff_preds, _bmp_preds, _par_preds
    else:
        return _gts, _preds, _msc_preds, _preds, _preds


def crf_proc(config):
    print("crf post-processing...")

    txt_name = os.path.join(config.dataset.name_list_dir, args.eval_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    if "test" not in args.eval_set:
        images_path = os.path.join(config.dataset.root_dir, 'JPEGImages',)
    else:
        images_path = os.path.join(config.dataset.root_dir, 'JPEGImages_test',)
    labels_path = os.path.join(config.dataset.root_dir, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=64,  # 121, 140
        bi_rgb_std=5,   # 5, 5
        bi_w=4,         # 4, 5
    )

    def _job(i):

        name = name_list[i]
        logit_name = os.path.join(args.work_dir, args.check_name, "logit", name + ".npy")

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_segs']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.eval_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(os.path.join(args.work_dir, args.check_name, "prediction", name + ".png"), np.squeeze(pred).astype(np.uint8))
        imageio.imsave(os.path.join(args.work_dir, args.check_name, "prediction_cmap", name + ".png"), encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = int(multiprocessing.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    score = evaluate.scores(gts, preds)

    print(score)
    
    return score

def main(cfg):
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage=args.eval_set,
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    model = BMP(backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True,
                pooling=args.pooling,
                args=args)
    
    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v

    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()
    
    gts, preds, msc_preds, cams_preds, aff_preds, bmp_preds, par_preds = validate(model=model, dataset=val_dataset, test_scales=[1, 0.5, 0.75])
    #gts, preds, msc_preds, cams_preds, aff_preds = validate(model=model, dataset=val_dataset, test_scales=[1, 0.5, 0.75, 1.5])
    torch.cuda.empty_cache()

    seg_score = evaluate.scores(gts, preds)
    msc_seg_score = evaluate.scores(gts, msc_preds)

    cams_score = evaluate.scores(gts, cams_preds)
    aff_score = evaluate.scores(gts, aff_preds)

    bmp_score = evaluate.scores(gts, bmp_preds)
    par_score = evaluate.scores(gts, par_preds)

    print("segs score:")
    print(seg_score)

    print("msc segs score:")
    print(msc_seg_score)

    print("cams score:")
    print(cams_score)

    print("aff cams score:")
    print(aff_score)

    print("par cams score:")
    print(par_score)

    print("bmp cams score:")
    print(bmp_score)


    crf_score = crf_proc(config=cfg)

    ####
    save_txt_path = os.path.join(args.work_dir, "val_iou.txt")
    #score = args.check_name + "," + str(crf_score["miou"]) + "," + str(seg_score["miou"]) + "," + str(msc_seg_score["miou"]) + "," + str(cams_score["miou"]) + "," + str(aff_score["miou"])
    score = args.check_name + "," + str(crf_score["miou"]) + str(seg_score["miou"]) + "," + str(msc_seg_score["miou"]) + "," + str(cams_score["miou"]) + "," + str(par_score["miou"]) + "," + str(aff_score["miou"]) + "," + str(bmp_score["miou"])
    if not os.path.exists(save_txt_path):
        with open(save_txt_path, "a") as f:
            f.write("settings, crf_iou, seg_iou, msc_seg_iou, cam_iou, par_cam_iou, afa_cam_iou, bmp_cam_iou" + "\n")
    with open(save_txt_path, "a") as f:
        f.write(score + "\n")
    ####

    logging.info("crf score:")
    logging.info(crf_score)

    logging.info("segs score:")
    logging.info(seg_score)

    logging.info("msc segs score:")
    logging.info(msc_seg_score)

    logging.info("cams score:")
    logging.info(cams_score)

    logging.info("aff cams score:")
    logging.info(aff_score)

    logging.info("bmp cams score:")
    logging.info(bmp_score)

    return True


def setup_logger(filename='test.log'):
    ## setup logger
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s') 
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    cfg.cam.bkg_score = args.bkg_score
    cfg.cam.low_thre= args.bkg_score
    print(cfg)
    print(args)



    #args.work_dir = os.path.join(args.work_dir, args.eval_set)
    args.work_dir = os.path.join(args.work_dir, args.eval_set)

    os.makedirs(args.work_dir + "/" + args.check_name + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/" + args.check_name + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/" + args.check_name + "/prediction_cmap", exist_ok=True)
    os.makedirs(args.work_dir + "/" + args.check_name + "/afa", exist_ok=True)
    os.makedirs(args.work_dir + "/" + args.check_name + "/cam", exist_ok=True)
    os.makedirs(args.work_dir + "/" + args.check_name + "/par", exist_ok=True)
    os.makedirs(args.work_dir + "/" + args.check_name + "/bmp", exist_ok=True)
    


    setup_logger(filename=os.path.join(args.work_dir, args.check_name,'test.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)


    main(cfg=cfg)
