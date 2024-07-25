import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import voc
from utils.losses import DenseEnergyLoss, get_aff_loss, get_energy_loss
from libs.PAR import PAR
from utils import evaluate, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils import (get_mask_by_radius, cam_to_label, cams_to_affinity_label, ignore_img_box,
                            multi_scale_cam, propagte_aff_cam_with_bkg, refine_cams_with_bkg_v2,
                            refine_cams_with_cls_label, background_aware_propagtation, multi_scale_cam_with_aff_mat_seg)
from utils.optimizer import PolyWarmupAdamW
from libs.model import BMP

def setup_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
    ###
    #torch.backends.cudnn.enabled = False
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    ###

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

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    #time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def get_down_size(ori_shape=(512,512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w
    
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")
parser.add_argument("--high_thre", default=0.55, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.35, type=float, help="low_bkg_score")
parser.add_argument('--backend', default='nccl')


parser.add_argument("--propogation", default="BSP", type=str, help="propogation methods PAR/AFA/BSP")

parser.add_argument("--loss_type_c2s", default="BCE", type=str, help="l_seg of loss c2s")
parser.add_argument("--loss_type_s2c", default="CE", type=str, help="l_seg of loss s2c")

parser.add_argument("--sigma_c", default=0.5, type=float, help="low_bkg_score")
parser.add_argument("--sigma_s", default=0.75, type=float, help="low_bkg_score")

parser.add_argument("--lambda_1", default=0.1, type=float, help="c2s loss")
parser.add_argument("--lambda_2", default=0.1, type=float, help="s2c loss")
parser.add_argument("--lambda_3", default=0.1, type=float, help="aff loss")

parser.add_argument("--warm_iter_c2s", default=2000, type=int, help="start c2s loss")
parser.add_argument("--warm_iter_s2c", default=4000, type=int, help="start s2c loss")
parser.add_argument("--warm_iter_bsp", default=4000, type=int, help="start bsp operation")

#parser.add_argument("--cls_piror", default=1, type=float, help="use cls piror for decoder")

parser.add_argument("--check_name", default="debug", type=str, help="checkname")
parser.add_argument("--seed", default=1, type=int, help="random seed")



def validate(model=None, data_loader=None, cfg=None):

    preds, gts, cams, aff_gts = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs, attn_pred, _ = model(inputs,val=True)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)


            #_cams, _, msc_segs = multi_scale_cam_with_aff_mat_seg(model, inputs=inputs, scales=cfg.cam.scales)

            
            ###
            _cams = multi_scale_cam(model, inputs, cfg.cam.scales) #原图大小
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)
            
            H, W = get_down_size(ori_shape=(h, w))
            infer_mask = get_mask_by_radius(h=H, w=W, radius=args.radius) #选取特征图临域
            valid_cam_resized = F.interpolate(resized_cam, size=(H,W), mode='bilinear', align_corners=False) #特征图大小
            aff_cam = propagte_aff_cam_with_bkg(valid_cam_resized, aff=attn_pred, mask=infer_mask, cls_labels=cls_label, bkg_score=0.35) #随机游走
            aff_cam = F.interpolate(aff_cam, size=labels.shape[1:], mode="bilinear", align_corners=False) #上采样回原图大小
            aff_label = aff_cam.argmax(dim=1)
            ###

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            aff_gts += list(aff_label.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:,0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            #np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    aff_score = evaluate.scores(gts, aff_gts)
    model.train()
    return cls_score, seg_score, cam_score, aff_score


def train(cfg):

    num_workers = 10

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,)
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    def _init_fn(worker_id):
        np.random.seed(int(1) + worker_id)

    ###Preparing dataset
    train_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    train_sampler = DistributedSampler(train_dataset,shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              #shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              sampler=train_sampler,
                              prefetch_factor=4,
                              worker_init_fn=_init_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False,
                            worker_init_fn=_init_fn)

    device = torch.device(args.local_rank)

    ###Preparing E2E WSSS Model
    model = BMP(backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True,
                pooling=args.pooling,
                args=args)

    logging.info('\nNetwork config: \n%s'%(model))
    param_groups = model.get_param_groups()
    model.to(device)

    ###Preparing Propogation Methods
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24])
    par.to(device)
    
    mask_size = int(cfg.dataset.crop_size // 16)
    infer_size = int((cfg.dataset.crop_size * max(cfg.cam.scales)) // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    attn_mask_infer = get_mask_by_radius(h=infer_size, w=infer_size, radius=args.radius)
    if args.local_rank==0:
        writer = SummaryWriter(cfg.work_dir.tb_logger_dir)
        #dummy_input = torch.rand(1, 3, 384, 384).cuda(0)
        #writer.add_graph(wetr, dummy_input)
    
    ###Preparing Optimizer
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0, ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    #model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    # loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    bkg_cls = torch.ones(size=(cfg.train.samples_per_gpu, 1))

    ###Start Training

    for n_iter in range(cfg.train.max_iters):
        
        ###Data Sample
        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        
        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_labels = cls_labels.to(device, non_blocking=True)

        ###Model Forward
        cls, P_s, attn_pred, P_c = model(inputs, seg_detach=args.seg_detach)
        #cams, aff_mat = multi_scale_cam_with_aff_mat(wetr, inputs=inputs, scales=cfg.cam.scales)

        cams, aff_mat, msc_segs = multi_scale_cam_with_aff_mat_seg(model, inputs=inputs, scales=cfg.cam.scales, propagation=args.propogation)
        
        ###Initial CAM Labels
        valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True, cfg=cfg)

        bkg_cls = bkg_cls.to(cams.device)
        _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

        ###Propogating CAM with AFA
        if args.propogation == "AFA":
            valid_cam_resized = F.interpolate(valid_cam, size=(infer_size, infer_size), mode='bilinear', align_corners=False)
            aff_cam_l = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=cfg.cam.low_thre)
            aff_cam_l = F.interpolate(aff_cam_l, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
            aff_cam_h = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=cfg.cam.high_thre)
            aff_cam_h = F.interpolate(aff_cam_h, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)


            refined_afa_cam_l = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_l, labels=_cls_labels, img_box=img_box)
            refined_afa_label_l = refined_afa_cam_l.argmax(dim=1)
            refined_afa_cam_h = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_h, labels=_cls_labels, img_box=img_box)
            refined_afa_label_h = refined_afa_cam_h.argmax(dim=1)

            aff_cam = aff_cam_l[:,1:]
            refined_afa_cam = refined_afa_cam_l[:,1:,]
            refined_afa_label = refined_afa_label_h.clone()
            refined_afa_label[refined_afa_label_h == 0] = cfg.dataset.ignore_index
            refined_afa_label[(refined_afa_label_h + refined_afa_label_l) == 0] = 0
            refined_afa_label = ignore_img_box(refined_afa_label, img_box=img_box, ignore_index=cfg.dataset.ignore_index) #黑色区域不考虑

        ###Propogating CAM with PAR
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=cams, cls_labels=cls_labels, cfg=cfg, img_box=img_box)

        ##Propogating CAM with Our Background Aware-Propagation
        #bg_cam = refine_seg_with_bkg_v3(par, inputs_denorm, cams=bg_cam, labels=_cls_labels, img_box=img_box)
        bg_cam = background_aware_propagtation(par, inputs_denorm, cams=cams, segs=msc_segs, labels=_cls_labels, img_box=img_box)
        max_bg_cam, _ = torch.max(bg_cam, dim=1)
        bsp_label = bg_cam.argmax(dim=1)
        bsp_label[max_bg_cam < args.sigma_c] = cfg.dataset.ignore_index
        bsp_label = bsp_label.type(torch.long)

        ##Generating Y_c
        if n_iter < args.warm_iter_bsp or args.propogation == "PAR": ##Warm Up Using PAR
            Y_c = refined_pseudo_label.detach().clone()
        elif args.propogation == "BSP" :
            Y_c = bsp_label.detach().clone()
        elif args.propogation == "AFA":
            Y_c = refined_afa_label.detach().clone()

        ####Classification Loss
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)

        ###Affinity Loss
        aff_label = cams_to_affinity_label(refined_pseudo_label, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        aff_loss, _, _ = get_aff_loss(attn_pred, aff_label)

        ###Interiactional Supervision Loss----Loss C2S
        if args.loss_type_c2s == "BCE":
            Y_c_mr = Y_c.view(-1).long()
            P_s_mr = P_s.permute(0, 2, 3, 1).contiguous().view(Y_c_mr.shape[0], -1)
            c2s_loss = F.multilabel_soft_margin_loss(P_s_mr[Y_c_mr!=cfg.dataset.ignore_index, :], F.one_hot(Y_c_mr[Y_c_mr!=cfg.dataset.ignore_index], P_s_mr.shape[1]))
        else:
            c2s_loss = F.cross_entropy(P_s, Y_c.type(torch.long).detach(), ignore_index=cfg.dataset.ignore_index)
        
        ##Generating Y_s
        max_P_s, Y_s = torch.max(P_s.detach().clone(), dim=1)
        Y_s = ignore_img_box(Y_s, img_box=img_box, ignore_index=cfg.dataset.ignore_index) #Ignore Non-Image Region
        Y_s[max_P_s < args.sigma_s] = cfg.dataset.ignore_index # Mask Uncertain Region
        Y_s[Y_s == 0] = cfg.dataset.ignore_index # Mask Background Region

        ###Interiactional Supervision Loss----Loss S2C
        if args.loss_type_s2c == "BCE":
            Y_s_mr = Y_s.view(-1).long() - 1
            P_c_mr = P_c.permute(0, 2, 3, 1).contiguous().view(Y_s.shape[0], -1)
            s2c_loss = F.multilabel_soft_margin_loss(P_c_mr[Y_s_mr!= (cfg.dataset.ignore_index - 1), :], F.one_hot(Y_s_mr[Y_s_mr!= (cfg.dataset.ignore_index - 1)], P_c_mr.shape[1]))
        else:
            s2c_loss = F.cross_entropy(P_c, Y_s.type(torch.long) - 1, ignore_index= (cfg.dataset.ignore_index - 1) )

        if n_iter <= args.warm_iter_c2s:
            loss = 1.0 * cls_loss + 0.0 * c2s_loss + 0.0 * aff_loss# + 0.0 * reg_loss
        elif n_iter <= args.warm_iter_s2c:
            loss = 1.0 * cls_loss + args.lambda_1 * c2s_loss + args.lambda_3 * aff_loss #+ 0.1 * reg_loss
        else:
            loss = 1.0 * cls_loss + args.lambda_1 * c2s_loss + args.lambda_2 * s2c_loss + args.lambda_3 * aff_loss

        avg_meter.add({'cls_loss': cls_loss.item(), 'c2s_loss': c2s_loss.item(), 's2c_loss':s2c_loss.item(), 'aff_loss': aff_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (n_iter+1) % cfg.train.log_iters == 0:
        #if (n_iter+1) % 1 == 0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(P_s,dim=1,).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)
            PAR_gts = refined_pseudo_label.cpu().numpy().astype(np.int16)
            BSP_gts = bsp_label.cpu().numpy().astype(np.int16)
            
            Y_c_gts = Y_c.cpu().numpy().astype(np.int16)
            Y_s_gts = Y_s.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size

            grid_imgs, grid_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=valid_cam)


            grid_seg, _ = cam_to_label(P_s[:, 1:, :, :].detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True, cfg=cfg)
            grid_msc_seg, _ = cam_to_label(msc_segs[:, 1:, :, :].detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True, cfg=cfg)
            _, grid_seg = imutils.tensorboard_image(imgs=inputs.clone(), cam=grid_seg)
            _, grid_msc_seg = imutils.tensorboard_image(imgs=inputs.clone(), cam=grid_msc_seg)

            grid_labels = imutils.tensorboard_label(labels=gts)
            grid_preds = imutils.tensorboard_label(labels=preds)

            grid_PAR_gt = imutils.tensorboard_label(labels=PAR_gts)
            grid_BSP_gt = imutils.tensorboard_label(labels=BSP_gts)

            grid_Y_s_gt = imutils.tensorboard_label(labels=Y_s_gts)
            grid_Y_c_gt = imutils.tensorboard_label(labels=Y_c_gts)

            if args.propogation == "AFA":
                AFA_gts = refined_afa_label.cpu().numpy().astype(np.int16)
                grid_AFA_gt = imutils.tensorboard_label(labels=AFA_gts)


            #grid_org_aff_label= imutils.tensorboard_label(labels=org_label.cpu().numpy().astype(np.int16))

            if args.local_rank==0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, aff_loss: %.4f, pseudo_seg_loss %.4f, cam_loss %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('aff_loss'), avg_meter.pop('c2s_loss'), avg_meter.pop('s2c_loss'), seg_mAcc))

                ###Seg Branch
                writer.add_image("seg/images", grid_imgs, global_step=n_iter)
                writer.add_image("seg/preds", grid_preds, global_step=n_iter)
                writer.add_image("seg/P_s", grid_seg, global_step=n_iter)
                writer.add_image("seg/Multi_P_s", grid_msc_seg, global_step=n_iter)
                writer.add_image("seg/Y_s", grid_Y_s_gt, global_step=n_iter)

                ###Cls Branch
                writer.add_image("cls/SEG INIT Label", grid_labels, global_step=n_iter)
                writer.add_image("cls/SEG PAR Label", grid_PAR_gt, global_step=n_iter)
                writer.add_image("cls/SEG BSP Label", grid_BSP_gt, global_step=n_iter)

                if args.propogation == "AFA":
                    writer.add_image("cls/SEG AFA Label", grid_AFA_gt, global_step=n_iter)

                writer.add_image("cls/P_c", grid_cam, global_step=n_iter)
                writer.add_image("cls/Y_c", grid_Y_c_gt, global_step=n_iter)
            
                ###Loss Term
                writer.add_scalar('train/c2s_loss', c2s_loss.item(), global_step=n_iter)
                writer.add_scalar('train/s2c_loss', s2c_loss.item(), global_step=n_iter)
                writer.add_scalar('train/cls_loss', cls_loss.item(), global_step=n_iter)
                writer.add_scalar('train/aff_loss', aff_loss.item(), global_step=n_iter)
                
                
        
        if (n_iter+1) % cfg.train.eval_iters == 0:
        #if (n_iter+1) % 1 == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "wetr_iter_%d.pth"%(n_iter+1))
            if args.local_rank==0:
                logging.info('Validating...')
                torch.save(model.state_dict(), ckpt_name)
            cls_score, seg_score, cam_score, aff_score = validate(model=model, data_loader=val_loader, cfg=cfg)
            if args.local_rank==0:
                logging.info("val cls score: %.6f"%(cls_score))
                logging.info("cams score:")
                logging.info(cam_score)
                logging.info("aff cams score:")
                logging.info(aff_score)
                logging.info("segs score:")
                logging.info(seg_score)

    return cam_score, aff_score, seg_score


if __name__ == "__main__":

    ## fix random seed

    args = parser.parse_args()

    ## fix random seed
    setup_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    cfg.cam.high_thre = args.high_thre
    cfg.cam.low_thre = args.low_thre

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    #timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    '''
    ####
    settings = str(args.rate_seg)
    settings += "_" + str(args.rate_aff)
    settings += "_" + str(args.rate_cam)
    settings += "_" + str(args.cr_iters)
    settings += "_" + str(args.rate_uncertain)
    settings += "_" + str(args.uncertain_iters)
    settings += "_" + str(args.bg_thresh)
    ####
    '''

    settings = args.check_name
    timestamp =  settings #+ "_" + timestamp


    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)
    

    if args.local_rank == 0:
        setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)
    
    ##save performance
    cam_score, aff_score, seg_score = train(cfg=cfg)

    save_txt_path = os.path.join(args.work_dir, "val_iou.txt")
    if not os.path.exists(save_txt_path):
        with open(save_txt_path, "a") as f:
            f.write("settings, cam_iou, aff_iou, seg_iou" + "\n")
    score = settings + "," + str(cam_score["miou"]) + "," + str(aff_score["miou"]) + "," + str(seg_score["miou"])
    with open(save_txt_path, "a") as f:
        f.write(score + "\n")
