import torch
import torch.nn.functional as F
from .imutils import denormalize_img, encode_cmap
from .dcrf import crf_inference_label
import numpy as np
import imageio



def refine_loc_with_bkg(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None, img_box=None, down_scale=2):

    b,_,h,w = images.shape
    _images = F.interpolate(images, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)

    #cls_labels = torch.ones(size=(b,2))
    #cls_labels = cls_labels.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * cfg.dataset.ignore_index
    refined_label = refined_label.to(cams.device)
    
    for idx, coord in enumerate(img_box):

        valid_key = torch.nonzero(cls_labels[idx,...])[:,0]
        valid_cams = cams[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        _refined_label, _refined_cams = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams, valid_key=valid_key, orig_size=(h, w))
        refined_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label[0, coord[0]:coord[1], coord[2]:coord[3]]

    return refined_label


def get_semantic_aff_loss(inputs, targets):

    pos_count = 0
    neg_count = 0
    pos_loss = torch.zeros(1).cuda()
    neg_loss = torch.zeros(1).cuda()
    for i in range(0, 20):
        pos_label = (targets == i).type(torch.int16)
        if pos_label.sum() == 0:
            continue
        pos_count = pos_label.sum() + 1
        neg_label = (targets != i).type(torch.int16)
        neg_count = neg_label.sum() + 1
        #inputs = torch.sigmoid(input=inputs)
        pos_loss += torch.sum(pos_label * (1 - inputs[:, i, :, :])) / pos_count
        neg_loss += torch.sum(neg_label * (inputs[:, i, :, :])) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label, refined_cams

def refine_seg_with_bkg(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None, img_box=None, down_scale=2, thresh=-100):

    b,_,h,w = images.shape
    _images = F.interpolate(images, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * cfg.dataset.ignore_index
    refined_label = refined_label.to(cams.device)

    ###
    cams = torch.sigmoid(cams)
    ###
    
    for idx, coord in enumerate(img_box):

        valid_key = torch.nonzero(cls_labels[idx,...])[:,0]
        valid_cams = cams[idx, valid_key, ...].unsqueeze(0)#.softmax(dim=1)
        _refined_label, _refined_cams = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams, valid_key=valid_key, orig_size=(h, w))
        #
        _refined_cams_max, _ = torch.max(_refined_cams, dim=1)
        #print(np.unique(_refined_cams_max.cpu().data))
        _refined_label[_refined_cams_max < thresh] = cfg.dataset.ignore_index
        #
        refined_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label[0, coord[0]:coord[1], coord[2]:coord[3]]

    #refined_label[_refined_cams < 0.3] = cfg.dataset.ignore_index

    return refined_label


def refine_seg_with_bkg_v2(ref_mod=None, images=None, labels=None, cams=None, img_box=None):
    
    refined_cams = torch.zeros_like(cams)
    b = images.shape[0]

    #bg_label = torch.ones(size=(b, 1),).to(labels.device)
    cls_label = labels

    ###
    #cams = torch.sigmoid(cams)
    cams = torch.softmax(cams, dim=1)
    ###

    for idx, coord in enumerate(img_box):

        _images = images[[idx], :, coord[0]:coord[1], coord[2]:coord[3]]

        _, _, h, w = _images.shape
        _images_ = F.interpolate(_images, size=[h//2, w//2], mode="bilinear", align_corners=False)

        valid_key = torch.nonzero(cls_label[idx,...])[:,0]
        valid_cams = cams[[idx], :, coord[0]:coord[1], coord[2]:coord[3]][:, valid_key,...]

        _refined_cams = ref_mod(_images_, valid_cams)
        _refined_cams = F.interpolate(_refined_cams, size=_images.shape[2:], mode="bilinear", align_corners=False)

        refined_cams[idx, valid_key, coord[0]:coord[1], coord[2]:coord[3]] = _refined_cams[0,...]

    return refined_cams




def refine_seg_with_bkg_v3(ref_mod=None, images=None, labels=None, cams=None, img_box=None):
    
    refined_cams = torch.zeros_like(cams)
    b = images.shape[0]

    #bg_label = torch.ones(size=(b, 1),).to(labels.device)
    cls_label = labels

    ###
    #cams = torch.sigmoid(cams)
    #cams = torch.softmax(cams, dim=1)
    ###

    for idx, coord in enumerate(img_box):

        _images = images[[idx], :, coord[0]:coord[1], coord[2]:coord[3]]

        _, _, h, w = _images.shape
        _images_ = F.interpolate(_images, size=[h//2, w//2], mode="bilinear", align_corners=False)

        valid_key = torch.nonzero(cls_label[idx,...])[:,0]
        valid_cams = cams[[idx], :, coord[0]:coord[1], coord[2]:coord[3]][:, valid_key,...]

        _refined_cams = ref_mod(_images_, valid_cams)
        _refined_cams = F.interpolate(_refined_cams, size=_images.shape[2:], mode="bilinear", align_corners=False)

        refined_cams[idx, valid_key, coord[0]:coord[1], coord[2]:coord[3]] = _refined_cams[0,...]

    return refined_cams





def propagte_aff_seg(cams_with_bkg, aff=None, mask=None, cls_labels=None):


    cams_rw = torch.zeros_like(cams_with_bkg)

    ##########

    b, c, h, w = cams_with_bkg.shape
    n_pow = 2
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            aff[i, mask==0] = 0

    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-1) ## avoid nan

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _cams = cams_with_bkg[i].reshape(c, -1)
        valid_key = torch.nonzero(cls_labels[i,...])[:,0]
        _cams = _cams[valid_key,...]
        _cams = F.softmax(_cams, dim=0)
        _aff = aff[i]
        _cams_rw = torch.matmul(_cams, _aff)
        cams_rw[i, valid_key,:] = _cams_rw.reshape(-1, cams_rw.shape[2], cams_rw.shape[3])

    return cams_rw


def propagte_cls_aff_cam_with_bkg(cams, aff=None, mask=None, cls_labels=None, bkg_score=None):

    b,_,h,w = cams.shape

    bkg = torch.ones(size=(b,1,h,w))*bkg_score
    bkg = bkg.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    cams_with_bkg = torch.cat((bkg, cams), dim=1)

    cams_rw = torch.zeros_like(cams_with_bkg)

    ##########

    b, c, h, w = cams_with_bkg.shape
    n_pow = 2
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            aff[i, mask==0] = 0

    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-1) ## avoid nan

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _cams = cams_with_bkg[i].reshape(c, -1)
        valid_key = torch.nonzero(cls_labels[i,...])[:,0]
        _cams = _cams[valid_key,...]
        _cams = F.softmax(_cams, dim=0)
        _aff = aff[i]
        _cams_rw = torch.matmul(_cams, _aff)
        cams_rw[i, valid_key,:] = _cams_rw.reshape(-1, cams_rw.shape[2], cams_rw.shape[3])

    return cams_rw







def propagate_cams(ref_mod=None, images=None, labels=None, cams=None, img_box=None):
    
    refined_cams = torch.zeros_like(cams)
    b = images.shape[0]

    cls_label = torch.ones(size=(b, cams.shape[1]),).to(labels.device)
    #cls_label = labels

    for idx, coord in enumerate(img_box):

        _images = images[[idx], :, coord[0]:coord[1], coord[2]:coord[3]]

        _, _, h, w = _images.shape
        _images_ = F.interpolate(_images, size=[h//2, w//2], mode="bilinear", align_corners=False)

        valid_key = torch.nonzero(cls_label[idx,...])[:,0]
        valid_cams = cams[[idx], :, coord[0]:coord[1], coord[2]:coord[3]][:, valid_key,...]

        _refined_cams = ref_mod(_images_, valid_cams)
        _refined_cams = F.interpolate(_refined_cams, size=_images.shape[2:], mode="bilinear", align_corners=False)

        refined_cams[idx, valid_key, coord[0]:coord[1], coord[2]:coord[3]] = _refined_cams[0,...]

    return refined_cams
