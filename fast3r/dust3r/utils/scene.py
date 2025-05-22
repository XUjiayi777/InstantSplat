import torch
import numpy as np
from fast3r.dust3r.utils.geometry import xy_grid
from fast3r.dust3r.utils.device  import to_numpy
import cv2

def calibrate_camera_pnpransac(pointclouds, img_points, masks, intrinsics):
    """
    Input:
        pointclouds: (bs, N, 3) 
        img_points: (bs, N, 2) 
    Return:
        rotations: (bs, 3, 3) 
        translations: (bs, 3, 1) 
        c2ws: (bs, 4, 4) 
    """
    bs = pointclouds.shape[0]
    camera_matrix = intrinsics.cpu().numpy()  # (bs, 3, 3)
    dist_coeffs = np.zeros((5, 1))

    rotations = []
    translations = []
    
    for i in range(bs):
        obj_points = pointclouds[i][masks[i]].cpu().numpy()
        img_pts = img_points[i][[masks[i]]].cpu().numpy()

        success, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, img_pts, camera_matrix[i], dist_coeffs)

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            rotations.append(torch.tensor(rotation_matrix, dtype=torch.float32))
            translations.append(torch.tensor(tvec, dtype=torch.float32))
        else:
            rotations.append(torch.eye(3))
            translations.append(torch.ones(3, 1))

    rotations = torch.stack(rotations).to(pointclouds.device)
    translations = torch.stack(translations).to(pointclouds.device)
    w2cs = torch.eye(4).repeat(bs, 1, 1).to(pointclouds.device)
    w2cs[:, :3, :3] = rotations
    w2cs[:, :3, 3:] = translations
    return torch.linalg.inv(w2cs)

def estimate_focal_knowing_depth(pts3d, valid_mask, min_focal=0., max_focal=np.inf):
    """ Reprojection method, for when the absolute depth is known:
        1) estimate the camera focal using a robust estimator
        2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape # valid_mask: [1, H, W], bs = 1
    assert THREE == 3

    # centered pixel grid
    pp = torch.tensor([[W/2, H/2]], dtype=torch.float32, device=pts3d.device)
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)
    valid_mask = valid_mask.flatten(1, 2)  # (B, HW, 1)
    pixels = pixels[valid_mask].unsqueeze(0)  # (1, N, 2)
    pts3d = pts3d[valid_mask].unsqueeze(0)  # (1, N, 3)

    # init focal with l2 closed form
    # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
    xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

    dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
    dot_xy_xy = xy_over_z.square().sum(dim=-1)

    focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

    # iterative re-weighted least-squares
    for iter in range(10):
        # re-weighting by inverse of distance
        dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
        # print(dis.nanmean(-1))
        w = dis.clip(min=1e-8).reciprocal()
        # update the scaling with the new weights
        focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)
    # print(focal)
    return focal

    
def scene_information(output,imgs, focals, min_conf_thr=3):
    for x, img in zip(output['views'], imgs[:]):
        x['img'] = img['img'].permute(0,2,3,1)
        
    with torch.no_grad():
        _, h, w = output['views'][0]['img'].shape[0:3]# [1, H, W, 3]
        rgbimg = [x['img'][0] for x in output['views']]
        
        for i in range(len(rgbimg)):
            rgbimg[i] = (rgbimg[i] + 1) / 2 # rgb value [0,1]
        pts3d = [torch.squeeze(x['pts3d_local_aligned_to_global'], dim=0)for x in output['preds']] 
        depthmaps=[torch.squeeze(x['pts3d_local'], dim=0)[:,:,-1]for x in output['preds']] 
        conf = torch.stack([x['conf'][0] for x in output['preds']], 0) # [N, H, W]
        conf_sorted = conf.reshape(-1).sort()[0]
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
        msk = conf >= conf_thres 
              
        intrinsics = torch.eye(3,)
        intrinsics[0, 0] = focals
        intrinsics[1, 1] = focals
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        intrinsics = intrinsics.cuda()

        focals = torch.Tensor([focals]).reshape(1,).repeat(len(rgbimg)) 
        print(' rgbimg', np.array(rgbimg).shape)
        
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda() # [H, W, 2] corrdinates of each pixels
        
        c2ws = []
        for (pr_pt, valid) in zip(pts3d, msk):
            c2ws_i = calibrate_camera_pnpransac(pr_pt.cuda().flatten(0,1)[None], pixel_coords.flatten(0,1)[None], valid.cuda().flatten(0,1)[None], intrinsics[None]) 
            # 1,4,4
            c2ws.append(c2ws_i[0])

        cams2world = torch.stack(c2ws, dim=0).cpu() # [N, 4, 4]
        focals = to_numpy(focals)
        pts3d = to_numpy(pts3d)
        msk = to_numpy(msk)  
        depthmaps= to_numpy(depthmaps)  
        
        return rgbimg, pts3d, conf, intrinsics, depthmaps,focals