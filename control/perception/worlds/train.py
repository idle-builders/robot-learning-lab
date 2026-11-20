"""Bounded gsplat reconstruction using upstream DefaultStrategy and rasterization.

The parameterization, Adam rates and L1/SSIM weighting follow gsplat v1.5.3
examples/simple_trainer.py (Apache-2.0). SSIM uses pytorch-msssim rather than
upstream's fused kernel. Camera poses remain fixed to COLMAP.
Input: undistorted JPEGs and scene.npz from package.py. No invented metric scale.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from gsplat import export_splats
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--refinement-mode', choices=['bounded','continuous-prune','upstream'], default='bounded')
    ap.add_argument('--max-gaussians', type=int, default=650000)
    args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    # cuDNN TF32 on the NHWC-derived SSIM tensors corrupts the moment
    # differences: our retained layout probe measured a 2.7e8-fold gradient
    # amplification on identical pixels. IEEE float32 agrees with float64.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(3915); np.random.seed(3915)
    torch.set_num_threads(8)
    started = time.monotonic()
    scene = np.load(args.data/'scene.npz')
    metadata_path = args.data/'scene-metadata.json'
    scene_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    points, rgbs = scene['points'], scene['colors']/255.
    names = scene['names'].tolist()
    c2ws = torch.tensor(scene['c2ws'], dtype=torch.float32, device='cuda')
    Ks = torch.tensor(scene['Ks'], dtype=torch.float32, device='cuda')
    train_ids = np.flatnonzero(scene['train']).tolist()
    heldout_ids = np.flatnonzero(~scene['train']).tolist()
    imgs = torch.stack([torch.from_numpy(np.array(Image.open(args.data/'images'/n))).float()/255 for n in names]).to('cuda')
    h, w = imgs.shape[1:3]
    scene_scale = float(scene['scene_scale'])
    dist = cKDTree(points).query(points, k=4)[0][:, 1:]
    scales = np.log(np.sqrt((dist**2).mean(-1)).clip(1e-5))[:, None].repeat(3, axis=1)
    n = len(points)
    sh = np.zeros((n, 16, 3), np.float32); sh[:, 0] = (rgbs-.5)/.28209479177387814
    init = {'means': points, 'scales': scales, 'quats': np.random.rand(n, 4),
            'opacities': np.full(n, math.log(.1/.9)), 'sh0': sh[:, :1], 'shN': sh[:, 1:]}
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device='cuda')) for k, v in init.items()})
    rates = {'means': 1.6e-4*scene_scale, 'scales': .005, 'quats': .001, 'opacities': .05, 'sh0': .0025, 'shN': .0025/20}
    opts = {k: torch.optim.Adam([{'params': v, 'lr': rates[k], 'name': k}], eps=1e-15) for k, v in splats.items()}
    refine_stop = 15000 if args.refinement_mode == 'upstream' else (args.steps if args.refinement_mode == 'continuous-prune' else 8000)
    strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=refine_stop, reset_every=3000,
                               refine_every=100, verbose=False)
    strategy.check_sanity(splats, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)

    def render(c2w, K, width=w, height=h, degree=3, depth=False):
        return rasterization(means=splats['means'], quats=splats['quats'], scales=splats['scales'].exp(),
            opacities=splats['opacities'].sigmoid(), colors=torch.cat([splats['sh0'], splats['shN']], 1),
            viewmats=torch.linalg.inv(c2w[None]), Ks=K[None], width=width, height=height,
            sh_degree=degree, packed=False, near_plane=.005*scene_scale,
            far_plane=1000*scene_scale, render_mode='RGB+ED' if depth else 'RGB')

    metrics = []
    eval_ids = heldout_ids + train_ids[::max(1, len(train_ids)//12)][:12]

    @torch.no_grad()
    def evaluate(step, save_images):
        rows = []
        for i in eval_ids:
            color, alpha, _ = render(c2ws[i], Ks[i], degree=min(step//1000, 3), depth=True)
            rgb = color[..., :3].clamp(0, 1)
            mse = F.mse_loss(rgb[0], imgs[i]).item()
            row = {'name': names[i], 'split': 'train' if i in train_ids else 'heldout', 'step': step,
                   'psnr_db': -10*math.log10(max(mse, 1e-12)),
                   'ssim': ssim(rgb.permute(0,3,1,2), imgs[i:i+1].permute(0,3,1,2), data_range=1).item(),
                   'alpha_mean': alpha.mean().item(), 'alpha_below_0_5_fraction': (alpha<.5).float().mean().item()}
            rows.append(row)
            if save_images:
                source = (imgs[i].cpu().numpy()*255).astype('uint8')
                rendered = (rgb[0].cpu().numpy()*255).astype('uint8')
                canvas = Image.new('RGB', (w*2, h+35), '#171717')
                canvas.paste(Image.fromarray(source),(0,35)); canvas.paste(Image.fromarray(rendered),(w,35))
                d = ImageDraw.Draw(canvas); d.text((10,10),row['split']+' source image',fill='white')
                d.text((w+10,10),f'Gaussian render | {row["psnr_db"]:.2f} dB | SSIM {row["ssim"]:.3f}',fill='white')
                canvas.save(out/('compare-'+Path(names[i]).stem+'.jpg'), quality=90)
        metrics.extend(rows)
        print(json.dumps({'eval_step': step, 'heldout_psnr_db': float(np.mean([r['psnr_db'] for r in rows if r['split']=='heldout'])), 'gaussians': len(splats['means'])}), flush=True)

    evaluate(0, False)
    with (out/'training.jsonl').open('w') as log:
        for step in range(args.steps):
            i = int(np.random.choice(train_ids))
            factor = 2 if step < 1000 else 1
            K = Ks[i].clone(); K[:2] /= factor
            target = imgs[i:i+1]
            if factor == 2:
                target = F.interpolate(target.permute(0,3,1,2), size=(h//2,w//2), mode='area').permute(0,2,3,1)
            rgb, _, info = render(c2ws[i], K, w//factor, h//factor, min(step//1000, 3))
            strategy.step_pre_backward(splats, opts, state, step, info)
            l1 = F.l1_loss(rgb, target)
            similarity = ssim(rgb.permute(0,3,1,2), target.permute(0,3,1,2), data_range=1)
            loss = .8*l1 + .2*(1-similarity)
            loss.backward()
            opts['means'].param_groups[0]['lr'] = rates['means']*(.01**(step/args.steps))
            for opt in opts.values(): opt.step(); opt.zero_grad(set_to_none=True)
            strategy.step_post_backward(splats, opts, state, step, info, packed=False)
            # Stop adding splats at the memory budget, but keep opacity/size pruning.
            # Stopping all refinement here lets oversized floaters persist and grow.
            if len(splats['means']) > args.max_gaussians:
                if args.refinement_mode in {'continuous-prune','upstream'}:
                    strategy.grow_grad2d = float('inf')
                else:
                    strategy.refine_stop_iter = min(strategy.refine_stop_iter, step+1)
            if step % 100 == 0:
                row = {'step': step, 'name': names[i], 'loss': loss.item(), 'l1': l1.item(), 'ssim': similarity.item(),
                       'render_min': rgb.detach().min().item(), 'render_max': rgb.detach().max().item(),
                       'gaussians': len(splats['means']), 'elapsed_s': time.monotonic()-started}
                log.write(json.dumps(row)+'\n'); log.flush()
                print(json.dumps(row), flush=True)
            if step+1 == 4000:
                evaluate(step+1, False)
    evaluate(args.steps, True)
    torch.save({'splats': splats.state_dict(), 'step': args.steps, 'scene_sha256': hashlib.sha256((args.data/'scene.npz').read_bytes()).hexdigest()},out/'checkpoint.pt')
    export_splats(**{k:splats[k] for k in ['means','scales','quats','opacities','sh0','shN']},format='ply',save_to=str(out/'room.ply'))
    np.savez_compressed(out/'camera-path.npz', c2ws=scene['c2ws'], Ks=scene['Ks'], names=scene['names'], train=scene['train'])
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    report = {'steps': args.steps, 'initial_points': n, 'gaussians': len(splats['means']), 'training_images': len(train_ids),
              'heldout_images':len(heldout_ids),'image_size':[w,h], 'scene_scale_arbitrary_units':scene_scale,
              'metric_scale':None, 'gpu':torch.cuda.get_device_name(), 'cuda':torch.version.cuda,
              'python':platform.python_version(), 'packages':{n:importlib.metadata.version(n) for n in ['gsplat','torch','numpy','pytorch-msssim']},
              'config':{'seed':3915,'rates':rates,'loss':'0.8 L1 + 0.2 (1-SSIM)','refine_stop_iter':strategy.refine_stop_iter,
                        'ssim_implementation':'pytorch-msssim', 'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                        'growth_memory_budget':args.max_gaussians,'refinement_mode':args.refinement_mode,
                        'pruning_continues_after_growth_stops':args.refinement_mode in {'continuous-prune','upstream'},
                        'training_resolution':'half resolution for first 1000 updates, full thereafter','camera_optimization':False},
              'pose_source':scene_metadata.get('pose_source','Training-only COLMAP map and heldout feature PnP'),
              'limitations':scene_metadata.get('limitations',['Single capture session; withheld views are temporally correlated.', 'Monocular arbitrary scale; splat depth is rendered model evidence, not measured depth.',
                             'No collision mesh, robot or dynamics; novel disocclusions are unobserved.', 'No GT camera pose; heldout poses are estimated with feature PnP.'])}
    with torch.no_grad():
        for suffix, displacement in [('route',0.),('lateral',.05*scene_scale)]:
            with imageio.get_writer(str(out/(suffix+'.mp4')), fps=12, codec='libx264', quality=8, macro_block_size=2) as writer:
                for i in range(len(names)):
                    pose = c2ws[i].clone()
                    pose[:3,3] += pose[:3,0]*displacement
                    rgb, _, _ = render(pose, Ks[i])
                    writer.append_data((rgb[0].clamp(0,1).cpu().numpy()*255).astype('uint8'))
    report['elapsed_s'] = time.monotonic()-started
    report['peak_gpu_memory_gb'] = torch.cuda.max_memory_allocated()/1e9
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
