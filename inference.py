import torch
from glob import glob
import os
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from time import time
from argparse import ArgumentParser
import onnxruntime as ort
import numpy as np

from misc import denormalize, divide_img_into_patches

@torch.no_grad()
def predict(session, img, patch_size=3584, log_para=1000):
    h, w = img.shape[2:]
    ps = patch_size
    if h >= ps or w >= ps:
        pred_dmap = torch.zeros(1, 1, h, w)
        pred_count = 0
        img_patches, nh, nw = divide_img_into_patches(img, ps)
        for i in range(nh):
            for j in range(nw):
                patch = img_patches[i * nw + j]
                patch_np = patch.cpu().numpy()  
                pred_dpatch = session.run(None, {'input': patch_np})[0]  
                pred_dpatch = torch.from_numpy(pred_dpatch)  
                pred_dmap[:, :, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = pred_dpatch
    else:
        img_np = img.cpu().numpy()  
        pred_dmap = session.run(None, {'input': img_np})[0]  
        pred_dmap = torch.from_numpy(pred_dmap)  
    pred_count = pred_dmap.sum().cpu().item() / log_para

    return pred_dmap.squeeze().cpu().numpy(), pred_count


def load_imgs(img_path, unit_size, device, pre_resize=1):
    if os.path.isdir(img_path):
        img_paths = glob(os.path.join(img_path, '*'))
    else:
        img_paths = [img_path]

    imgs = []
    for img_path in img_paths:
        assert os.path.exists(img_path), f'Image path {img_path} does not exist.'
        assert img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg') \
            or img_path.lower().endswith('.png'), 'Only support .jpg and .png image format.'

        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.BILINEAR)

        img = F.to_tensor(img)
        img = F.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img = img.unsqueeze(0).to(device)
        imgs.append(img)

    img_names = [os.path.basename(img_path) for img_path in img_paths]

    return imgs, img_names


def load_model(onnx_path, device='cuda'):

    providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


def main(args):
    imgs, img_names = load_imgs(args.img_path, args.unit_size, args.device, args.pre_resize)
    session = load_model(args.model_path, args.device)

    start_time = time()
    for img, img_name in zip(imgs, img_names):
        pred_dmap, pred_count = predict(session, img, args.patch_size, args.log_para)
        print(f'{img_name}: {pred_count}')

        if args.save_path is not None:
            with open(args.save_path, 'a') as f:
                f.write(f'{img_name}: {pred_count}\n')

        if args.vis_dir is not None:
            os.makedirs(args.vis_dir, exist_ok=True)
            denormed_img = denormalize(img)[0].cpu().permute(1, 2, 0).numpy()
            fig = plt.figure(figsize=(10, 5))
            ax_img = fig.add_subplot(121)
            ax_img.imshow(denormed_img)
            ax_img.set_title(img_name)
            ax_dmap = fig.add_subplot(122)
            ax_dmap.imshow(pred_dmap)
            ax_dmap.set_title(f'Predicted count: {pred_count}')
            plt.savefig(os.path.join(args.vis_dir, img_name.split('.')[0] + '.png'))
            plt.close(fig)
    print(f'Total time: {time() - start_time:.2f}s')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image or directory containing images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the ONNX model.')
    parser.add_argument('--save_path', type=str, default=None, help='Path of the text file to save the prediction results.')
    parser.add_argument('--vis_dir', type=str, default=None, help='Directory to save the visualization results.')
    parser.add_argument('--unit_size', type=int, default=16, help='Unit size for image resizing. Normally set to 16 and no need to change.')
    parser.add_argument('--patch_size', type=int, default=3584, help='Patch size for image division. Decrease this value if OOM occurs.')
    parser.add_argument('--log_para', type=int, default=1000, help='Parameter for log transformation. Normally set to 1000 and no need to change.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model. Default is cuda.')
    parser.add_argument('--pre_resize', type=float, default=1.0, help='Resize factor for input images. Default is 1.0 (no resizing).')
    args = parser.parse_args()

    main(args)