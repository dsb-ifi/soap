from PIL import Image
import numpy as np

def resize_pil(I, patch_size=16) : 
    w, h = I.size
    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    feat_w, feat_h = new_w // patch_size, new_h // patch_size
    return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h

def mask_color_compose(org, mask, mask_color = [173, 216, 230]) :
    mask_fg = mask > 0.5
    rgb = np.copy(org)
    rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)