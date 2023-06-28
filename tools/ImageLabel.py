import sys
sys.path.append("E:\Projects\segment-anything-main")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Any, Dict, List
import os
class ImageAutoLabeling:
    def __init__(self, checkpoint, model_type, device):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device
        
    def run(self, img):
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=self.device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        
        img = self.show_anns(image, masks)
        return img, masks
    def show_anns(self, image, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, :3] = image / 255
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        return img
    def write_masks_to_folder(self, masks: List[Dict[str, Any]]) -> None:
        header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
        metadata = [header]
        for i, mask_data in enumerate(masks):
            # mask = mask_data["segmentation"]
            # filename = f"{i}.png"
            # cv2.imwrite(os.path.join(path, filename), mask * 255)
            mask_metadata = [
                str(i),
                str(mask_data["area"]),
                *[str(x) for x in mask_data["bbox"]],
                *[str(x) for x in mask_data["point_coords"][0]],
                str(mask_data["predicted_iou"]),
                str(mask_data["stability_score"]),
                *[str(x) for x in mask_data["crop_box"]],
            ]
            row = ",".join(mask_metadata)
            metadata.append(row)
        metadata_path = os.path.join("metadata.csv")
        with open(metadata_path, "w") as f:
            f.write("\n".join(metadata))

        return