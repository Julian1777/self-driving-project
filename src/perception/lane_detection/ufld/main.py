import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ufldv2.model.model_culane import get_model as get_culane_model
from ufldv2.utils.config import Config


def pred2coords(
    pred,
    row_anchor,
    col_anchor,
    local_width=1,
    original_image_width=1640,
    original_image_height=590,
):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred["loc_row"].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred["loc_col"].shape

    max_indices_row = pred["loc_row"].argmax(1).cpu()
    valid_row = pred["exist_row"].argmax(1).cpu()

    max_indices_col = pred["loc_col"].argmax(1).cpu()
    valid_col = pred["exist_col"].argmax(1).cpu()

    pred["loc_row"] = pred["loc_row"].cpu()
    pred["loc_col"] = pred["loc_col"].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, max_indices_row[0, k, i] - local_width),
                                min(
                                    num_grid_row - 1,
                                    max_indices_row[0, k, i] + local_width,
                                )
                                + 1,
                            )
                        )
                    )

                    out_tmp = (
                        pred["loc_row"][0, all_ind, k, i].softmax(0)
                        * all_ind.float()
                    ).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, max_indices_col[0, k, i] - local_width),
                                min(
                                    num_grid_col - 1,
                                    max_indices_col[0, k, i] + local_width,
                                )
                                + 1,
                            )
                        )
                    )

                    out_tmp = (
                        pred["loc_col"][0, all_ind, k, i].softmax(0)
                        * all_ind.float()
                    ).sum() + 0.5
                    out_tmp = (
                        out_tmp / (num_grid_col - 1) * original_image_height
                    )
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords


def generate_smoothed_spline_vote_mask(
    coords, frame_shape, prev_coeffs_dict, lane_width=4, alpha=0.35, min_points=3
):
    """Converts UFLDv2 keypoints into a thin, temporally smoothed binary lane mask.

    - prev_coeffs_dict: State dictionary carrying polynomial history across frames to stop jitter
    - lane_width: Thinner lines (default: 4px)
    - alpha: Smoothing factor between 0 and 1 (lower = smoother/less jitter)
    """
    h, w = frame_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    for lane_idx, lane in enumerate(coords):
        if len(lane) < min_points:
            if lane_idx in prev_coeffs_dict:
                del prev_coeffs_dict[lane_idx]
            continue

        pts = np.array(lane)
        x = pts[:, 0]
        y = pts[:, 1]

        try:
            curr_coeffs = np.polyfit(y, x, 2)

            if prev_coeffs_dict.get(lane_idx) is not None:
                smoothed_coeffs = alpha * curr_coeffs + (1 - alpha) * prev_coeffs_dict[lane_idx]
            else:
                smoothed_coeffs = curr_coeffs

            prev_coeffs_dict[lane_idx] = smoothed_coeffs

            y_min, y_max = int(np.min(y)), int(np.max(y))
            y_dense = np.linspace(y_min, y_max, num=(y_max - y_min + 1))
            x_dense = np.polyval(smoothed_coeffs, y_dense)

            curve_points = np.column_stack((x_dense, y_dense)).astype(np.int32)
            curve_points[:, 0] = np.clip(curve_points[:, 0], 0, w - 1)
            curve_points[:, 1] = np.clip(curve_points[:, 1], 0, h - 1)
            curve_points = curve_points.reshape((-1, 1, 2))

            cv2.polylines(
                binary_mask,
                [curve_points],
                isClosed=False,
                color=255,
                thickness=lane_width,
            )

        except (np.linalg.LinAlgError, ValueError):
            pts_formatted = pts.reshape((-1, 1, 2))
            cv2.polylines(
                binary_mask,
                [pts_formatted],
                isClosed=False,
                color=255,
                thickness=lane_width,
            )

    return binary_mask


class UFLDv2Inference:
    def __init__(self, model_path, config_path, device=None):
        """
        Initialize UFLDv2 inference wrapper with temporal smoothing.

        Args:
            model_path (str): Path to the model checkpoint
            config_path (str): Path to the config file
            device (torch.device): Device to run inference on (cuda or cpu)
        """
        self.device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        self.cfg = self._load_config()
        self._setup_anchors()
        self.model = self._load_model()
        self.model.eval()

        self.img_transform = transforms.Compose([
            transforms.Resize((self.cfg.train_height, self.cfg.train_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.prev_lane_coeffs = {}

        print(f"[UFLDv2] Model loaded from {self.model_path}")
        print(f"[UFLDv2] Config: height={self.cfg.train_height}, width={self.cfg.train_width}, crop_ratio={self.cfg.crop_ratio}")
        print(f"[UFLDv2] Row anchors: {len(self.row_anchor)} points, Col anchors: {len(self.col_anchor)} points")

    def _load_config(self):
        """Load config from Python file."""
        cfg = Config.fromfile(str(self.config_path))
        return cfg

    def _setup_anchors(self):
        """Setup row and column anchors based on config."""
        self.row_anchor = torch.linspace(0.42, 1.0, self.cfg.num_row, dtype=torch.float32)
        self.col_anchor = torch.linspace(0.0, 1.0, self.cfg.num_col, dtype=torch.float32)

    def _load_model(self):
        """Load the model from checkpoint."""
        model = get_culane_model(self.cfg)
        model = model.to(self.device)

        if self.model_path.exists():
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            compatible_state_dict = {}
            for k, v in state_dict.items():
                if 'module.' in k:
                    compatible_state_dict[k[7:]] = v
                else:
                    compatible_state_dict[k] = v

            model.load_state_dict(compatible_state_dict, strict=False)
            print(f"[UFLDv2] Loaded checkpoint from {self.model_path}")
        else:
            print(f"[UFLDv2] Warning: Model checkpoint not found at {self.model_path}")

        return model

    def _preprocess(self, image, debug=False):
        """Preprocess image for UFLDv2 inference."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img_tensor = self.img_transform(image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor

    def infer(self, image, debug=False):
        """
        Run inference on an image and return binary lane mask.

        Args:
            image (numpy array): RGB image (H, W, 3) in uint8. Image is cropped from bottom before processing.
            debug (bool): Whether to print debug info

        Returns:
            numpy array: Binary lane mask (H, W) with lanes drawn as white pixels in full frame coordinates
        """
        if image is None or image.size == 0:
            return np.zeros((image.shape[0] if image is not None else 480,
                           image.shape[1] if image is not None else 640), dtype=np.uint8)

        try:
            h_orig, w_orig = image.shape[:2]

            crop_h = int(h_orig * self.cfg.crop_ratio)
            cropped_image = image[h_orig - crop_h:, :, :]

            if debug:
                print(f"[UFLDv2] Original size: {h_orig}x{w_orig}, Cropped: {crop_h}px from bottom")

            img_tensor = self._preprocess(cropped_image, debug=debug)

            with torch.no_grad():
                pred = self.model(img_tensor)

            coords = pred2coords(
                pred,
                self.row_anchor,
                self.col_anchor,
                original_image_width=w_orig,
                original_image_height=h_orig,
            )

            ufld_mask = generate_smoothed_spline_vote_mask(
                coords,
                (h_orig, w_orig),
                prev_coeffs_dict=self.prev_lane_coeffs,
                lane_width=4,
                alpha=0.25
            )

            return (ufld_mask > 0).astype(np.uint8)

        except Exception as e:
            print(f"[UFLDv2] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
