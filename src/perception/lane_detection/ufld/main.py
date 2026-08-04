import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ufldv2.model.model_culane import get_model as get_culane_model
from ufldv2.utils.config import Config


class UFLDv2Inference:
    def __init__(self, model_path, config_path, device=None):
        """
        Initialize UFLDv2 inference wrapper.

        Args:
            model_path (str): Path to the model checkpoint (models/ufld/culane_res18.pth)
            config_path (str): Path to the config file (ufldv2/configs/culane_res18.py)
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

        print(f"[UFLDv2] Model loaded from {self.model_path}")
        print(f"[UFLDv2] Config: height={self.cfg.train_height}, width={self.cfg.train_width}")
        print(f"[UFLDv2] Num rows={self.cfg.num_row}, Num cols={self.cfg.num_col}, Max lanes={self.cfg.num_lanes}")
        print(f"[UFLDv2] Row anchors: {len(self.row_anchor)} points, Col anchors: {len(self.col_anchor)} points")

    def _load_config(self):
        """Load config from Python file."""
        cfg = Config.fromfile(str(self.config_path))
        return cfg

    def _setup_anchors(self):
        """Setup row and column anchors based on config."""
        # Generate row anchors: evenly spaced from 0.42 to 1.0 (bottom to top of image)
        # CULane training height is 320, so these correspond to pixel positions
        self.row_anchor = torch.linspace(0.42, 1.0, self.cfg.num_row, dtype=torch.float32)

        # Generate col anchors: evenly spaced from 0.0 to 1.0 (left to right of image)
        # CULane training width is 1600
        self.col_anchor = torch.linspace(0.0, 1.0, self.cfg.num_col, dtype=torch.float32)

        if hasattr(self.cfg, '_debug_anchors'):
            print(f"[UFLDv2] Row anchors ({len(self.row_anchor)}): {self.row_anchor[:3]}...{self.row_anchor[-3:]}")
            print(f"[UFLDv2] Col anchors ({len(self.col_anchor)}): {self.col_anchor[:3]}...{self.col_anchor[-3:]}")

    def _load_model(self):
        """Load the model from checkpoint."""
        model = get_culane_model(self.cfg)
        model = model.to(self.device)

        if self.model_path.exists():
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            # Handle both direct state_dict and wrapped 'model' key
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (from DataParallel)
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
        """
        Preprocess image for UFLDv2 inference.

        Args:
            image (numpy array): RGB image (H, W, 3) in uint8
            debug (bool): Whether to print debug info

        Returns:
            torch.Tensor: Preprocessed image tensor (1, 3, H, W)
        """
        # Convert numpy array to PIL Image for transforms
        from PIL import Image
        if isinstance(image, np.ndarray):
            if debug:
                print(f"[UFLDv2] Converting numpy array to PIL Image...")
            image = Image.fromarray(image)

        if debug:
            print(f"[UFLDv2] Applying transforms...")

        img_tensor = self.img_transform(image)
        if debug:
            print(f"[UFLDv2] After transforms: {img_tensor.shape}, dtype: {img_tensor.dtype}")

        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        if debug:
            print(f"[UFLDv2] After unsqueeze and to device: {img_tensor.shape}")

        return img_tensor

    def _extract_lanes(self, pred_dict, image_shape, debug=True):
        """
        Extract lane coordinates from model output using proper anchor-based decoding.

        Args:
            pred_dict (dict): Model output dict with loc_row, exist_row, loc_col, exist_col
            image_shape (tuple): Original image shape (H, W)
            debug (bool): Whether to print debug info

        Returns:
            list: List of lane coordinate arrays (each is Nx2 for cv2.polylines)
        """
        lanes = []
        h_orig, w_orig = image_shape[:2]

        loc_row = pred_dict['loc_row'].cpu()
        exist_row = pred_dict['exist_row'].cpu()
        loc_col = pred_dict['loc_col'].cpu()
        exist_col = pred_dict['exist_col'].cpu()

        # Apply argmax on class dimension (dim 1) to get class predictions
        # Shape: [batch, num_classes, num_anchors, num_lanes] -> [batch, num_anchors, num_lanes]
        exist_row_class = exist_row.argmax(1)
        exist_col_class = exist_col.argmax(1)

        if debug:
            print(f"[UFLDv2 DEBUG] Input image shape: {h_orig}x{w_orig}")
            print(f"[UFLDv2 DEBUG] loc_row shape: {loc_row.shape}")
            print(f"[UFLDv2 DEBUG] exist_row shape (original): {exist_row.shape}")
            print(f"[UFLDv2 DEBUG] exist_row_class shape (after argmax): {exist_row_class.shape}")
            print(f"[UFLDv2 DEBUG] exist_row_class values (lane 1): {exist_row_class[0, :, 1]}")
            print(f"[UFLDv2 DEBUG] exist_row_class values (lane 2): {exist_row_class[0, :, 2]}")
            print(f"[UFLDv2 DEBUG] loc_col shape: {loc_col.shape}")
            print(f"[UFLDv2 DEBUG] exist_col shape (original): {exist_col.shape}")
            print(f"[UFLDv2 DEBUG] exist_col_class shape (after argmax): {exist_col_class.shape}")
            print(f"[UFLDv2 DEBUG] exist_col_class values (lane 0): {exist_col_class[0, :, 0]}")
            print(f"[UFLDv2 DEBUG] exist_col_class values (lane 3): {exist_col_class[0, :, 3]}")

        batch_size = loc_row.shape[0]

        for batch_idx in range(batch_size):
            # Process row-based lanes (indices 1, 2)
            for lane_idx in [1, 2]:
                tmp = []
                valid_row = (exist_row_class[batch_idx, :, lane_idx] == 1).cpu().numpy()
                num_valid = valid_row.sum()

                if debug:
                    print(f"[UFLDv2 DEBUG] Lane {lane_idx} (row-based): {num_valid}/{len(valid_row)} valid rows (threshold: {len(valid_row)/2})")

                if num_valid > len(valid_row) / 2:
                    max_indices = loc_row[batch_idx, :, :, lane_idx].argmax(0)

                    for row_idx in range(len(valid_row)):
                        if valid_row[row_idx]:
                            # Get indices around the max
                            local_width = 1
                            max_idx = max_indices[row_idx].item()
                            all_indices = torch.arange(
                                max(0, max_idx - local_width),
                                min(self.cfg.num_cell_row - 1, max_idx + local_width) + 1
                            )

                            # Softmax and weighted sum to get precise x position
                            scores = loc_row[batch_idx, all_indices, row_idx, lane_idx].softmax(0)
                            out_x = (scores * all_indices.float()).sum() + 0.5
                            out_x = out_x / (self.cfg.num_cell_row - 1) * w_orig
                            out_y = self.row_anchor[row_idx] * h_orig

                            if debug and row_idx < 2:
                                print(f"  Row {row_idx}: max_idx={max_idx}, out_x={out_x:.1f}, out_y={out_y:.1f}")

                            tmp.append((int(out_x.item()), int(out_y.item())))

                    if debug:
                        print(f"  Lane {lane_idx} extracted {len(tmp)} points")

                if len(tmp) >= 2:
                    lanes.append(np.array(tmp, dtype=np.int32))
                    if debug:
                        print(f"  Lane {lane_idx} ADDED to lanes list")

            # Process col-based lanes (indices 0, 3)
            for lane_idx in [0, 3]:
                tmp = []
                valid_col = (exist_col_class[batch_idx, :, lane_idx] == 1).cpu().numpy()
                num_valid = valid_col.sum()

                if debug:
                    print(f"[UFLDv2 DEBUG] Lane {lane_idx} (col-based): {num_valid}/{len(valid_col)} valid cols (threshold: {len(valid_col)/4})")

                if num_valid > len(valid_col) / 4:
                    max_indices = loc_col[batch_idx, :, :, lane_idx].argmax(0)

                    for col_idx in range(len(valid_col)):
                        if valid_col[col_idx]:
                            # Get indices around the max
                            local_width = 1
                            max_idx = max_indices[col_idx].item()
                            all_indices = torch.arange(
                                max(0, max_idx - local_width),
                                min(self.cfg.num_cell_col - 1, max_idx + local_width) + 1
                            )

                            # Softmax and weighted sum to get precise y position
                            scores = loc_col[batch_idx, all_indices, col_idx, lane_idx].softmax(0)
                            out_y = (scores * all_indices.float()).sum() + 0.5
                            out_y = out_y / (self.cfg.num_cell_col - 1) * h_orig
                            out_x = self.col_anchor[col_idx] * w_orig

                            if debug and col_idx < 2:
                                print(f"  Col {col_idx}: max_idx={max_idx}, out_x={out_x:.1f}, out_y={out_y:.1f}")

                            tmp.append((int(out_x.item()), int(out_y.item())))

                    if debug:
                        print(f"  Lane {lane_idx} extracted {len(tmp)} points")

                if len(tmp) >= 2:
                    lanes.append(np.array(tmp, dtype=np.int32))
                    if debug:
                        print(f"  Lane {lane_idx} ADDED to lanes list")

        if debug:
            print(f"[UFLDv2 DEBUG] Total lanes extracted: {len(lanes)}")

        return lanes

    def infer(self, image, debug=True):
        """
        Run inference on an image and return binary lane mask.

        Args:
            image (numpy array): RGB image (H, W, 3) in uint8
            debug (bool): Whether to print debug info

        Returns:
            numpy array: Binary lane mask (H, W) with lanes drawn as white pixels
        """
        if image is None or image.size == 0:
            if debug:
                print("[UFLDv2] Input image is None or empty!")
            return np.zeros((image.shape[0] if image is not None else 480,
                           image.shape[1] if image is not None else 640), dtype=np.uint8)

        try:
            h_orig, w_orig = image.shape[:2]
            if debug:
                print(f"\n[UFLDv2] === INFERENCE START ===")
                print(f"[UFLDv2] Input image shape: {image.shape}, dtype: {image.dtype}")

            img_tensor = self._preprocess(image, debug=debug)
            if debug:
                print(f"[UFLDv2] Tensor value range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

            with torch.no_grad():
                pred_dict = self.model(img_tensor)

            if debug:
                print(f"[UFLDv2] Model output keys: {pred_dict.keys()}")

            lanes = self._extract_lanes(pred_dict, image.shape, debug=debug)

            binary_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

            if debug:
                print(f"[UFLDv2] Drawing {len(lanes)} lanes on mask...")

            for lane_idx, lane in enumerate(lanes):
                if len(lane) >= 2:
                    cv2.polylines(binary_mask, [lane], isClosed=False, color=1, thickness=15)
                    if debug:
                        print(f"[UFLDv2] Lane {lane_idx}: {len(lane)} points drawn, pixels in mask: {np.sum(binary_mask)}")

            if debug:
                print(f"[UFLDv2] Final mask pixels: {np.sum(binary_mask)}")
                print(f"[UFLDv2] === INFERENCE END ===\n")

            return binary_mask
        except Exception as e:
            print(f"[UFLDv2] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    def get_lanes_as_polylines(self, image, debug=False):
        """
        Get detected lanes as polyline coordinates (useful for visualization).

        Args:
            image (numpy array): RGB image (H, W, 3)
            debug (bool): Whether to print debug info

        Returns:
            list: List of lane coordinate arrays
        """
        if image is None or image.size == 0:
            return []

        try:
            img_tensor = self._preprocess(image, debug=debug)

            with torch.no_grad():
                pred_dict = self.model(img_tensor)

            lanes = self._extract_lanes(pred_dict, image.shape, debug=debug)
            return lanes
        except Exception as e:
            print(f"[UFLDv2] Lane extraction error: {e}")
            import traceback
            traceback.print_exc()
            return []
