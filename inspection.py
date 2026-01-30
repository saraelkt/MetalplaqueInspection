#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metal Plaque Inspection System
Automated dimension measurement using YOLO and perspective transformation
"""

import os
import math
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# =========================
# CONFIGURATION PARAMETERS
# =========================

# Camera parameters
HFOV_DEG = 93.9  # Horizontal field of view in degrees
D = np.array([[0.025], [0.045], [0.0], [0.0]], dtype=np.float32)  # Distortion coefficients

# File paths
IMG_PATH = r"D:\Users\Hp\Desktop\Inspection_Plaques\brute04.jpg"
CSV_PATH = r"D:\Users\Hp\Desktop\Inspection_Plaques\echelles_bandes.csv"
MODEL_PATH = r"D:\Users\Hp\Desktop\Inspection_Plaques\best (9).pt"
OUT_PATH = r"D:\Users\Hp\Desktop\Inspection_Plaques\brute04_result.png"

# Image correction
rotation_angle = -3.0  # Rotation angle in degrees

# YOLO detection parameters
CONF = 0.30  # Confidence threshold
IOU = 0.60   # IOU threshold

# Measurement grid parameters
N_LEN = 5   # Number of vertical measurement lines
N_WID = 1   # Number of internal horizontal lines

# Visualization parameters
COL_V = (90, 0, 90)      # Vertical lines color (BGR)
COL_H = (90, 0, 0)       # Horizontal lines color (BGR)
THICK_V = 4              # Vertical lines thickness
THICK_H = 4              # Horizontal lines thickness
MASK_ALPHA = 0.25        # Mask overlay transparency
FONT = cv2.FONT_HERSHEY_SIMPLEX
TXT_COL = (255, 255, 255)  # Text color
TXT_TH = 2                  # Text thickness

# =========================
# UTILITY FUNCTIONS
# =========================

def rotate_image_keep_bounds(image, angle_deg):
    """Rotate image while keeping all content visible"""
    (h, w) = image.shape[:2]
    cX, cY = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - cX
    M[1, 2] += (new_h / 2) - cY
    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )


def bande_de_x(x, x_start, x_end):
    """Find calibration band index for given x coordinate"""
    for i in range(len(x_start)):
        if x_start[i] <= x < x_end[i]:
            return i
    return 0 if x < x_start[0] else len(x_start) - 1


def long_cm_bandes(x0, x1, x_start, x_end, px_cm_X):
    """Calculate length in cm across multiple calibration bands"""
    cm = 0.0
    for xs, xe, r in zip(x_start, x_end, px_cm_X):
        left, right = max(x0, xs), min(x1, xe)
        if right > left:
            cm += (right - left) / r
    return cm


def clean_mask(mask):
    """Clean segmentation mask by removing noise and filling holes"""
    m = (mask > 0).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    
    # Keep only largest component
    if n > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (lab == idx).astype(np.uint8) * 255
    
    # Morphological operations
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m


def fit_top_bottom_lines(mask):
    """Fit lines to top and bottom edges of mask"""
    xs = np.where(mask.any(axis=0))[0]
    top_pts, bot_pts = [], []
    
    for x in xs:
        ys = np.where(mask[:, x])[0]
        if ys.size:
            top_pts.append([x, ys.min()])
            bot_pts.append([x, ys.max()])
    
    top_pts = np.array(top_pts, np.float32)
    bot_pts = np.array(bot_pts, np.float32)
    
    # Fit lines
    vxT, vyT, x0T, y0T = cv2.fitLine(top_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vxB, vyB, x0B, y0B = cv2.fitLine(bot_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Convert to scalars
    vxT, vyT, x0T, y0T = vxT.item(), vyT.item(), x0T.item(), y0T.item()
    vxB, vyB, x0B, y0B = vxB.item(), vyB.item(), x0B.item(), y0B.item()
    
    # Create functions for y values
    def y_top_at(x):
        return int(round(y0T + (vyT / vxT) * (x - x0T)))
    
    def y_bot_at(x):
        return int(round(y0B + (vyB / vxB) * (x - x0B)))
    
    return y_top_at, y_bot_at, xs.min(), xs.max()


def vertical_range_robust(mask, x, win, y_top_at, y_bot_at, rel=0.6):
    """Get robust vertical range at given x position"""
    H, W = mask.shape
    x0, x1 = max(0, x - win), min(W - 1, x + win)
    
    # Predicted positions
    y0p, y1p = y_top_at(x), y_bot_at(x)
    y0p = max(0, min(H - 1, y0p))
    y1p = max(0, min(H - 1, y1p))
    
    # Check actual mask
    col_any = mask[:, x0:x1 + 1].any(axis=1)
    ys = np.where(col_any)[0]
    
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        if (y1 - y0) < rel * max(1, (y1p - y0p)):
            return y0p, y1p
        return y0, y1
    
    return y0p, y1p


# =========================
# MAIN PROCESSING
# =========================

def main():
    print("=" * 60)
    print("Metal Plaque Inspection System")
    print("=" * 60)
    
    # 1. Load and undistort image
    print("\n[1/4] Loading and correcting image distortion...")
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise RuntimeError(f"Cannot read image: {IMG_PATH}")
    
    h, w = img.shape[:2]
    
    # Calculate camera matrix
    f = w / (2 * math.tan(math.radians(HFOV_DEG / 2)))
    K = np.array([
        [f, 0, w / 2],
        [0, f, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Undistort fisheye
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=1.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    img = cv2.remap(
        img, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    print("✅ Fisheye distortion corrected")
    
    # 2. Apply rotation
    print("\n[2/4] Applying rotation correction...")
    img = rotate_image_keep_bounds(img, rotation_angle)
    print(f"✅ Rotated {rotation_angle}°")
    
    # 3. Load calibration data
    print("\n[3/4] Loading calibration data...")
    tab = pd.read_csv(CSV_PATH)
    x_start = tab["x_start"].to_numpy()
    x_end = tab["x_end"].to_numpy()
    px_cm_X = tab["px_cm_X"].to_numpy()
    px_cm_Y = tab["px_cm_Y"].to_numpy()
    print(f"✅ Loaded {len(x_start)} calibration bands")
    
    # 4. Run YOLO detection
    print("\n[4/4] Running YOLO detection...")
    model = YOLO(MODEL_PATH)
    pred = model.predict(
        img,
        conf=CONF,
        iou=IOU,
        imgsz=1024,
        rect=True,
        save=False,
        retina_masks=True
    )[0]
    
    num_masks = 0 if (pred.masks is None or pred.masks.data is None) else len(pred.masks.data)
    print(f"🔎 Detected objects: {num_masks}")
    
    overlay = img.copy()
    
    if num_masks == 0:
        cv2.imwrite(OUT_PATH, overlay)
        print("\n⚠️  No masks detected by YOLO on this image.")
        return
    
    # Process each detected plaque
    for idx, (m_raw, box) in enumerate(zip(
        pred.masks.data.cpu().numpy(),
        pred.boxes.xyxy.cpu().numpy().astype(int)
    ), 1):
        print(f"\n  Processing plaque {idx}/{num_masks}...")
        
        # Resize mask to full resolution
        mask = cv2.resize(
            (m_raw > 0.35).astype(np.uint8) * 255,
            img.shape[:2][::-1],
            cv2.INTER_NEAREST
        )
        mask = clean_mask(mask)
        
        # Fit top/bottom boundary lines
        y_top_at, y_bot_at, xs_min, xs_max = fit_top_bottom_lines(mask)
        
        # Set up homography for metric conversion
        H_CM = 201.5  # Fixed height in cm
        PX_PER_CM = 10
        
        xL, xR = xs_min, xs_max
        pTL = (xL, y_top_at(xL))
        pTR = (xR, y_top_at(xR))
        pBR = (xR, y_bot_at(xR))
        pBL = (xL, y_bot_at(xL))
        
        # Calculate output dimensions
        h_px_left = pBL[1] - pTL[1]
        h_px_right = pBR[1] - pTR[1]
        h_px_avg = (h_px_left + h_px_right) / 2.0
        SCALE_Y = H_CM / max(h_px_avg, 1e-6)
        
        w_px_top = np.hypot(pTR[0] - pTL[0], pTR[1] - pTL[1])
        w_px_bot = np.hypot(pBR[0] - pBL[0], pBR[1] - pBL[1])
        W_CM_est = SCALE_Y * (w_px_top + w_px_bot) / 2.0
        
        W_out = int(round(W_CM_est * PX_PER_CM))
        H_out = int(round(H_CM * PX_PER_CM))
        
        # Compute homography
        src = np.float32([pTL, pTR, pBR, pBL])
        dst = np.float32([[0, 0], [W_out - 1, 0], [W_out - 1, H_out - 1], [0, H_out - 1]])
        Hmat, _ = cv2.findHomography(src, dst)
        
        def to_metric_cm(pts_xy):
            """Convert pixel coordinates to metric coordinates in cm"""
            pts_xy = np.asarray(pts_xy, np.float32).reshape(-1, 1, 2)
            pts_m = cv2.perspectiveTransform(pts_xy, Hmat).reshape(-1, 2)
            return pts_m / PX_PER_CM
        
        # Create measurement grid
        grid = np.zeros((*mask.shape, 3), np.uint8)
        
        # Draw vertical measurement lines
        for t in np.linspace(0, 1, N_LEN):
            x = int(round(xs_min + t * (xs_max - xs_min)))
            
            if t == 0 or t == 1:
                y0, y1 = y_top_at(x), y_bot_at(x)
            else:
                y0, y1 = vertical_range_robust(
                    mask, x, win=6,
                    y_top_at=y_top_at,
                    y_bot_at=y_bot_at,
                    rel=0.6
                )
            
            if y0 is None:
                continue
            
            cv2.line(grid, (x, y0), (x, y1), COL_V, THICK_V)
            P0_cm, P1_cm = to_metric_cm([(x, y0), (x, y1)])
            long_cm_v = float(np.linalg.norm(P1_cm - P0_cm))
            cv2.putText(
                overlay,
                f"{long_cm_v:.0f} cm",
                (x + 8, (y0 + y1) // 2),
                FONT, 0.9, TXT_COL, TXT_TH, cv2.LINE_AA
            )
        
        # Draw horizontal measurement lines
        ys_mask = np.where(mask.any(axis=1))[0]
        if ys_mask.size:
            y_min, y_max = ys_mask.min(), ys_mask.max()
            
            # Top line
            x_left, x_right = xs_min, xs_max
            y_left_top = y_top_at(x_left)
            y_right_top = y_top_at(x_right)
            cv2.line(grid, (x_left, y_left_top), (x_right, y_right_top), COL_H, THICK_H)
            A_cm, B_cm = to_metric_cm([(x_left, y_left_top), (x_right, y_right_top)])
            long_cm_top = float(np.linalg.norm(B_cm - A_cm))
            cv2.putText(
                overlay,
                f"{long_cm_top:.2f} cm",
                ((x_left + x_right) // 2, min(y_left_top, y_right_top) - 10),
                FONT, 0.9, TXT_COL, TXT_TH, cv2.LINE_AA
            )
            
            # Bottom line
            bottom_pts = []
            for x in np.where(mask.any(axis=0))[0]:
                ys = np.where(mask[:, x])[0]
                bottom_pts.append([x, ys.max()])
            bottom_pts = np.array(bottom_pts, dtype=np.float32)
            
            vx, vy, x0b, y0b = cv2.fitLine(bottom_pts, cv2.DIST_L2, 0, 0.01, 0.01)
            x_left = int(bottom_pts[:, 0].min())
            x_right = int(bottom_pts[:, 0].max())
            y_left = int(y0b + vy / vx * (x_left - x0b))
            y_right = int(y0b + vy / vx * (x_right - x0b))
            cv2.line(grid, (x_left, y_left), (x_right, y_right), COL_H, THICK_H)
            A2_cm, B2_cm = to_metric_cm([(x_left, y_left), (x_right, y_right)])
            long_cm_bot = float(np.linalg.norm(B2_cm - A2_cm))
            cv2.putText(
                overlay,
                f"{long_cm_bot:.2f} cm",
                ((x_left + x_right) // 2, y_right + 22),
                FONT, 0.9, TXT_COL, TXT_TH, cv2.LINE_AA
            )
            
            # Middle lines
            for s in np.linspace(0, 1, N_WID + 2)[1:-1]:
                y_mid = int(round(y_min + s * (y_max - y_min)))
                xs_mid = np.where(mask[y_mid])[0]
                if not xs_mid.size:
                    continue
                
                cv2.line(grid, (xs_mid.min(), y_mid), (xs_mid.max(), y_mid), COL_H, THICK_H)
                A3_cm, B3_cm = to_metric_cm([(xs_mid.min(), y_mid), (xs_mid.max(), y_mid)])
                long_cm_mid = float(np.linalg.norm(B3_cm - A3_cm))
                cv2.putText(
                    overlay,
                    f"{long_cm_mid:.2f} cm",
                    ((xs_mid.min() + xs_mid.max()) // 2, y_mid + 22),
                    FONT, 0.9, TXT_COL, TXT_TH, cv2.LINE_AA
                )
        
        # Blend mask and grid with image
        mask_rgb = cv2.merge([np.zeros_like(mask), mask, np.zeros_like(mask)])
        overlay = cv2.addWeighted(overlay, 1.0, mask_rgb, MASK_ALPHA, 0)
        overlay = cv2.addWeighted(overlay, 1.0, grid, 1.0, 0)
    
    # Save result
    cv2.imwrite(OUT_PATH, overlay)
    print(f"\n✅ Result saved to: {OUT_PATH}")
    
    # Display result
    cv2.imshow("Inspection Result", cv2.resize(overlay, (1200, 700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Inspection completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
