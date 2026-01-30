# Metal Plaque Inspection System

Automated metal plaque inspection system using computer vision and deep learning for dimension measurement and defect detection.

## 🔍 Overview

This project implements an automated inspection system for metal plaques using:

- **YOLO** for plaque detection and segmentation
- **Fisheye lens distortion correction**
- **Perspective transformation** for accurate measurements
- **Real-world dimension calculation** using calibrated scales

## 📋 Features

- ✅ Fisheye lens undistortion
- ✅ Automatic image rotation correction
- ✅ YOLO-based plaque detection with instance segmentation
- ✅ Perspective correction using homography
- ✅ Multi-band pixel-to-cm calibration
- ✅ Automatic dimension measurements (length, width, height)
- ✅ Visual overlay with measurement annotations

## 🛠️ Requirements

```bash
pip install opencv-python numpy pandas ultralytics
```

### Dependencies

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Pandas
- Ultralytics YOLO

## 📁 Project Structure

```
Inspection_Plaques/
├── CodeAvtInteg.ipynb       # Main inspection notebook
├── inspection.py            # Standalone Python script
├── best (9).pt              # YOLO model weights
├── echelles_bandes.csv      # Pixel-to-cm calibration data
├── README.md                # This file
└── [input images]           # Images to process
```

## 🚀 Quick Start

### 1. Prepare Your Environment

Ensure you have the YOLO model and calibration file:

- `best (9).pt` - Trained YOLO segmentation model
- `echelles_bandes.csv` - Calibration data for pixel-to-cm conversion

### 2. Configure Parameters

Edit the parameters in the script:

```python
# Input/Output paths
IMG_PATH   = "path/to/your/image.jpg"
CSV_PATH   = "echelles_bandes.csv"
MODEL_PATH = "best (9).pt"
OUT_PATH   = "output_result.png"

# Camera parameters
HFOV_DEG = 93.9  # Horizontal field of view
rotation_angle = -3.0  # Manual rotation correction

# Detection parameters
CONF, IOU = 0.30, 0.60  # YOLO confidence and IOU thresholds
```

### 3. Run the Inspection

#### Using Jupyter Notebook:

```bash
jupyter notebook CodeAvtInteg.ipynb
```

#### Using Python Script:

```bash
python inspection.py
```

## 📊 Calibration File Format

The `echelles_bandes.csv` file contains calibration data for different image regions:

```csv
x_start,x_end,px_cm_X,px_cm_Y
0,500,12.5,10.2
500,1000,13.1,10.8
...
```

- `x_start`, `x_end`: Pixel range boundaries
- `px_cm_X`: Pixels per cm in X direction
- `px_cm_Y`: Pixels per cm in Y direction

## 🎯 How It Works

1. **Distortion Correction**: Removes fisheye lens distortion using camera calibration parameters
2. **Rotation**: Applies rotation correction to align plaques
3. **Detection**: YOLO model detects and segments metal plaques
4. **Mask Processing**: Cleans and refines segmentation masks
5. **Perspective Transform**: Creates top-down view using homography
6. **Measurement**: Calculates real-world dimensions using calibrated scales
7. **Visualization**: Draws measurement grid and annotations

## 📐 Measurement Grid

The system generates:

- **Vertical lines**: Height measurements at multiple positions
- **Horizontal lines**: Width measurements (top, middle, bottom)
- **Annotations**: Real measurements in centimeters

## ⚙️ Advanced Configuration

### Camera Distortion Parameters

```python
D = np.array([[0.025],[0.045],[0.0],[0.0]], dtype=np.float32)
```

### Grid Density

```python
N_LEN = 5   # Number of vertical measurement lines
N_WID = 1   # Number of internal horizontal lines
```

### Visual Style

```python
COL_V, COL_H = (90,0,90), (90,0,0)  # Colors (BGR)
THICK_V, THICK_H = 4, 4              # Line thickness
MASK_ALPHA = 0.25                    # Mask overlay transparency
```

## 📝 Output

The system generates:

- Annotated image with measurement overlay
- Console output with detection statistics
- Visual display of results (if running interactively)

Example output:

```
✅ Undistort appliqué
✅ Rotation appliquée
🔎 Objets détectés: 1
✅ Image finale avec mesures → brute04_result.png
```

## 🔧 Troubleshooting

### No objects detected

- Check YOLO confidence threshold (CONF)
- Verify model path and weights
- Ensure image quality is sufficient

### Incorrect measurements

- Verify calibration CSV file
- Check rotation angle parameter
- Adjust YOLO IOU threshold

### Distortion issues

- Verify HFOV_DEG parameter matches your camera
- Check distortion coefficients (D array)

## 📄 License

This project is for research and industrial inspection purposes.

## 👥 Contributors

- Sara El Koutit

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: Make sure to have sufficient lighting and a clear view of the metal plaque for optimal detection and measurement accuracy.
