# Dataset Conversion Tool: Custom Format to YOLOv5 📁➡️🎯

This repository contains a Python script to convert datasets from a custom class-based directory structure to the YOLOv5 format required for object detection training.

## 📊 Dataset Format Conversion

### Source Format
```
├── Plain Background
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── ...
│   └── 35
├── Random Background
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── ...
│   └── 35
```

### Target Format (YOLOv5)
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

## 🔧 Requirements

### Python Version
- Python 3.7 or higher

### Required Python Packages

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install Pillow numpy
```

### Package Details

- **Pillow (PIL)** - For image processing and manipulation
  ```bash
  pip install Pillow
  ```

- **NumPy** - For numerical operations and array handling
  ```bash
  pip install numpy
  ```

### Optional Packages (for YOLOv5 training)

If you plan to use YOLOv5 for training after conversion, you'll also need:

```bash
pip install torch torchvision torchaudio
pip install ultralytics
```

Or clone and install YOLOv5 directly:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/dataset-converter.git
   cd dataset-converter
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

1. **Configure the script**: Edit the `convert_to_yolov5.py` file and update the following paths:
   ```python
   source_root = 'path/to/your/current/dataset'  # Your source dataset path
   target_root = 'path/to/new/yolov5/dataset'    # Target YOLOv5 dataset path
   ```

2. **Run the conversion script**:
   ```bash
   python convert_to_yolov5.py
   ```

3. **Verify the output**: Check that the `target_root` directory contains:
   - `images/train/` and `images/val/` with your images
   - `labels/train/` and `labels/val/` with corresponding label files
   - `dataset.yaml` configuration file

## ⚙️ Configuration Options

### Validation Split
Adjust the train/validation split ratio by modifying:
```python
val_split = 0.2  # 20% for validation, 80% for training
```

### Bounding Box Annotations
The script currently creates default bounding boxes (centered, 80% of image size). If you have actual bounding box data, modify the `create_label_file()` function.

### Class Names
Update the class names in the generated `dataset.yaml` file with meaningful names instead of numeric IDs.

## 📂 File Structure After Conversion

```
your-target-directory/
├── images/
│   ├── train/
│   │   ├── Plain_Background_0_image1.jpg
│   │   ├── Random_Background_1_image2.jpg
│   │   └── ...
│   └── val/
│       ├── Plain_Background_0_image3.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── Plain_Background_0_image1.txt
│   │   ├── Random_Background_1_image2.txt
│   │   └── ...
│   └── val/
│       ├── Plain_Background_0_image3.txt
│       └── ...
└── dataset.yaml
```

## 🏋️ Training with YOLOv5

After conversion, you can train your YOLOv5 model using:

```bash
python train.py --data path/to/your/dataset.yaml --weights yolov5s.pt --epochs 100
```

## 📝 requirements.txt

Create a `requirements.txt` file with the following content:

```
Pillow>=8.0.0
numpy>=1.21.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔧 Troubleshooting

### Common Issues

1. **Import Error**: Make sure all required packages are installed
   ```bash
   pip install --upgrade Pillow numpy
   ```

2. **Path Issues**: Ensure your source and target paths are correct and accessible

3. **Memory Issues**: For large datasets, consider processing in batches by modifying the script

4. **Image Format Issues**: The script supports common formats (PNG, JPG, JPEG). For other formats, modify the file extension filter in the script.

## 💬 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information about your problem

---

⭐ **Star this repository if it helped you!** ⭐
