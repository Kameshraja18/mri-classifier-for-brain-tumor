# Brain Tumor Detection System

This application uses a U-Net CNN model with EfficientNet backbone for precise brain tumor segmentation in MRI scans.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download BraTS dataset:
   - Visit: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
   - Download the dataset (requires Kaggle account)
   - Extract the zip file to `BraTS2020_TrainingData` in the project root

3. Train the model:
```bash
python train.py
```

4. Run the application:
```bash
streamlit run main.py
```

## Usage
- Upload an MRI scan image (PNG, JPG, JPEG)
- The system will show the original image and tumor detection overlay
- Tumor area percentage will be displayed

## Model Architecture
- U-Net with EfficientNet-B3 encoder
- Connected component postprocessing
- Transfer learning with frozen encoder weights
"# mri-classifier" 
