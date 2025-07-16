
# 🛰️ Satellite Image Segmentation using U-Net with ResNet34

This project focuses on segmenting satellite images using a deep learning model based on **U-Net** architecture with a **pretrained ResNet34 backbone**. The objective is to accurately extract meaningful features like land, vegetation, or buildings from high-resolution satellite imagery.

---

## 📌 Features

- 🔍 **Semantic Segmentation** of satellite images  
- 🧠 **U-Net** architecture with a **ResNet34 encoder (ImageNet pretrained)**  
- ⚙️ Combined **Binary Crossentropy + Dice Loss** for better performance on imbalanced data  
- 🔁 Data preprocessing and augmentation  
- 📉 Uses callbacks like EarlyStopping and ReduceLROnPlateau  
- 📦 Trained using TensorFlow/Keras with support for Google Colab  

---

## 🧪 Dataset

- Satellite images and their corresponding binary masks.
- Images are converted from `.tif` to `.png` and resized to `256x256`.
- Masks represent target features for segmentation (e.g., buildings or vegetation).

---

## 🧰 Model Architecture

- **U-Net** with an **encoder from ResNet34** pretrained on ImageNet
- Final layer uses `sigmoid` activation for binary segmentation
- Uses `segmentation-models` library for easy backbone swapping

---

## 🧑‍💻 Setup Instructions

```bash
# Install compatible versions
pip uninstall -y keras keras-nightly
pip install keras==2.11.0
pip install segmentation-models==1.0.1
```

```python
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

# Load the model
model = sm.Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')
```

---

## 🏋️ Training

```python
# Compile model
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])

# Train
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)
```

---

## 📈 Evaluation & Prediction

```python
# Predict and threshold
pred_mask = model.predict(test_image)
pred_mask = (pred_mask > 0.5).astype('uint8')
```

---

## 📊 Results

> Add sample input/output images and metrics here (IoU, Dice coefficient, etc.)

---

## 🚀 Future Improvements

- Multi-class segmentation
- Deploying as a web app (Streamlit or Gradio)
- Experiment with larger backbones (EfficientNet, ResNeXt)

---

## 🧾 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests and suggestions are welcome! If you find a bug or want a feature, feel free to open an issue.
