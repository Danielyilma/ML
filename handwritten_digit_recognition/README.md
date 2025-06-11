# 🧠 Handwritten Digit Recognition

A machine learning project to recognize handwritten digits (0–9) using the MNIST dataset. This includes:

- **Data loading & preprocessing**
- **CNN model training and evaluation**
- **Optional Flask web interface** for image upload and prediction

---

## 📁 Project Structure

```
handwritten_digit_recognition/
├── handwritten_digit_recognition.ipynb   # Exploratory notebook: data, model training, results
├── train_model.py                       # Script: builds, trains, and saves CNN model (mnist_cnn_model.h5)
├── app.py                               # Flask app: handles image upload, preprocessing & predictions
├── mnist_cnn_model.h5                   # Pre-trained model (optional to include)
├── test_img.jpg                         # Sample test digit image
└── requirements.txt                     # Python dependencies
```

---

## 📋 Requirements

Key dependencies (install using `pip install -r requirements.txt`):

- `tensorflow` / `keras`
- `numpy`
- `flask`
- `Pillow`

---

## 🧪 Training the Model

Use `train_model.py` to train your CNN:

```bash
python train_model.py
```

What it does:

1. **Load & preprocess** MNIST (28×28 grayscale): reshape & normalize.
2. **Define CNN architecture** (Conv → Pool → Dense).
3. **Train with callbacks**: early stopping & LR reduction.
4. **Save model** as `mnist_cnn_model.h5`.

---

## 🚀 Running the Web App

Use `app.py` to start a Flask UI:

```bash
flask run app.py
```

App features:

1. **Image upload** (JPEG, PNG).
2. **Preprocessing**: resize to 28×28, grayscale, normalize, reshape.
3. **Prediction** using the trained CNN model.
4. **Display of image & result** via a simple Flask web interface.

---

## 📊 Notebook Explorations

`handwritten_digit_recognition.ipynb` walks through:

- Loading and visualizing MNIST.
- Training/validating the CNN.
- Displaying metrics: accuracy & loss curves.
- Running noisy/pseudo-input predictions.
- (Optional) Data augmentation exploration.

---

## 🛠 Usage Workflow

**Training**:

```bash
python train_model.py
```

**Launching App**:

```bash
flask run app.py
```

Workflow Recap:

1. Train (or skip if `mnist_cnn_model.h5` exists).
2. Run Flask, upload image.
3. Get digit prediction and download result.

---

## 🎯 Tips & Extensions

- **Improve model**: Try deeper CNN layers or dropout.
- **Data augmentation**: Enhance generalization with flips/rotations.
- **Add evaluation**: Display confusion matrix & classification report.
- **Add live frontend**: e.g., use FastAPI + React for real-time predictions.

---

## 🧾 Credits & References

Inspired by public digit recognition repos using CNN + Flask. Adapted standard MNIST CNN pipelines (Conv2D + pooling + dense layers).

---

## 🤝 Contributing & License

Feel free to submit issues or PRs for enhancements: architecture, UI, deployment.  
Include license details here (e.g. MIT, Apache) if you wish.
