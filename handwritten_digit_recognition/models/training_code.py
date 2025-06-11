!pip install tensorflow
!pip install opencv-python

def load_emnist_csv(train_path, test_path):
    # Read CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Extract labels (first column) and pixel values (remaining columns)
    y_train = train_df.iloc[:, 0].values
    x_train = train_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values
    x_test = test_df.iloc[:, 1:].values
    
    # Reshape pixel values to 28x28 images
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)

# (x_train, y_train), (x_test, y_test) = load_emnist_csv("emnist-digits-train.csv", "emnist-digits-test.csv")



mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test,y_test)=mnist.load_data()# split the data in training set as tuple

x_train = tf.keras.utils.normalize(x_train , axis = 1)
x_test = tf.keras.utils.normalize(x_test , axis = 1)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # Make sure input has a channel dimension
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2)
model.save("handwritten_digit_recognition.keras")


loss, accuracy = model.evaluate(x_test, y_test)

print("loss", loss)
print("accuracy: ", accuracy)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

x_test_reshaped = x_test.reshape(-1, 28, 28, 1)

# Predict the classes
y_pred_probs = model.predict(x_test_reshaped)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


import pandas as pd
import tensorflow as tf
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
model = tf.keras.models.load_model("handwritten_digit_recognition.keras")

for x in range(1,9):

    img=cv.imread(f'{x}.png')[:,:,0]
    img = cv.resize(img, (28, 28))
    img=np.invert(np.array([img]))
    prediction=model.predict(img)
    print("----------------")
    print("The predicted value is : ",np.argmax(prediction))
    print("----------------")
    plt.imshow(img[0],cmap=plt.cm.binary)#change the color in black and white
    plt.show()
