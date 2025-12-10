import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# -------------------------------
# Settings
# -------------------------------
model_path = "save_at_1.keras"  # path to your trained model
image_size = (180, 180)  


print(model_path)                    # must match the training size


model = keras.models.load_model("save_at_20.keras")
print(f"Loaded model from {model_path}")
# -------------------------------
# Function to predict a single image
# -------------------------------
def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load image
    img = keras.utils.load_img(img_path, target_size=image_size)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # Convert to array and add batch dimension
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    pred_class_index = np.argmax(predictions[0])
    class_names = model.class_names if hasattr(model, "class_names") else ["Class_" + str(i) for i in range(predictions.shape[1])]
    print("Class 0: Cat Class 1: Dog Class 2: People")
    print(f"Predicted class: {class_names[pred_class_index]}")
    print(f"Class probabilities: {predictions[0]}")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    test_image_path = "testImages/29.jpg"  # replace with your image
    predict_image(test_image_path)
