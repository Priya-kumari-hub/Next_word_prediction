🔹 Overview
This project is a Next Word Prediction Model built using TensorFlow & Keras. The model is trained on a text dataset to predict the most probable next word given an input sequence.

🔹 Dataset
The model is trained on a text dataset stored in pizza.txt, which contains multiple sentences related to various topics. A tokenizer is used to preprocess the text by converting words into numerical representations.

🔹 Model Architecture
The model is implemented using LSTM (Long Short-Term Memory) layers to capture sequential dependencies in text. The architecture includes:

Embedding Layer: Converts words into dense vectors.
LSTM Layers: Captures long-term dependencies in sequences.
Dropout Layer: Prevents overfitting.
Dense Layer with Softmax Activation: Outputs the next word prediction probabilities.

🔹 Training Details
Epochs: 500
Optimizer: Adam
Loss Function: Categorical Crossentropy
Final Accuracy: 97.74%
Epoch 496/500
51/51 [==============================] - 1s 20ms/step - loss: 0.0679 - accuracy: 0.9668
Epoch 497/500
51/51 [==============================] - 1s 20ms/step - loss: 0.0571 - accuracy: 0.9687
Epoch 498/500
51/51 [==============================] - 1s 24ms/step - loss: 0.0545 - accuracy: 0.9687
Epoch 499/500
51/51 [==============================] - 1s 20ms/step - loss: 0.0510 - accuracy: 0.9681
Epoch 500/500
51/51 [==============================] - 1s 21ms/step - loss: 0.0473 - accuracy: 0.9705

🔹 Model Performance 🎯
During training, the model achieved:
✅ Accuracy: 97.74%
✅ Loss: 0.0496

Example Predictions:
Input: "Technology"
Output: "Technology will play a significant role in shaping the future of pizza making lies and delivery"

Input: "India"
Output: "India is much more than a delicious dish—it is a culinary phenomenon that has captured the hearts and palates of people around the world"


🔹 Repository Structure

📂 Next_word_prediction
 ┣ 📜 next_word_model.keras   # Trained model file
 ┣ 📜 tokenizer.pkl           # Tokenizer file
 ┣ 📜 pizza.txt               # Training dataset
 ┣ 📜 README.md               # Project Documentation
 ┗ 📜 next_word_prediction_code.py # Prediction script
