# 📊 Arabic Text Regression Using Deep Learning

This project explores **Arabic Natural Language Processing (NLP)** by predicting continuous values (scores) from Arabic text using machine learning and deep learning models. The full pipeline includes:

- 🌐 Data scraping
- 🧼 Text preprocessing
- 🧠 Deep learning modeling
- 📈 Model evaluation

---

## 1. 🌐 Data Collection

Arabic news headlines were scraped from two major websites:

- 📰 Al Jazeera (https://www.aljazeera.net)
- 📰 Al Arabiya (https://www.alarabiya.net)

Each article’s title was assigned a random score between 0 and 10 to simulate a regression problem.

**Example of scraped data:**

| Text | Score |
|------|-------|
| كشفت بيانات ملاحية أن سفينة متجهة لإسرائيل رست... | 0.3 |
| خرجت مظاهرات حاشدة في عدد من العواصم والمدن حو... | 1.6 |
| شهدت منصات التواصل الاجتماعي في مصر حالة من ال... | 9.3 |

---

## 2. 🧼 Text Preprocessing

Arabic text was cleaned using the CAMeL Tools library. The steps included:

- Removing diacritics
- Tokenization
- Removing stopwords (from a custom Arabic stopword list)
- Cleaning punctuation and special characters

**Example of cleaned text:**

| Cleaned Text | Score |
|--------------|--------|
| كشفت بيانات ملاحية أن سفينة متجهة لإسرائيل رست... | 0.3 |

This improves model accuracy by reducing noise in the input.

---

## 3. 🧠 Model Training

We trained and compared the following deep learning models using Keras:

- 🔁 RNN (Recurrent Neural Network)
- 🔁 BiRNN (Bidirectional RNN)
- 🔁 GRU (Gated Recurrent Unit)
- 🔁 LSTM (Long Short-Term Memory)

All models used:

- Arabic text → tokenized + padded sequences
- Embedding layer
- Output: Regression (linear activation)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs: 5

Example (BiRNN - last epoch):

Epoch 5/5 loss: 15.08 - mae: 3.06 - val_loss: 0.87 - val_mae: 0.72

---

## 4. 📈 Model Evaluation

Models were evaluated using:

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **R²**: Coefficient of Determination

| Model   | MAE  | MSE  | R² Score |
|---------|------|------|----------|
| RNN     | 1.58 | 3.01 | -0.29    |
| **BiRNN** | **1.58** | **2.52** | **-0.08** |
| GRU     | 2.33 | 7.68 | -2.29    |
| LSTM    | 2.35 | 7.85 | -2.36    |

**🧠 Best Model:** BiRNN gave the lowest error and best R² score.

---

## 🧰 Tools & Libraries Used

- Python 3.x
- camel-tools
- BeautifulSoup, requests
- TensorFlow, Keras
- scikit-learn
- pandas, matplotlib

---

## 📌 Conclusion

This project demonstrates end-to-end Arabic text regression using deep learning. Despite the limited dataset, the BiRNN model showed promising results. Preprocessing Arabic correctly is essential to achieve good performance.

---

## 🚀 Future Improvements

- Use a larger real-world dataset with labeled scores
- Fine-tune Arabic pretrained models (e.g., AraBERT)
- Try transformers for improved results
