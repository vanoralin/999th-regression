# ♻️ Trash Classification & Price Analysis System

This project is a web application that uses **Machine Learning and Data Analysis**  
to **classify trash types from images** and **analyze price trends of recyclable materials**.  
The application is built with **FastAPI** and provides results through a web interface.

---

## 🔍 Features

- 📸 Upload an image to classify trash type
- 🧠 Image classification using Deep Learning (CNN)
- 📊 Analysis of historical trash price data
- 📈 Visualization: Time Series, Correlation Matrix, and Regression graphs
- 🔮 Price prediction based on historical trends
- 🌐 Web-based interface using Jinja2 templates

---

## 🛠️ Technologies Used

- **Backend**: FastAPI, Uvicorn  
- **Machine Learning**: TensorFlow / Keras  
- **Image Processing**: OpenCV  
- **Data Analysis**: Pandas, NumPy, Scikit-learn, SciPy  
- **Visualization**: Matplotlib, Seaborn  
- **Frontend**: HTML (Jinja2 Templates)

---

## 🚀 How to Run This Project

**1.Install dependencies**
```bash
pip install -r requirements.txt
```
**2️. Run the application**
``` bash
python -m uvicorn main:app
```

## 🌍 Access the Application

After running the server, open your browser and go to:

http://127.0.0.1:8000

## 📊 Output

Predicted trash type

Prediction confidence (similarity score)

Price trend graphs (time series and regression)

Correlation analysis between different trash types

Predicted future price

Textual explanation based on data analysis