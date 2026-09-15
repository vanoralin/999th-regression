import base64
import uvicorn
import shutil
import os
import tensorflow as tf
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fastapi import FastAPI, UploadFile, File, responses, templating, staticfiles, Request
from sklearn import metrics, linear_model, preprocessing
from scipy import spatial

# uvicorn main:app

app = FastAPI()

templates = templating.Jinja2Templates(directory="template")
app.mount("/template", staticfiles.StaticFiles(directory="template"), name="template")
app.mount("/image", staticfiles.StaticFiles(directory="image"), name="image")
app.mount("/plots", staticfiles.StaticFiles(directory="plots"), name="plots")

def detect_img(img, trash_type_lst):
    model_1 = tf.keras.models.load_model(os.path.join('model', 'trash_classifier_1.h5'))
    model_2 = tf.keras.models.load_model(os.path.join('model', 'trash_classifier_2.h5'))
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resize = tf.image.resize(img, (256,256))
    plt.imshow(resize.numpy().astype(int))
    
    yhat_1 = model_1.predict(np.expand_dims(resize/255, 0))
    yhat_2 = model_2.predict(np.expand_dims(resize/255, 0))

    similarity = '{:.4f}'.format(max(np.max(yhat_1), np.max(yhat_2))*100)
    similarity_2 = '{:.4f}'.format(min(np.max(yhat_1), np.max(yhat_2))*100)
    
    print("model_1",np.max(yhat_1), trash_type_lst[np.argmax(yhat_1, axis=-1)[0]])
    print("model_2",np.max(yhat_2), trash_type_lst[np.argmax(yhat_2, axis=-1)[0]])
    
    if np.max(yhat_1) >= np.max(yhat_2):
        predicted_img_class = trash_type_lst[np.argmax(yhat_1, axis=-1)[0]]
        predicted_img_class_2 = trash_type_lst[np.argmax(yhat_2, axis=-1)[0]]
    else:
        predicted_img_class = trash_type_lst[np.argmax(yhat_2, axis=-1)[0]]
        predicted_img_class_2 = trash_type_lst[np.argmax(yhat_1, axis=-1)[0]]
    
    return {"predicted_img_class": predicted_img_class, "similarity": similarity, "predicted_img_class_2": predicted_img_class_2, "similarity_2": similarity_2}

def save_plot_to_file(plot_func, filename, x=None, y=None):
    img_path = os.path.join("plots", filename)
    plt.figure()
    if x == None and y == None:
        plot_func()
    if y != None:
        plot_func(x, y) 
    else:
        plot_func(x) 
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()  
    return img_path

def find_best_degree(X, y, max_degree=5):
    mse_values = []

    for degree in range(1, max_degree + 1):
        poly = preprocessing.PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        model = linear_model.LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        mse = metrics.mean_squared_error(y, y_pred)
        mse_values.append(mse)

    best_degree = int(np.argmin(mse_values)) + 1
    return best_degree

@app.get("/", response_class=responses.HTMLResponse)
async def get_image(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="main_page.html",
        context={"request": request},
    )

@app.post("/upload", response_class=responses.HTMLResponse)
async def show_result(request: Request, image: UploadFile = File(None)):
    
    trash_type = ['Glass', 'Steel', 'Paper', 'Plastic']
    
    if image and image.filename:
        file_location = os.path.join("image", os.path.basename(image.filename))
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        with open(file_location, "rb") as uploaded_file:
            encoded_image = base64.b64encode(uploaded_file.read()).decode("ascii")
        result_img = f"data:{image.content_type or 'application/octet-stream'};base64,{encoded_image}"

        try:
            img = cv2.imread(file_location)
            img_result = detect_img(img, trash_type)
            predicted_img_class = img_result["predicted_img_class"]
            similarity = img_result["similarity"]
            predicted_img_class_2 = img_result["predicted_img_class_2"]
            similarity_2 = img_result["similarity_2"]
        finally:
            os.remove(file_location)
    else: 
        form_data = await request.form()
        result_img = form_data.get('result_img')
        predicted_img_class = form_data.get('predicted_img_class_2')
        predicted_img_class_2 = form_data.get('predicted_img_class')
        similarity = form_data.get('similarity_2')
        similarity_2 = form_data.get('similarity')

    df = pd.read_csv('./content/linear_trash_price.csv', skiprows=1) 
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)

    data = {
        "Steel": df["Steel"].tolist(),
        "Paper": df["Paper"].tolist(),
        "Glass": df["Glass"].tolist(),
        "Plastic": df["Plastic"].tolist(),
    }

    dates = df.index.tolist()
    data_array = np.array([data["Steel"], data["Paper"], data["Glass"], data["Plastic"]])

    if not isinstance(predicted_img_class, str) or predicted_img_class not in data:
        raise ValueError("Invalid predicted trash type")
    x_axis = predicted_img_class
    
    #-----------------------------------------------------------------------------------------------------------------------
    #correlation matrix 
    def plot_correlation_matrix(x_axis):
        corr_matrix = np.corrcoef(data_array)
        trash_type = ['Glass', 'Steel', 'Paper', 'Plastic']
        index_x = trash_type.index(x_axis)
        new_corr_matrix = corr_matrix[index_x, :]

        plt.figure(figsize=(8, 2))
        sns.heatmap(new_corr_matrix.reshape(1, -1), annot=True, xticklabels=trash_type, yticklabels=[x_axis], cmap='Pastel1')

        plt.title(f'Correlation Matrix of {x_axis} Prices')
        corr_array = np.delete(new_corr_matrix, index_x)
        corr_array = corr_array.reshape(-1)
    
        return corr_array
        
    corr_matrix_img_path = save_plot_to_file(plot_correlation_matrix, "corr_matrix.png", x_axis)
    
    #-----------------------------------------------------------------------------------------------------------------------
    #plot time series
    def plot_time_series(x_axis):
        plt.figure(figsize=(20, 6))
        plt.plot(dates, data["Steel"], color='blue', label='Steel', marker='o')
        plt.plot(dates, data["Paper"], color='red', label='Paper', marker='o')
        plt.plot(dates, data["Glass"], color='salmon', label='Glass', marker='o')
        plt.plot(dates, data["Plastic"], color='green', label='Plastic', marker='o')
        plt.xlabel('Date')
        plt.ylabel('Prices')
        plt.title('Graph of Steel, Paper, Glass and Plastic Prices in 29/12/2564 - 01/10/2567')
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
    time_series_img_path = save_plot_to_file(plot_time_series, "time_series.png", x_axis)
    
    #-----------------------------------------------------------------------------------------------------------------------
    #polynomial regression
    def plot_regression_x_days(x_axis):
        y = np.array(data[x_axis]).reshape(-1, 1) 
        n = len(y)
        time = np.array(range(1, n + 1)).reshape(-1, 1)

        degree = find_best_degree(time, y)
        poly = preprocessing.PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(time)
        model = linear_model.LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        new_day = np.array([[n + 1]])
        new_day_poly = poly.transform(new_day)
        global predicted_price, predicted_price_2 
        predicted_price = model.predict(new_day_poly)
        predicted_price_2 = '{:.2f}'.format(predicted_price[0][0])
        
        global r_squared
        r_squared = metrics.r2_score(y, y_pred)
        
        plt.figure(figsize=(20, 6))
        plt.scatter(time, y, color='tab:gray', label='Actual Data', zorder=3)  
        plt.scatter(time, y_pred, color='red', label='Predicted Data', alpha=0.5, zorder=3)  
        plt.scatter(n + 1, predicted_price[0][0], color='purple', s=80, label='Predicted Price Point', zorder=4)
        plt.axhline(y=predicted_price[0][0], color='purple', linestyle='-.', zorder=1)  
        plt.axvline(x=n+1, color='purple', linestyle='-.', zorder=1)
        plt.plot(time, y, color='tab:gray', linewidth=0.5, zorder=2)
        plt.plot(time, y_pred, color='orange', zorder=2)
        plt.xticks(ticks=np.arange(1, n + 2), labels=dates + ["Predict"], rotation=90)
        plt.ylabel(f'{x_axis} Prices')
        plt.xlabel('Time')
        plt.title(f'Polynomial Regression of {x_axis} Prices')
        plt.legend()
        plt.grid(True, zorder=0)  
        plt.tight_layout()
    
    regression_x_img_path = save_plot_to_file(plot_regression_x_days, "regression_x_plot.png", x_axis)
    regression_x_drcrp = []
    
    if predicted_img_class == "Steel":
        regression_x_drcrp.append(f"\
            จากสถิติข้อมูลในช่วงวันที่ 29 ธันวาคม 2564 - วันที่ 1 ตุลาคม 2567 ข้อมูลของ Steel มีราคารับซื้อมีแนวโน้มลดลงมากกว่าในช่วง 2 ปีที่ผ่านมา \
            และมีแนวโน้มลดลงอย่างต่อเนื่อง โดยคาดว่าในช่วง 15  วันนี้อาจจะลดลงถึง {predicted_price_2} บาท ซึ่งจากการคำนวณมีความแม่นยำ \
            {'{:.2f}'.format(r_squared*100)}%")
        regression_x_drcrp.append("แนะนำให้เก็บไว้ก่อน เดี๋ยวค่อยขาย!")
    
    elif predicted_img_class == "Plastic":
        regression_x_drcrp.append(f"\
            จากสถิติข้อมูลในช่วงวันที่ 29 ธันวาคม 2564 - วันที่ 1 ตุลาคม 2567 ข้อมูลของ Plastic มีราคารับซื้อมีแนวโน้มเพิ่มขึ้นมากกว่าในช่วง 2 ปีที่ผ่านมา \
            และมีแนวโน้มเพิ่มขึ้นอย่างต่อเนื่อง โดยคาดว่าในช่วง 15  วันนี้อาจจะเพิ่มขึ้นถึง {predicted_price_2} บาท ซึ่งจากการคำนวณมีความแม่นยำ \
            {'{:.2f}'.format(r_squared*100)}%")
        regression_x_drcrp.append("แนะนำให้เก็บไว้ก่อน ราคาอาจจะเพิ่มขึ้นอีก !")
        
    elif predicted_img_class == "Glass":
        regression_x_drcrp.append(f"\
            จากสถิติข้อมูลในช่วงวันที่ 29 ธันวาคม 2564 - วันที่ 1 ตุลาคม 2567 ข้อมูลของ Glass มีราคารับซื้อมีแนวโน้มลดลงมากกว่าในช่วง 2 ปีที่ผ่านมา \
            และมีแนวโน้มลดลงอย่างต่อเนื่อง โดยคาดว่าในช่วง 15  วันนี้อาจจะลดลงถึง {predicted_price_2} บาท ซึ่งจากการคำนวณมีความแม่นยำ \
            {'{:.2f}'.format(r_squared*100)}%")
        regression_x_drcrp.append("แนะนำให้ขายเลย! ก่อนจะลดมากกว่านี้นะ!")
        
    elif predicted_img_class == "Paper":
        regression_x_drcrp.append(f"\
            จากสถิติข้อมูลในช่วงวันที่ 29 ธันวาคม 2564 - วันที่ 1 ตุลาคม 2567 ข้อมูลของ Paper มีราคารับซื้อมีแนวโน้มลดลง \
            แต่จากข้อมูลที่ผ่านมาราคาของ Paper มีแนวโน้มที่จะเพิ่มสูงขึ้นเมื่อเวลาผ่านไป โดยคาดว่าในช่วง 15  วันนี้อาจจะลดลงถึง {predicted_price_2} บาท \
            ซึ่งจากการคำนวณมีความแม่นยำ {'{:.2f}'.format(r_squared*100)}%")
        regression_x_drcrp.append("แนะนำให้เก็บไว้ก่อน เดี๋ยวค่อยขาย!")
    #-----------------------------------------------------------------------------------------------------------------------
    
    #cosine_similarity
    def cosine_similarity(vec1, vec2):
        # Calculate the cosine similarity using scipy
        cosine_sim = 1 - spatial.distance.cosine(vec1, vec2)
        return cosine_sim
    #-----------------------------------------------------------------------------------------------------------------------

    def plot_regression_xy(x_axis, y_axis):
        X = np.array(data[x_axis]).reshape(-1, 1) 
        y = np.array(data[y_axis])
        
        poly = preprocessing.PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)
        model = linear_model.LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        plt.scatter(X, y, color='blue', label='Actual Data')
        plt.plot(X, y_pred, color='green', label='Fitted Polynomial Curve')
        plt.xlabel(f'{x_axis} Prices')
        plt.ylabel(f'{y_axis} Prices')
        plt.title(f'Polynomial Regression between {x_axis} and {y_axis} Prices')
        plt.legend()
        
    regression_xy_img_path = []
    regression_y_name = []
    
    for i in range(len(trash_type)):
        if trash_type[i] != x_axis:
            regression_xy_img_path.append(save_plot_to_file(plot_regression_xy, f"regression_xy_plot{i+1}.png", x_axis, trash_type[i]))
            regression_y_name.append(trash_type[i])
    
    #คำอธิบายกราฟ regression
    corr_array = plot_correlation_matrix(x_axis)
    descript_show = []
    for i in range(len(regression_y_name)):
        descript_show_i = f"จากกราฟ regression ระหว่าง {x_axis} และ {regression_y_name[i]} มีค่าสัมประสิทธิ์ความสัมพันธ์ระหว่างข้อมูล คือ {corr_array[i]:.2f} "

        if abs(corr_array[i]) == 0 :
            descript_show_i += "แสดงว่าข้อมูลทั้งสองไม่มีความสัมพันธ์กัน"
        elif corr_array[i] > 0:
            descript_show_i += "แสดงว่าข้อมูลทั้งสองมีการแปรผันตามกัน"
        elif corr_array[i] < 0:
            descript_show_i += "แสดงว่าข้อมูลทั้งสองมีการแปรผกผันกัน"

        if abs(corr_array[i]) >= 0.8 and abs(corr_array[i]) <= 1 :
            descript_show_i += "และมีความสัมพันธ์กันมาก"
        elif abs(corr_array[i]) >= 0.5 and abs(corr_array[i]) < 0.8 :
            descript_show_i += "และมีความสัมพันธ์กันปานกลาง"
        elif abs(corr_array[i]) >= 0.2 and abs(corr_array[i]) < 0.5 :
            descript_show_i += "และมีความสัมพันธ์กันต่ำ"
        elif abs(corr_array[i]) > 0 and abs(corr_array[i]) < 0.2 :
            descript_show_i += "และมีความสัมพันธ์กันต่ำมาก"
            
        cosine_i = cosine_similarity(data[x_axis],data[regression_y_name[i]])
        descript_show_i += f" โดยมีความคล้ายกันของข้อมูล {x_axis} และ {regression_y_name[i]} {cosine_i*100:.2f}%"
        descript_show.append(descript_show_i)
    
    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
        
    return templates.TemplateResponse(
        request=request,
        name="result_page.html",
        context={
            "request": request,
            "result_img": result_img,
            "predicted_img_class": predicted_img_class,
            "similarity": similarity,
            "predicted_img_class_2": predicted_img_class_2,
            "similarity_2": similarity_2,
            "corr_matrix_img": corr_matrix_img_path,
            "time_series_img": time_series_img_path,
            "regression_x_img": regression_x_img_path,
            "regression_x_drcrp": regression_x_drcrp,
            "regression_xy_lst": regression_xy_img_path,
            "regression_xy_dscrp": descript_show,
            "regression_y_name": regression_y_name,
            "predict_price": predicted_price_2,
        },
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
