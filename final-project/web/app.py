from flask import Flask, request, render_template
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from scipy import stats

app = Flask(__name__)

# Model ve scaler'ı yükle
model = load('ridge_regression_model.joblib')
scaler = load('scaler.joblib')

# Training'de kullanılan kolonları yükle
with open('training_columns.txt', 'r') as file:
    training_columns = file.read().splitlines()

@app.route('/')
def home():
    return render_template('index.html')

# Tahmin fonksiyonu
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            # Dosyayı oku
            if file.filename.endswith('.csv'):
                data = pd.read_csv(file)
            else:
                data = pd.read_excel(file)
            print(data)
            # Veriyi işle
            data = preprocess_data(data)
            print(data)
            data = align_columns(data, training_columns)

            # Veriyi ölçeklendir
            input_data_scaled = scaler.transform(data)
            print("input_data_scaled")
            print(input_data_scaled)
            # Tahmin yap
            prediction = model.predict(input_data_scaled)
            
            prediction_list = ['Predicted Price: ${:.2f}]'.format(abs(pred)) for pred in prediction]

            # Sonucu döndür
            return render_template('index.html', prediction_text=prediction_list)
    return render_template('index.html', prediction_text='Invalid file format. Please upload a CSV or XLSX file.')

# Veriyi ön işleme fonksiyonu
def preprocess_data(data):

    data = data.drop(columns=['Heating','Street','Utilities','Condition2', 'RoofMatl', 'Id', 'Neighborhood'])

    # Yeni özellikler oluşturma
    data['TotalLivArea'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'] 
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['RemodAge'] = data['YrSold'] - data['YearRemodAdd']
    data['TotalBath'] = data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])
    data['TotalRooms'] = data['TotRmsAbvGrd'] + data['BedroomAbvGr']
    data['GarageAge'] = data['YrSold'] - data['GarageYrBlt']
    data['TotalPorchSF'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']
    data['LivAreaPerRoom'] = data['TotalLivArea'] / data['TotalRooms']
    data['GarageCarsPerArea'] = data['GarageArea'] / data['GarageCars']
    data['TotalConstructionArea'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'] + data['GarageArea']

    def get_season(month):
        if month in [12, 1, 2]:
            return 1  # Winter
        elif month in [3, 4, 5]:
            return 2  # Spring
        elif month in [6, 7, 8]:
            return 3  # Summer
        else:
            return 4  # Fall

    data['MonthlySeasonalIndex'] = data['MoSold'].apply(get_season)

    def get_age_category(age):
        if age < 10:
            return 'New'
        elif age < 50:
            return 'MidAge'
        else:
            return 'Old'

    data['AgeCategory'] = data['HouseAge'].apply(get_age_category)
    
    # Kategorik değişkenlerin encoding'i
    categorical_features, numerical_features, cat_but_car = grab_col_names(data)
    data = pd.get_dummies(data, columns=[col for col in categorical_features if col])
    print("data")
    print(data)
    return data

# Prediction data'yı training data'nın kolonlarıyla hizalama fonksiyonu
def align_columns(data, training_columns):
    for col in training_columns:
        if col not in data.columns:
            data[col] = 0
    
    data = data[training_columns]
    
    return data

# Column isimlerini alma fonksiyonu
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtype != 'O']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtype == 'O']

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

if __name__ == '__main__':
    app.run(debug=True)
