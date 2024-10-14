import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = {
    'Square_Footage': np.random.randint(500, 5000, 50),
    'Num_Bedrooms': np.random.randint(1, 5, 50),
    'Num_Bathrooms': np.random.randint(1, 4, 50),
    'Year_Built': np.random.randint(1950, 2023, 50),
    'Lot_Size': np.random.uniform(0.1, 5.0, 50),
    'Garage_Size': np.random.randint(0, 3, 50),
    'Neighborhood_Quality': np.random.randint(1, 10, 50),
    'House_Price': np.random.randint(100000, 500000, 50)
}

df = pd.DataFrame(data)
X = df.drop(columns=['House_Price'])
y = df['House_Price']

# Membuat dan melatih model regresi linear
model = LinearRegression()
model.fit(X, y)

# Judul Aplikasi
st.title("Aplikasi Prediksi Harga Rumah")

# Input Data Baru
st.header("Masukkan Data Rumah Baru:")
square_footage = st.number_input("Square Footage", min_value=500, max_value=5000, step=1)
num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, step=1)
year_built = st.number_input("Year Built", min_value=1900, max_value=2023, step=1)
lot_size = st.number_input("Lot Size (acres)", min_value=0.1, max_value=10.0, step=0.1)
garage_size = st.number_input("Garage Size", min_value=0, max_value=5, step=1)
neighborhood_quality = st.number_input("Neighborhood Quality", min_value=1, max_value=10, step=1)

# Prediksi Harga Rumah
if st.button("Prediksi Harga"):
    input_data = pd.DataFrame({
        'Square_Footage': [square_footage],
        'Num_Bedrooms': [num_bedrooms],
        'Num_Bathrooms': [num_bathrooms],
        'Year_Built': [year_built],
        'Lot_Size': [lot_size],
        'Garage_Size': [garage_size],
        'Neighborhood_Quality': [neighborhood_quality]
    })
    
    prediction = model.predict(input_data)
    st.success(f"Prediksi Harga Rumah: ${prediction[0]:,.2f}")
