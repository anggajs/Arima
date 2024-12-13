import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from streamlit_option_menu import option_menu
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# Path ke file CSV yang sudah ada
wisata_file_path = os.path.join('data', 'Data_Wisata.csv')

# Navigasi sidebar
with st.sidebar:
    selected = option_menu('Menu',
                           ['Home',
                            'Data',
                            'Perhitungan ARIMA'],
                           icons=['house', 'bar-chart-fill', 'plus-slash-minus'],
                           menu_icon='cast',
                           default_index=0)
st.sidebar.markdown(
    "<div style='text-align: center; font-weight: bold;'>2024 @ by Forcasting Arima Kelompok19</div>",
    unsafe_allow_html=True
)

if selected == 'Home':
    st.title('Peramalan Jumlah Perjalanan Wisatawan Nusantara ke Kota Surabaya dengan Metode ARIMA')
    st.write('Peramalan jumlah perjalanan wisatawan Nusantara ke Kota Surabaya dengan metode ARIMA (AutoRegressive Integrated Moving Average) adalah teknik statistika yang digunakan untuk memprediksi atau meramalkan nilai suatu variabel di masa depan berdasarkan data historis yang tersedia. Dalam hal ini, variabel yang diprediksi adalah jumlah perjalanan wisatawan Nusantara yang berkunjung ke Kota Surabaya, dan data historis yang digunakan adalah jumlah perjalanan wisatawan pada periode waktu tertentu.')

# Halaman Data
if selected == 'Data':
    st.title('Jumlah Perjalanan Wisatawan Nusantara ke Kota Surabaya')

    # Membaca data dari file lokal
    try:
        data = pd.read_csv(wisata_file_path)

        # Fitur untuk menambahkan data baru
        st.subheader("Tambah Data Baru")

        # Form input untuk data baru
        with st.form(key='form_add_data'):
            new_time = st.text_input("Masukkan Waktu (format: YYYY-MM)", value="")
            new_value = st.text_input("Masukkan Jumlah Wisatawan (gunakan titik untuk pemisah ribuan)", value="")
            submit_button = st.form_submit_button(label='Tambah Data')

        # Proses penambahan data baru
        if submit_button:
            try:
                # Validasi format waktu menggunakan regex
                if not new_time or not pd.to_datetime(new_time, format='%Y-%m', errors='coerce'):
                    st.error("Format waktu salah. Pastikan formatnya adalah YYYY-MM, misalnya 2024-03.")
                else:
                    # Mengubah new_value dengan titik pemisah ribuan menjadi angka
                    if new_value:
                        new_value = new_value.replace('.', '')  # Hapus titik (pemisan ribuan)
                        new_value = float(new_value)  # Ubah ke float

                    new_time_parsed = pd.to_datetime(new_time, format='%Y-%m').strftime('%Y-%m')

                    # Tambahkan data ke DataFrame
                    new_data = pd.DataFrame({'Bulan': [new_time_parsed], 'Jumlah Wisatawan': [new_value]})
                    data = pd.concat([data, new_data], ignore_index=True)

                    # Simpan kembali ke file CSV
                    data.to_csv(wisata_file_path, index=False)
                    st.success("Data berhasil ditambahkan dan disimpan!")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

        # Fitur untuk menghapus data berdasarkan indeks
        st.subheader("Hapus Data")

        # Input indeks data yang ingin dihapus
        with st.form(key='form_delete_data'):
            index_to_delete = st.number_input("Masukkan Indeks Data yang Ingin Dihapus", min_value=0, step=1, value=0)
            delete_button = st.form_submit_button(label='Hapus Data')

        # Proses penghapusan data
        if delete_button:
            try:
                if 0 <= index_to_delete < len(data):
                    data = data.drop(index_to_delete).reset_index(drop=True)
                    data.to_csv(wisata_file_path, index=False)
                    st.success("Data berhasil dihapus dan disimpan!")
                else:
                    st.error("Indeks yang dimasukkan tidak valid.")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

        # Tampilkan data terbaru
        st.write("Data terbaru:")
        st.write(data)

    except FileNotFoundError:
        st.error("File tidak ditemukan. Pastikan file CSV berada di lokasi yang benar.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")

# Halaman Perhitungan ARIMA
if selected == 'Perhitungan ARIMA':
    st.title('Perhitungan ARIMA di Jawa Timur')

    # Membaca data CSV lokal
    try:
        data = pd.read_csv(wisata_file_path)

        # Menampilkan beberapa baris pertama untuk memeriksa data
        st.subheader("Data Jumlah Wisatawan Ke Kota Surabaya")
        st.write(data)

        # Memastikan kolom waktu diubah menjadi format bulanan
        if 'Bulan' in data.columns:
            data['Bulan'] = pd.to_datetime(data['Bulan'], errors='coerce').dt.to_period('M').dt.to_timestamp()
            data = data.dropna(subset=['Bulan'])
        else:
            st.error("Kolom waktu ('Bulan') tidak ditemukan dalam data.")

        # Menyusun data untuk ARIMA
        data.set_index('Bulan', inplace=True)

        # Memilih kolom yang relevan untuk model ARIMA
        st.subheader("Pilih Kolom Data untuk Peramalan")
        value_column = st.selectbox("Pilih Kolom Data", data.columns)
        data = data[[value_column]]

        # Memastikan kolom yang dipilih memiliki cukup data
        if data.shape[0] < 2:
            st.error("Kolom yang dipilih harus memiliki lebih dari satu data point untuk peramalan.")
        else:
            st.success("Data siap untuk peramalan.")

            # Memilih parameter ARIMA
            st.subheader("Pilih Parameter ARIMA (p, d, q)")
            p = st.slider('p (order AR)', 0, 6, 1)
            d = st.slider('d (order differencing)', 0, 5, 1)
            q = st.slider('q (order MA)', 0, 10, 1)

            # Input jumlah langkah peramalan
            forecast_steps = st.number_input("Jumlah Langkah Peramalan", min_value=0, max_value=100, value=0 , step=1)

            # Latih model ARIMA
            if st.button("Hitung Model ARIMA"):
                try:
                    # Latih model ARIMA dengan parameter yang dipilih
                    model = ARIMA(data[value_column].values, order=(p, d, q))
                    model_fit = model.fit()

                    # Menampilkan ringkasan model
                    st.subheader("Ringkasan Model ARIMA")
                    st.write(model_fit.summary())

                    # Peramalan
                    forecast = model_fit.forecast(steps=forecast_steps)

                    # Menampilkan hasil peramalan
                    st.subheader('Peramalan')
                    last_period = data.index[-1].to_period('M') + 1  # Menambahkan satu bulan
                    forecast_index = [last_period + i for i in range(forecast_steps)]
                    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=["Jumlah Wisata"])

                    st.write(forecast_df)
                    
                    # Hitung MAPE dan nilai error lainnya
                    st.subheader("Evaluasi Model: Error Metrics")
                    actual_length = min(len(data[value_column]), len(forecast))
                    actual = data[value_column].values[-actual_length:]
                    predicted = forecast[:actual_length]

                    if actual_length > 0:
                        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                        mae = np.mean(np.abs(actual - predicted))
                        mse = np.mean((actual - predicted) ** 2)
                        rmse = np.sqrt(mse)

                        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                    else:
                        st.write("Tidak cukup data untuk menghitung metrik error.")

                    # Plotting hasil peramalan
                    st.subheader("Grafik Peramalan")
                    plt.figure(figsize=(10, 6))
                    plt.plot(data, label="Data Asli")
                    plt.plot(forecast_df, label="Peramalan", color='red')
                    plt.legend(loc="best")
                    plt.title("Peramalan Waktu Menggunakan ARIMA")
                    st.pyplot(plt)

                    

                    # Plot ACF dan PACF dari residual
                    st.subheader("Analisis Residual: Autokorelasi (ACF) dan Partial Autokorelasi (PACF)")

                    # Residuals
                    residuals = model_fit.resid

                    # Plot ACF dari residual
                    st.subheader("Plot ACF (Residual)")
                    fig_acf_residual, ax_acf_residual = plt.subplots(figsize=(10, 6))
                    plot_acf(residuals, ax=ax_acf_residual)
                    st.pyplot(fig_acf_residual)

                    # Plot PACF dari residual
                    st.subheader("Plot PACF (Residual)")
                    fig_pacf_residual, ax_pacf_residual = plt.subplots(figsize=(10, 6))
                    plot_pacf(residuals, ax=ax_pacf_residual, method='ywm')
                    st.pyplot(fig_pacf_residual)

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")

    except FileNotFoundError:
        st.error("File tidak ditemukan. Pastikan file CSV berada di lokasi yang benar.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")
