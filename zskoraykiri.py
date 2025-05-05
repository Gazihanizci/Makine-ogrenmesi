import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Excel dosyasını oku
df = pd.read_excel('dataset.xlsx')

# MinMaxScaler ile ölçekleme işlemi
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Normalleştirilmiş veriyi yeni bir Excel dosyasına kaydet
df_scaled.to_excel('dataset_normalized.xlsx', index=False)

print("Veri Min-Max ölçeklendirilip kaydedildi.")
