import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Excel dosyasını oku
file_path = "eksikdataset.xlsx"
df = pd.read_excel(file_path)

# Age sütunu hariç diğer sütunları seç
columns_to_normalize = df.columns.difference(['Age'])

# Min-Max Normalizasyon uygulama
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Normalizasyon sonrası veriyi yeni bir dosyaya kaydet
output_file = "dataset_normalizasyon.xlsx"
df.to_excel(output_file, index=False)

print(f"Min-Max normalizasyon tamamlandı. Sonuç {output_file} olarak kaydedildi.")
