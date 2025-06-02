
# Hướng dẫn tải dữ liệu, huấn luyện và suy luận mô hình CPVS

## 1. Tải dữ liệu

Dữ liệu sử dụng cho dự án chưa được đính kèm trong repo. Bạn cần:

- Tải bộ dữ liệu theo hướng dẫn cụ thể từ nguồn phù hợp.
- Đặt dữ liệu vào thư mục có cấu trúc như sau:

```
CPVS/
├── data/
│   ├── train/
│   ├── test/
│   └── validation/
└── computer-vision.ipynb
```

## 2. Huấn luyện (Training)

Mở file `computer-vision.ipynb`:

- Chạy từng cell theo thứ tự:
  - Load và tiền xử lý dữ liệu (đọc ảnh, resize, chuẩn hóa).
  - Xây dựng mô hình (CNN, ResNet, VGG,...).
  - Huấn luyện mô hình trên tập huấn luyện (`train`) và xác thực với tập kiểm chứng (`validation`).
- Lưu mô hình sau khi huấn luyện:

```python
model.save('model_shape_classification.h5')
```

## 3. Suy luận (Inference)

Sau khi đã có mô hình huấn luyện:

- Load mô hình đã lưu:

```python
from tensorflow.keras.models import load_model
model = load_model('model_shape_classification.h5')
```

- Tiền xử lý hình ảnh cần phân loại:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('path_to_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0
```

- Thực hiện dự đoán:

```python
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)
print("Predicted class:", predicted_class)
```

Kết quả trả về sẽ là nhãn của hình ảnh được dự đoán.

---


