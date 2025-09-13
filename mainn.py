import cv2
import numpy as np

from tensorflow.lite.python.interpreter import Interpreter

# ===== Load model TFLite =====
interpreter = Interpreter(model_path="./models/model recognite/facenet_int_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input:", input_details)
print("Output:", output_details)


# ===== Tiền xử lý ảnh =====
def preprocess_face(img_path, target_size=(160,160)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0  # scale [0,1]
    img = np.expand_dims(img, axis=0)     # (1,160,160,3)
    return img


# ===== Extract embedding =====
def get_embedding(img_path):
    img = preprocess_face(img_path)
    # đảm bảo dtype khớp với model
    if img.dtype != input_details[0]['dtype']:
        img = img.astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]['index'])[0]  # (128,)

    # L2 normalize
    emb = emb / np.linalg.norm(emb)
    return emb


# ===== Demo =====
if __name__ == "__main__":
    emb1 = get_embedding("./images/make.png")
    emb2 = get_embedding("./images/moc.png")

    # cosine similarity
    cos_sim = np.dot(emb1, emb2)
    print("Cosine similarity:", cos_sim)
