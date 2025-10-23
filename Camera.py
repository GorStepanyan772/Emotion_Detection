import pickle
import cv2
import numpy as np
labels = ['angry', 'disgusted','fearful','happy','neutral','sad','suprised']
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break
    # копия кадра для обработки
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image.reshape(1, 48, 48, 1)
    y_pred = model.predict(image)
    emotion = labels[np.argmax(y_pred)]
    prob = np.max(y_pred)  # вероятность предсказания
    # рисуем предсказание на кадре
    text = f"{emotion} ({prob:.2f})"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()