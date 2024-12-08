import cv2
import mediapipe as mp

# Mediapipe bileşenleri
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Başparmak Yukarı Algılama Fonksiyonu
def is_thumbs_up(hand_landmarks, frame_height, frame_width):
    # Landmark pozisyonlarını piksel koordinatlarına çevir
    thumb_tip_y = int(hand_landmarks.landmark[4].y * frame_height) #başparmağı algılıyor  Başparmağın ucunu L4
    thumb_tip_x = int(hand_landmarks.landmark[4].x * frame_width)
    thumb_base_y = int(hand_landmarks.landmark[2].y * frame_height) #ID 2: Başparmağın orta eklem noktası.

    # İşaret parmağı ucu (Landmark 8)
    index_tip_y = int(hand_landmarks.landmark[8].y * frame_height)

    # Kontrol: Başparmak yukarı mı?
    if thumb_tip_y < thumb_base_y and thumb_tip_y < index_tip_y:
        return True
    return False

# Mediapipe Hands çözümü
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read() #kameranın açılması 
    if not ret:
        break

    # OpenCV ve Mediapipe entegrasyonu
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Kullanıcı Arayüzü: Tıklanabilir Düğme
    cv2.rectangle(frame, (100, 100), (300, 200), (255, 0, 0), -1)
    cv2.putText(frame, "Click Me", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Başparmak Yukarı Algılama
            if is_thumbs_up(hand_landmarks, frame.shape[0], frame.shape[1]):
                cv2.putText(frame, "Thumbs Up Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Landmark bilgilerini al ve konsola yazdır
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Eğer başparmak düğme alanındaysa
                if 100 < cx < 300 and 100 < cy < 200:
                    cv2.putText(frame, "Button Clicked!", (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
