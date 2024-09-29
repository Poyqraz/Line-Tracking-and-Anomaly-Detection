import sys
sys.path.append('/home/pi/Myenv/python3.9/lib/python3.9/site-packages')
sys.path.append('/home/pi/Myenv/python3.11/lib/python3.11/site-packages')

import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque
from picamera2 import Picamera2
import kablotakip3 as motor


def zoom(x,img):
    zoom_factor = x
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2
    crop_width = width // zoom_factor
    crop_height = height // zoom_factor
    x1 = center_x - crop_width // 2
    y1 = center_y - crop_height // 2
    x2 = center_x + crop_width // 2
    y2 = center_y + crop_height // 2
    cropped_img = img[y1:y2, x1:x2]
    img = cv2.resize(cropped_img, (width, height))
    return img

class UnderwaterShapeDetector:
    def __init__(self):
        self.shapes = ['UCGEN', 'KARE', 'DIKDORTGEN', 'BESGEN', 'ALTIGEN', 'YILDIZ', 'DAIRE'
                       ]
        self.detected_shapes = []
        self.start_time = None
        self.following_cable = True
        self.yellow_object_detected = False
        self.cable_direction = None
        self.last_cable_position = None
        self.initial_direction = "Duz git"
        self.previous_directions = []
        self.turn_detected = False
        self.turn_direction = None
        self.cable_positions = deque(maxlen=20)  # Son 20 pozisyonu saklayacağız
        self.current_direction = "Duz git"
        self.last_turn_time = time.time()
        self.vertical_movement = False
        self.direction_change_threshold = 20  # piksel cinsinden
        self.last_direction_change = time.time()
        self.direction_change_cooldown = 1  # saniye cinsinden
        self.initial_cable_center = None
        self.screen_center = None       # Ekranın merkezini tutacak
        self.last_sharp_turn_time = 0  # saniye cinsinden
        self.sharp_turn_cooldown = 3  # 3 saniye
        self.alignment_threshold = 20  # piksel cinsinden
        self.cable_positions = deque(maxlen=30)  # Son 30 pozisyonu saklayacağız
        self.alignment_threshold = 10  # piksel cinsinden
        self.turn_prediction_threshold = 20  # piksel cinsinden
        #self.motor_speed = 0  # 0 ile 100 arası bir değer
        self.cable_direction = "unknown"  # Kablo yönü
        self.camera_rotation = 0  # 0, 90, 180 veya 270, yani görüntü döndürme
        self.cap = None  # Kamera bağlantısını sınıf özelliği olarak tanımla

        # Kamera bağlantısını Picamera2 ile başlatın
        self.picam2 = Picamera2(0)
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        self.picam2.start()

    
    def preprocess_image(self, image):
        # Parlaklığı azalt
        alpha = 1  # Kontrast kontrolü (1.0 -3.0)
        beta = 0  # Parlaklık kontrolü (0 -100)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def detect_yellow(self, frame):
        # RGB'den HSV'ye dönüşüm
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Sarı renk aralığını genişletme
        lower_yellow = np.array([20, 80, 130])  # H değerini düşürdük, S değerini düşürdük
        upper_yellow = np.array([70, 255, 255])  # H değerini yükselttik

        # Renk maskesi oluşturma
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Gürültüyü azaltmak için morfolojik işlemler
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Kontrastı artırma
        mask = cv2.equalizeHist(mask)

        # Eşikleme işlemi
        _, mask = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)

        return mask

    def detect_shape(self, contour):
        epsilon = 0.028 * cv2.arcLength(contour, True)  # Epsilon değerini biraz düşürdük, degistirdim
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        shape = "UNKNOWN"

        edge_lengths = []
        for i in range(vertices):
            point1 = approx[i][0]
            point2 = approx[(i + 1) % vertices][0]
            edge_length = np.linalg.norm(point1 - point2)
            edge_lengths.append(edge_length)
        min_length = min(edge_lengths)
        max_length = max(edge_lengths)

        if 3 <= vertices <= 6:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            if vertices == 3:
                shape = "UCGEN"
            elif vertices == 4:
                # Kare ve dikdörtgen ayrımı için daha hassas kriterler
                if 0.95 <= aspect_ratio <= 1.8:
                    shape = "KARE"
                else:
                    # Dikdörtgen için ek kontrol
                    area = cv2.contourArea(contour)
                    rect_area = w * h
                    extent = float(area) / rect_area
                    if extent > 0.9:  # Dikdörtgen şeklinin doluluğunu kontrol ediyoruz
                        shape = "DIKDORTGEN"
                    else:
                        shape = "TROMBUS"
            elif vertices == 5:
                if max_length / min_length <= 2:  # Altıgen için daha sıkı oran kontrolü
                    shape = "BESGEN"
                else:
                    shape = "YONCA"

            elif vertices == 6:
                if max_length / min_length <= 2:
                    shape = "ALTIGEN"
                else:
                    shape = "YONCA"

        elif self.is_star(contour):
            shape = "YILDIZ"
        else:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.8:  # Dairesellik eşiğini biraz yükselttik
                shape = "DAIRE"
            else:
                shape = "ELIPS"  # Yeni bir şekil türü ekledim

        return shape, approx

    def is_star(self, contour):
        area = cv2.contourArea(contour)
        if area < 100:
            return False

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return False

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        distances = [np.linalg.norm(np.array(point[0]) - np.array(center)) for point in approx]

        sorted_distances = sorted(distances)

        if len(sorted_distances) >= 10:
            max_distances = sorted_distances[-5:]
            min_distances = sorted_distances[:5]
            avg_max = sum(max_distances) / 5
            avg_min = sum(min_distances) / 5

            if avg_max > 1.5 * avg_min:
                return True

        return False

    def detect_black_cable(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([103, 255, 168])    #169

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 100

        if contours:
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.drawContours(image, [largest_contour], 0, (0, 0, 255), 2)
                    cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

                    rows, cols = image.shape[:2]
                    [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.02, 0.01)

                    lefty = int((-x[0] * vy[0] / vx[0]) + y[0])
                    righty = int(((cols - x[0]) * vy[0] / vx[0]) + y[0])

                    start_point = (cols - 1, righty)
                    end_point = (0, lefty)

                    if 0 <= lefty <= rows and 0 <= righty <= rows:
                        cv2.line(image, start_point, end_point, (0, 255, 0), 2)

                    if self.initial_cable_center is None:
                        self.initial_cable_center = (cx, cy)

                    if self.screen_center:
                        cv2.line(image, self.screen_center, (cx, cy), (255, 0, 0), 2)
                        dx = cx - self.screen_center[0]
                        dy = cy - self.screen_center[1]
                        distance = np.sqrt(dx ** 2 + dy ** 2)
                        cv2.putText(image, f"Distance: {distance:.2f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 0, 0), 2)
                        cv2.putText(image, f"dx: {dx}, dy: {dy}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    return image, thresh, (cx, cy), (vx[0], vy[0]), (dx, dy)

        return image, thresh, None, None, None


    def update_cable_direction(self, current_position, direction_vector):
        if current_position is None:
            return

        self.cable_positions.append(current_position)

        if len(self.cable_positions) < 10:  # Increased from 5 to 10
            return

        last_positions = list(self.cable_positions)[-10:]  # Using last 10 positions instead of 5
        dx = last_positions[-1][0] - last_positions[0][0]
        dy = last_positions[-1][1] - last_positions[0][1]

        # Increase the threshold for horizontal/vertical determination
        direction_threshold = 10  # Increased from implicit 0 to 20

        if abs(dx) > abs(dy) + direction_threshold:
            self.cable_direction = "horizontal"
            if dx > 0:
                new_direction = "Saga git"
            else:
                new_direction = "Sola git"
            self.vertical_movement = False
        elif abs(dy) > abs(dx) + direction_threshold:
            self.cable_direction = "vertical"
            if dy > 0:
                new_direction = "Asagi git"
            else:
                new_direction = "Yukari git"
            self.vertical_movement = True
        else:
            new_direction = "Duz git"  # Default to straight if the direction is unclear

        current_time = time.time()
        turn_direction = self.determine_turn_direction()

        if turn_direction and (current_time - self.last_direction_change) > self.direction_change_cooldown:
            self.turn_detected = True
            self.turn_direction = turn_direction
            self.last_direction_change = current_time
        else:
            self.turn_detected = False
            self.turn_direction = None

        self.current_direction = new_direction if not self.turn_detected else self.turn_direction


    def determine_turn_direction(self):
        if len(self.cable_positions) < 30: # Increased from 10 to 30
            return None

        last_positions = list(self.cable_positions)[-10:]
        first_half = last_positions[:5]
        second_half = last_positions[5:]

        dx1 = first_half[-1][0] - first_half[0][0]
        dy1 = first_half[-1][1] - first_half[0][1]
        dx2 = second_half[-1][0] - second_half[0][0]
        dy2 = second_half[-1][1] - second_half[0][1]

        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)

        angle_diff = angle2 - angle1

        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        if angle_diff > 0:
            return "Sola don"
        else:
            return "Saga don"

    def detect_sharp_turn(self):
        if len(self.cable_positions) < 50: # Increased from 10 to 50  #40 olarak dene bir de zoomla
            return False

        current_time = time.time()
        if current_time - self.last_sharp_turn_time < self.sharp_turn_cooldown:
            return False

        last_positions = list(self.cable_positions)[-10:]
        first_half = last_positions[:5]
        second_half = last_positions[5:]

        dx1 = first_half[-1][0] - first_half[0][0]
        dy1 = first_half[-1][1] - first_half[0][1]
        dx2 = second_half[-1][0] - second_half[0][0]
        dy2 = second_half[-1][1] - second_half[0][1]

        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)

        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

        if angle_diff > np.pi / 6:  # 30 dereceden fazla değişim varsa keskin dönüş var demektir
            self.last_sharp_turn_time = current_time
            return True

        return False

    def get_alignment_instruction(self, cable_center):
        if not cable_center or not self.screen_center:
            return "Hizalama bilgisi yok", 0

        dx = cable_center[0] - self.screen_center[0]

        # Increase the alignment threshold
        self.alignment_threshold = 40  # Increased from 10 to 30 - ben 50 den 40 yaptım, degistirdim

        if abs(dx) <= self.alignment_threshold:
            return "Hizali", 0
        elif dx > 0:
            return "Saga hizala", min(abs(dx) / 2, 100)
        else:
            return "Sola hizala", min(abs(dx) / 2, 100)

    def get_direction_instruction(self, cable_center):
        if not cable_center or not self.screen_center:
            return "Yön bilgisi yok", 0

        dx = cable_center[0] - self.screen_center[0]

        # Increase the straight threshold significantly
        straight_threshold = self.alignment_threshold * 3  # Increased from 2 to 3 times the alignment threshold

        if abs(dx) <= straight_threshold:
            return "Duz git", 60 # Default speed is 60
        elif dx > 0: # Increase the turn threshold
            return "Saga git", min(abs(dx), 100)
        else:
            return "Sola git", min(abs(dx), 100)

    def predict_turn(self):
        if len(self.cable_positions) < 20: # Increased from 10 to 20
            return None

        positions = list(self.cable_positions)
        first_10 = positions[-20:-10]
        last_10 = positions[-10:]

        dx1 = sum(pos[0] for pos in first_10) / 10
        dx2 = sum(pos[0] for pos in last_10) / 10


    def get_next_direction(self):
        sharp_turn = self.detect_sharp_turn()

        if sharp_turn:
            return self.determine_turn_direction()

        if self.turn_detected:
            return self.turn_direction

        return self.current_direction

    def set_camera_rotation(self, rotation):
        if rotation in [0, 90, 180, 270]:
            self.camera_rotation = rotation
            print(f"Kamera rotasyonu {rotation} derece olarak ayarlandı.")
        else:
            print("Geçersiz rotasyon değeri. 0, 90, 180 veya 270 olmalıdır.")

    def rotate_image(self, image):
        if self.camera_rotation == 0:
            return image
        elif self.camera_rotation == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.camera_rotation == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif self.camera_rotation == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image

    def run(self):
        self.start_time = time.time()

        # İlk kareyi al
        frame = self.picam2.capture_array()
        frame = zoom(1, frame)
        self.screen_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        sharp_turn_detected = False

        # Benzersiz bir dosya adı oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"Saf_kayit_{timestamp}.avi"
        yellow_video_filename = f"Yellow_detection_{timestamp}.avi"

        fourcc1 = cv2.VideoWriter_fourcc(*"XVID")
        out1 = cv2.VideoWriter(video_filename, fourcc1, 20.0, (1200, 400))
        out_yellow = cv2.VideoWriter(yellow_video_filename, fourcc1, 20.0, (frame.shape[1], frame.shape[0]))

        yellow_detected_frames = 0
        yellow_detection_threshold = 3  # Kaç kare boyunca sarı tespit edilirse kaydetmeye başlayacağız
        yellow_recording = False

        while True:
            # Kareyi Picamera2'den al
            frame = self.picam2.capture_array()
            frame = zoom(1, frame)

            # BGR formatına dönüştür
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Görüntüyü döndür
            frame = self.rotate_image(frame)
            frame = self.preprocess_image(frame)

            self.frame_center = frame.shape[1] // 2
            self.frame_center_y = frame.shape[0] // 2

            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            yellow_mask = self.detect_yellow(blurred)
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color_mask = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
            contour_image = frame.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

            detected_shapes = []

            if len(contours) > 0:
                yellow_detected_frames += 1
                if yellow_detected_frames >= yellow_detection_threshold:
                    yellow_recording = True
            else:
                yellow_detected_frames = 0
                yellow_recording = False

            if yellow_recording:
                out_yellow.write(frame)

            # Aydınlatma koşullarını kontrol et
            average_brightness = np.mean(frame)
            if average_brightness < 100:  # Düşük ışık
                self.lower_yellow = np.array([20, 50, 50])
                self.upper_yellow = np.array([40, 255, 255])
            elif average_brightness > 200:  # Yüksek ışık
                self.lower_yellow = np.array([20, 100, 100])
                self.upper_yellow = np.array([40, 255, 255])
            else:  # Normal ışık
                self.lower_yellow = np.array([20, 70, 90])
                self.upper_yellow = np.array([40, 255, 255])
            # Sarı renk tespiti
            yellow_mask = self.detect_yellow(blurred)


            for contour in contours:
                if cv2.contourArea(contour) > 250:  # Minimum alanı biraz düşürdük, degistirdim
                    shape, approx = self.detect_shape(contour)

                    x, y, w, h = cv2.boundingRect(contour)
                    left = x
                    right = x + w
                    top = y
                    bottom = y + h

                    if left > 0 and right < 640 and top > 0 and bottom < 480:
                        if shape != "UNKNOWN":
                            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.putText(frame, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            detected_shapes.append(shape)

                            if shape not in [s[0] for s in self.detected_shapes]:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"shape_{shape}_{timestamp}.jpg"
                                cv2.imwrite(filename, frame)
                                self.detected_shapes.append((shape, filename))
                                self.yellow_object_detected = True
                                print(f"Tespit edilen sari cisim: {shape}")
                                print(f"Kablo yonu: {self.cable_direction}")

            if self.following_cable:
                frame, black_mask, cable_center, direction_vector, distance_vector = self.detect_black_cable(frame)

                if cable_center and direction_vector:
                    self.cable_positions.append(cable_center)

                    alignment_instruction, alignment_speed = self.get_alignment_instruction(cable_center)
                    direction_instruction, direction_speed = self.get_direction_instruction(cable_center)

                    turn_prediction = self.predict_turn()

                    # Komutları teker teker verme
                    if alignment_instruction != "Hizali":
                        active_instruction = alignment_instruction
                        active_speed = alignment_speed
                    else:
                        active_instruction = direction_instruction
                        active_speed = direction_speed

                    cv2.putText(frame, f"Aktif Komut: {active_instruction}", (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Motor kontrol simülasyonu
                    motor_status = []
                    if active_instruction == "Saga hizala":
                        motor_status.append("Sol on motor calisiyor")
                        motor.turn_right(0.1) # Sağa dönme işlemi

                    elif active_instruction == "Saga git":
                        motor_status.append("Sol motorlar calisiyor")
                        motor.go_right(0.1) # Sağa gitme işlemi

                    elif active_instruction == "Sola hizala":
                        motor_status.append("Sag on motor calisiyor")
                        motor.turn_left(0.1) # Sola dönme işlemi

                    elif active_instruction == "Sola git":
                        motor_status.append("Sag motorlar calisiyor")
                        motor.go_left(0.1) # Sola gitme işlemi

                    elif active_instruction == "Duz git":
                        motor_status.append("Iki on motor calisiyor")
                        motor.move_forward(0.1) # Düz gitme işlemi

                    # Motor durumlarını alt alta yazdırma
                    for i, status in enumerate(motor_status):
                        cv2.putText(frame, f"Motor Durumu: {status}", (10, 310 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    if turn_prediction:
                        cv2.putText(frame, f"Donus Tahmini: {turn_prediction}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)

            if self.yellow_object_detected:
                cv2.putText(frame, "Sari cisim tespit edildi!", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),2)
            else:
                cv2.putText(frame, "Sari cisim yok", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if not self.following_cable and not detected_shapes:
                self.following_cable = True
                self.yellow_object_detected = False

            if self.screen_center:
                cv2.circle(frame, self.screen_center, 5, (255, 0, 0), -1)

            combined = np.hstack((frame, color_mask, contour_image))
            cv2.imshow('Combined', combined)

            # Siyah kablo maskesini göster
            cv2.imshow('Black Cable Mask', black_mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame = self.preprocess_image(frame)
            
            saf_video = cv2.resize(combined, (1200,400))
            out1.write(saf_video)
            combined_frame = np.hstack((frame, color_mask, contour_image))
            out1.write(combined_frame)


        #self.cap.release()  # Kamera bağlantısını kapat

        # Temizlik
        out1.release()
        out_yellow.release()
        cv2.destroyAllWindows()
        self.picam2.stop()

        print("Tespit edilen şekiller:")
        for shape, filename in self.detected_shapes:
            print(f"{shape}: {filename}")


if __name__ == "__main__":
    detector = UnderwaterShapeDetector()
    detector.set_camera_rotation(0)  # Kamera rotasyonunu ayarla
    detector.run()  # Programı çalıştır
