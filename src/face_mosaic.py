import cv2

def detect_and_mosaic_faces(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    
    # 얼굴 감지를 위한 Haarcascade 사용
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # 감지된 얼굴에 모자이크 처리
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (w//10, h//10))  # 모자이크 크기 조절
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_AREA)
        image[y:y+h, x:x+w] = face
    
    # 모자이크 처리된 이미지를 반환
    return image

if __name__ == "__main__":
    # 이미지 경로 설정
    input_image_path = "data/input/image.jpg"
    output_image_path = "data/output/image_mosaic.jpg"

    # 얼굴 감지 및 모자이크 처리
    result_image = detect_and_mosaic_faces(input_image_path)

    # 결과 이미지 저장
    cv2.imwrite(output_image_path, result_image)
