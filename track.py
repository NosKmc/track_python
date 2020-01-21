import dlib
import cv2
import copy


predictor_path = "./shape_predictor_68_face_landmarks.dat"
resize_rate = 4
cam_number = 1

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

video_input = cv2.VideoCapture(cam_number)

def pdistance(p1, p2):
    return (p1.x - p2.x)**2 + (p1.y - p2.y)**2

def face_param(face_points):
    reye, leye = calc_eye_ratio(face_points)
    mouth = calc_mouth_ratio(face_points)
    return [reye, leye, mouth]

def calc_eye_ratio(face_points):
    rp = face_points[36:42] # 右目は36~41番
    h = pdistance(rp[0], rp[3])
    if h == 0:
        return [0, 0]
    v = pdistance(rp[1], rp[5])
    r_ratio = v / h

    lp = face_points[42:48] # 左目は42~47番
    h = pdistance(lp[0], lp[3])
    if h == 0:
        return [0, 0]
    v = pdistance(lp[2], lp[4])
    l_ratio = v / h
    return [r_ratio, l_ratio]

def calc_mouth_ratio(face_points):
    mp = face_points[60:68] # 口は60~67
    h = pdistance(mp[0], mp[4])
    if h == 0:
        return 0
    v = pdistance(mp[2], mp[6])
    return v / h

def unresize(point):
    return int(point * resize_rate)


while(video_input.isOpened()):
    ret, frame = video_input.read()
    temp_frame = copy.deepcopy(frame)

    # フレーム縮小
    height, width = frame.shape[:2]
    temp_frame = cv2.resize(
        frame, (int(width/resize_rate), int(height/resize_rate)))

    # 顔検出
    dets = detector(temp_frame, 1)
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))

        # face landmark
        shape = predictor(temp_frame, d)
        face_points = []
        # 描画
        for shape_point_count in range(shape.num_parts):
            shape_point = shape.part(shape_point_count)
            
            cv2.circle(frame, (unresize(shape_point.x),
                                unresize(shape_point.y)), 2, (128, 255, 0), -1)
            # 各点をlistに入れる
            face_points.append(shape_point)
        if len(face_points) == 68:
            reye, leye, mouth = face_param(face_points)
            print(f"r_eye: {reye:.3f}, l_eye: {leye:.3f}, mouth: {mouth:.3f}")

    cv2.imshow('face landmark detector', frame)

    c = cv2.waitKey(50) & 0xFF

    if c == 27:  # ESC
        break

video_input.release()
cv2.destroyAllWindows()
