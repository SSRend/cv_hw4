import cv2 as cv
import numpy as np
import os
from PIL import Image

# === 내부 파라미터 ===
K = np.array([
    [1972.3442, 0, 1042.9267],
    [0, 1920.0901, 296.8256],
    [0, 0, 1]
], dtype=np.float32)

# === 왜곡 계수 ===
dist_coeff = np.array([
    -0.167291, 0.543998, 0.0, 0.0, 0.0
], dtype=np.float32)

# === 체스보드 설정 ===
board_pattern = (10, 7)
board_cellsize = 1.0

# === 큐브 정의: 좌상단 교차점 기준 (6x6x1 크기) ===
cube_origin = np.array([0, 0, 0])
cube = np.float32([
    [0, 0, 0], [6, 0, 0], [6, 6, 0], [0, 6, 0],
    [0, 0, -1], [6, 0, -1], [6, 6, -1], [0, 6, -1]
]) + cube_origin

def load_gif_frames(gif_path):
    gif_path = os.path.join(os.path.dirname(__file__), gif_path)
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.copy().convert('RGB')
            frames.append(np.array(frame))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames

# === 움짤 로드 ===
gif_frames = load_gif_frames('happy_cat.gif')
frame_idx = 0

# === 영상 열기 및 저장 설정 ===
video_path = os.path.join(os.path.dirname(__file__), 'chessboard.mp4')
out_path = os.path.join(os.path.dirname(__file__), 'chessboard_arCube_withCat.mp4')
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

fps = cap.get(cv.CAP_PROP_FPS)
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(out_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCornersSB(gray, board_pattern, flags=cv.CALIB_CB_NORMALIZE_IMAGE)

    if found:
        corners = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        objp = np.zeros((board_pattern[0]*board_pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
        objp *= board_cellsize

        ret, rvec, tvec = cv.solvePnP(objp, corners, K, dist_coeff)

        imgpts, _ = cv.projectPoints(cube, rvec, tvec, K, dist_coeff)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # 기본 큐브 그리기
        frame = cv.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
        for i, j in zip(range(4), range(4, 8)):
            frame = cv.line(frame, imgpts[i], imgpts[j], (255, 0, 0), 2)
        frame = cv.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)

        # AR 이미지 투사 (윗면)
        top_plane = cube[4:8]  # 위쪽 꼭짓점 네 개
        img_top, _ = cv.projectPoints(top_plane, rvec, tvec, K, dist_coeff)
        img_top = np.int32(img_top).reshape(-1, 2)

        cat = gif_frames[frame_idx % len(gif_frames)]
        src_pts = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])
        dst_pts = np.float32(img_top)
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv.warpPerspective(cat, M, (w, h))

        gray_cat = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray_cat, 10, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)
        bg = cv.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv.bitwise_and(warped, warped, mask=mask)
        frame = cv.add(bg, fg)

        frame_idx += 1

    out.write(frame)
    cv.imshow('AR with Cat', frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
out.release()
cv.destroyAllWindows()
print(f"✅ 저장 완료: {out_path}")