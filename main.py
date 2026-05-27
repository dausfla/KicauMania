import math
import time

import cv2

from audio import AudioPlayer
from detector import DetectorConfig, KicauDetector
from ui import draw_debug_overlay, GifPlayer, GifWindow

ASSET_MUSIC_PATH = "assets/music.mp3"
ASSET_GIF_PATH = "assets/cat.gif"

MIRROR_PREVIEW = False
DANCE_GRACE_S = 1
HAND_MOUTH_DISTANCE = 120.0  # pixel, makin besar makin mudah trigger


def main() -> None:
    config = DetectorConfig(
        mouth_distance_px=120.0,
        wave_speed_px_s=80.0,
        wave_dir_changes=1,
        wave_min_move_px=2.0,
        wave_window_s=2.0,
        wave_horizontal_ratio=0.5,
        mouth_hold_s=0.1,
        wave_hold_s=0.1,
        cooldown_s=2.0,
    )

    detector = KicauDetector(config)
    audio = AudioPlayer(ASSET_MUSIC_PATH)
    gif = GifPlayer(ASSET_GIF_PATH)
    gif_window = GifWindow()
    dance_live = False
    last_dance_active = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            data = detector.process(frame, now)
            data["trajectory"] = detector.motion.get_points()
            draw_debug_overlay(frame, data)

            # Cek tangan mana saja (kiri atau kanan) yang dekat mulut/hidung
            mouth = data.get("mouth_center")
            left_hand = data.get("left_hand")
            right_hand = data.get("right_hand")

            hand_near_mouth = False
            if mouth:
                if left_hand:
                    dist = math.hypot(
                        mouth[0] - left_hand["palm_center"][0],
                        mouth[1] - left_hand["palm_center"][1],
                    )
                    if dist <= HAND_MOUTH_DISTANCE:
                        hand_near_mouth = True
                if right_hand:
                    dist = math.hypot(
                        mouth[0] - right_hand["palm_center"][0],
                        mouth[1] - right_hand["palm_center"][1],
                    )
                    if dist <= HAND_MOUTH_DISTANCE:
                        hand_near_mouth = True

            if hand_near_mouth:
                last_dance_active = now
                if not dance_live:
                    dance_live = True
                    audio.play_once()

            if (
                dance_live
                and not hand_near_mouth
                and (now - last_dance_active) > DANCE_GRACE_S
            ):
                dance_live = False
                audio.stop()
                gif.stop()
                gif_window.hide()

            if dance_live:
                if not gif.playing:
                    gif.start(now)
                cat_frame = gif.get_frame(now)
                if cat_frame is not None:
                    gif_window.show(cat_frame)
            else:
                gif.stop()
                gif_window.hide()

            display = cv2.flip(frame, 1) if MIRROR_PREVIEW else frame
            cv2.imshow("Kicau Mania Detector", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gif_window.hide()
        detector.shutdown()
        audio.shutdown()


if __name__ == "__main__":
    main()