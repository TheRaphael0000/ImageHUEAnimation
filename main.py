import cv2
from opensimplex import OpenSimplex
from console_progressbar import ProgressBar
import numpy as np
import time
import os
import math

imagePath = "12287109_p0.jpg"
# Simplexnoise steps
dX = 0.3
dY = 0.3
dT = 2  # animation speed
NB_FRAMES = 120
EFFECT_STRENGTH = 50  # between 0 and 255

# Display
DISPLAY = True
DISPLAY_NOISE = True
INPUT_SCALE = 0.25
EXIT_KEY = ord('q')
FRAME_RATE = 30 # if DISPLAY enabled
FRAME_PERIODE = 1000 // FRAME_RATE

# Save
SAVE = True

def mili():
    return int(round(time.time() * 1000))

if __name__ == '__main__':
    # Loadings
    imagePath = os.path.join(os.getcwd(), imagePath)

    if not os.path.isfile(imagePath):
        print(imagePath, "can't be found")
        exit()

    img = cv2.imread(imagePath)
    if img is None:
        print(imagePath, "is not an image")
        exit()

    img = cv2.resize(img, None, fx=INPUT_SCALE, fy=INPUT_SCALE)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    simplex = OpenSimplex()
    noise = np.zeros((20, 20))

    pb = ProgressBar(total=NB_FRAMES-1, decimals=1, length=40, fill='-', zfill=' ')

    print("Your animation is in process : ")

    for step in range(NB_FRAMES):
        stepPercent = step / NB_FRAMES
        pb.print_progress_bar(step)

        tx = dT * math.cos(stepPercent * 2 * math.pi)
        ty = dT * math.sin(stepPercent * 2 * math.pi)

        # Create the noise plane
        for x in range(noise.shape[0]):
            for y in range(noise.shape[1]):
                n = simplex.noise4d(x * dX, y * dY, tx, ty)
                noise[x, y] = n

        # Resize it to fit the image
        resized_noise = cv2.resize(noise, (img.shape[1], img.shape[0]))

        # apply the effect to the hue
        hue_effect = h + resized_noise * EFFECT_STRENGTH
        np.clip(hue_effect, 0, 255, out=hue_effect)
        hue_effect = hue_effect.astype('uint8')

        # create the image back to be displayed
        effect = cv2.merge((hue_effect, s, v))
        effect = cv2.cvtColor(effect, cv2.COLOR_HSV2BGR)

        if SAVE:
            outPath = os.path.join("out", str(step) + ".jpg")
            cv2.imwrite(outPath, effect)

        if DISPLAY:
            lastT = mili()
            deltaT = FRAME_PERIODE - mili() - lastT
            deltaT = np.array(deltaT)
            np.clip(deltaT, 1, FRAME_PERIODE, out=deltaT)

            if cv2.waitKey(deltaT) == EXIT_KEY:
                exit()

            if DISPLAY_NOISE:
                cv2.imshow("noise", resized_noise)
            cv2.imshow(imagePath, effect)


    print("Done")
