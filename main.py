import cv2
from opensimplex import OpenSimplex
import numpy as np
import time
import os

imagePath = "2.jpg"

imagePath = os.path.join(os.getcwd(), imagePath)

if not os.path.isfile(imagePath):
    print(imagePath, "can't be found")
    exit()

img = cv2.imread(imagePath)
if img is None:
    print(imagePath, "is not an image")
    exit()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Simplexnoise steps
dX = 0.05
dY = 0.05
dT = 0.01 # animation speed
t = 0

# Frame rate
FRAME_RATE = 30
FRAME_PERIODE = 1000 // FRAME_RATE

EFFECT_STRENGTH = 100 # between 0 and 255

EXIT_KEY = ord('q')

simplex = OpenSimplex()
noise = np.zeros((20,20)) # can be augemented

def mili():
    return int(round(time.time() * 1000))

while True:
    lastT = mili()

    # Create the noise plane
    for x in range(noise.shape[0]):
        for y in range(noise.shape[1]):
            n = simplex.noise3d(x*dX, y*dY, t)
            noise[x,y] = n
    t += dT

    # Resize it to fit the image
    resized_noise = cv2.resize(noise, (img.shape[1], img.shape[0]))

    # apply the effect to the hue
    hue_effect = h + resized_noise * EFFECT_STRENGTH
    np.clip(hue_effect, 0, 255, out=hue_effect)
    hue_effect = hue_effect.astype('uint8')

    # create the image back to be displayed
    effect = cv2.merge((hue_effect, s, v))
    effect = cv2.cvtColor(effect, cv2.COLOR_HSV2BGR)

    # animation frame_rate
    deltaT = FRAME_PERIODE - mili() - lastT
    deltaT = np.array(deltaT)
    np.clip(deltaT, 1, FRAME_PERIODE, out=deltaT)

    if cv2.waitKey(deltaT) == EXIT_KEY:
        exit()

    cv2.imshow("noise", resized_noise)
    cv2.imshow(imagePath, effect)
