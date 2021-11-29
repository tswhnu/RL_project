from itertools import cycle
import random
import sys
import os
import copy
import argparse
import pickle

import pygame
from pygame.locals import *

sys.path.append(os.getcwd())

from DQN import *

SCREENWIDTH = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

# image width height indices for ease of use
IM_WIDTH = 0
IM_HEIGTH = 1
# image, Width, Height
PIPE = [52, 320]
PLAYER = [34, 24]
BASE = [336, 112]
BACKGROUND = [288, 512]
LOAD_MODEL = False

# Initialize the bot
dqn = DQN()
if LOAD_MODEL:
    dqn.load_model()


def main():
    global HITMASKS
    total_score = 0
    ave_score = 0
    # load dumped HITMASKS
    with open("hitmask.pkl", "rb") as input:
        HITMASKS = pickle.load(input)

    print('gathering the data')
    for i in range(50000):
        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo, i)
        score = crashInfo['score']
        total_score += score
        if i != 0:
            ave_score = total_score / i
        print('curr_score', crashInfo['score'], 'ave_score:', ave_score, 'epi:', i)
        showGameOverScreen(crashInfo)

        if i % 1000 == 0 and i != 0:
            print('saving model')
            dqn.save_model()


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndexGen = cycle([0, 1, 2, 1])

    playery = int((SCREENHEIGHT - PLAYER[IM_HEIGTH]) / 2)

    basex = 0

    # player shm for up-down motion on welcome screen
    playerShmVals = {"val": 0, "dir": 1}

    return {
        "playery": playery + playerShmVals["val"],
        "basex": basex,
        "playerIndexGen": playerIndexGen,
    }


def mainGame(movementInfo, epi):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo["playerIndexGen"]

    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo["playery"]

    basex = movementInfo["basex"]
    baseShift = BASE[IM_WIDTH] - BACKGROUND[IM_WIDTH]

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {"x": SCREENWIDTH + 200, "y": newPipe1[0]["y"]},
        {"x": SCREENWIDTH + 200 + (SCREENWIDTH / 2), "y": newPipe2[0]["y"]},
    ]

    # list of lowerpipe
    lowerPipes = [
        {"x": SCREENWIDTH + 200, "y": newPipe1[1]["y"]},
        {"x": SCREENWIDTH + 200 + (SCREENWIDTH / 2), "y": newPipe2[1]["y"]},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY = -9  # player's velocity along Y, default same as playerFlapped
    playerMaxVelY = 10  # max vel along Y, max descend speed
    playerMinVelY = -8  # min vel along Y, max ascend speed
    playerAccY = 1  # players downward accleration
    playerFlapAcc = -9  # players speed on flapping
    playerFlapped = False  # True when player flaps

    while True:
        ######### self code ######################
        state = get_state(playerx, playery, playerVelY, copy.deepcopy(lowerPipes))
        action = dqn.select_action(state)

        if action:
            if playery > -2 * PLAYER[IM_HEIGTH]:
                playerVelY = playerFlapAcc
                playerFlapped = True
        ######### self code ######################

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
        playerHeight = PLAYER[IM_HEIGTH]
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe["x"] += pipeVelX
            lPipe["x"] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]["x"] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]["x"] < -PIPE[IM_WIDTH]:
            upperPipes.pop(0)
            lowerPipes.pop(0)

        end_state = get_state(playerx, playery, playerVelY, copy.deepcopy(lowerPipes))
        # check for crash here
        crashTest = checkCrash(
            {"x": playerx, "y": playery, "index": playerIndex}, upperPipes, lowerPipes
        )
        if crashTest[0]:
            ######### self code ######################
            reward = -1000
            # Update the q scores
            dqn.store_transition(state, action, reward, end_state)
            ######### self code ######################
            return {
                "y": playery,
                "groundCrash": crashTest[1],
                "basex": basex,
                "upperPipes": upperPipes,
                "lowerPipes": lowerPipes,
                "score": score,
                "playerVelY": playerVelY,
            }
        ######### self code ######################
        reward = 1
        dqn.store_transition(state, action, reward, end_state)

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.optimize_model()
        ######### self code ######################

        # check for score
        playerMidPos = playerx + PLAYER[IM_WIDTH] / 2
        for pipe in upperPipes:
            pipeMidPos = pipe["x"] + PIPE[IM_WIDTH] / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
        if crashTest[0]:
            print('reward', reward)


def showGameOverScreen(crashInfo):
    return


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm["val"]) == 8:
        playerShm["dir"] *= -1

    if playerShm["dir"] == 1:
        playerShm["val"] += 1
    else:
        playerShm["val"] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = PIPE[IM_HEIGTH]
    pipeX = SCREENWIDTH + 10

    return [
        {"x": pipeX, "y": gapY - pipeHeight},  # upper pipe
        {"x": pipeX, "y": gapY + PIPEGAPSIZE},  # lower pipe
    ]


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player["index"]
    player["w"] = PLAYER[IM_WIDTH]
    player["h"] = PLAYER[IM_HEIGTH]

    # if player crashes into ground
    if (player["y"] + player["h"] >= BASEY - 1) or (player["y"] + player["h"] <= 0):
        return [True, True, False]
    else:

        playerRect = pygame.Rect(player["x"], player["y"], player["w"], player["h"])
        pipeW = PIPE[IM_WIDTH]
        pipeH = PIPE[IM_HEIGTH]

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe["x"], uPipe["y"], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe["x"], lPipe["y"], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS["player"][pi]
            uHitmask = HITMASKS["pipe"][0]
            lHitmask = HITMASKS["pipe"][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide:
                return [True, False, True]

            if lCollide:
                return [True, False, False]

    return [False, False, False]


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False


def get_state(x, y, vel, pipe):
    """
            Get current state of bird in environment.
            :param x: bird x
            :param y: bird y
            :param vel: bird y velocity
            :param pipe: pipe
            :return: current state (x0_y0_v_y1) where x0 and y0 are diff to pipe0 and y1 is diff to pipe1
            """

    # Get pipe coordinates
    pipe0, pipe1 = pipe[0], pipe[1]
    if x - pipe[0]["x"] >= 50:
        pipe0 = pipe[1]
        if len(pipe) > 2:
            pipe1 = pipe[2]

    x0 = pipe0["x"] - x
    y0 = pipe0["y"] - y
    if -50 < x0 <= 0:
        y1 = pipe1["y"] - y
    else:
        y1 = 0

    # Evaluate player position compared to pipe
    if x0 < -40:
        x0 = int(x0)
    elif x0 < 140:
        x0 = int(x0) - (int(x0) % 10)
    else:
        x0 = int(x0) - (int(x0) % 70)

    if -180 < y0 < 180:
        y0 = int(y0) - (int(y0) % 10)
    else:
        y0 = int(y0) - (int(y0) % 60)

    if -180 < y1 < 180:
        y1 = int(y1) - (int(y1) % 10)
    else:
        y1 = int(y1) - (int(y1) % 60)

    # state = str(int(x0)) + "_" + str(int(y0)) + "_" + str(int(vel)) + "_" + str(int(y1))
    state = np.array([int(x0), int(y0), int(vel), int(y1)]).astype(dtype=np.float32)
    return state


if __name__ == "__main__":
    main()
