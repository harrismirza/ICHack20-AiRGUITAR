import cv2
import numpy as np
import requests
import base64
from datetime import datetime
import math
from collections import deque
import sys
import mido
from time import sleep

standardTuningMIDINotes = [40, 45, 50, 55, 59, 64]

def parseChordInfo(chordFilePath):
	chords = []
	with open(chordFilePath, "r") as chordFile:
		for chordInfo in chordFile:
			chordInfo = chordInfo.strip()
			chordNameNotes = chordInfo.split(':')
			chordName = chordNameNotes[0]
			chordNotes = chordNameNotes[1].split('|')
			chordMidiNotes = [-1, -1, -1, -1, -1, -1]
			for string, note in enumerate(chordNotes):
				if note != "x":
					chordMidiNotes[string] = standardTuningMIDINotes[string] + int(note)

			chords.append((chordName, chordMidiNotes))
	return chords

chords = parseChordInfo(sys.argv[1])

fretBoardLength = 300
fretLineLength = 50
numberOfNotes = len(chords)

print(chords)

def drawFretboard(frame, pose):
	try:
		leftShoulderPos = None
		rightShoulderPos = None
		rightHandPos = None
		for keypoint in pose["keypoints"]:
			if keypoint["part"] == "leftShoulder":
				leftShoulderPos = keypoint["position"]
			elif keypoint["part"] == "rightShoulder":
				rightShoulderPos = keypoint["position"]
			elif keypoint["part"] == "rightWrist":
				rightHandPos = keypoint["position"]

		gradient = ((rightShoulderPos["y"] - leftShoulderPos["y"])/(rightShoulderPos["x"] - leftShoulderPos["x"]))
		theta = -math.tan(gradient)

		topLineStart = (int(rightShoulderPos["x"]), int(rightShoulderPos["y"]))
		topLineEnd =  (int(topLineStart[0] - fretBoardLength * math.cos(theta)), int(topLineStart[1] + fretBoardLength * math.sin(theta)))
		cv2.line(frame, topLineStart, topLineEnd, (0, 0, 255))

		bottomLineStart = (int(rightShoulderPos["x"] +  + fretLineLength * math.sin(theta))), int(rightShoulderPos["y"] + fretLineLength * math.cos(theta))
		bottomLineEnd =  (int(bottomLineStart[0] - fretBoardLength * math.cos(theta)), int(bottomLineStart[1] + fretBoardLength * math.sin(theta)))
		cv2.line(frame, bottomLineStart, bottomLineEnd, (0, 0, 255))

		rightHand = (int(rightHandPos["x"]), int(rightHandPos["y"]))

		for chordIndex, interDistance in enumerate(range(0, fretBoardLength + 1, fretBoardLength//numberOfNotes)):
			interLineStart = (int(topLineEnd[0] + interDistance * math.cos(-theta)), int(topLineEnd[1] + interDistance * math.sin(-theta)))
			interLineEnd = (int(interLineStart[0] + fretLineLength * math.sin(theta)), int(interLineStart[1] + fretLineLength * math.cos(theta)))
			cv2.line(frame, interLineStart, interLineEnd, (0, 0, 255))
			if chordIndex < len(chords):
				chordName = chords[chordIndex][0]
				fontScale = 1
				while cv2.getTextSize(chordName, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)[0][0] >= 70:
					fontScale *= 0.9
				cv2.putText(frame, chordName, (interLineEnd[0] + 5, interLineEnd[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 2)

		d = (abs((((topLineEnd[1] - topLineStart[1])*rightHand[0] - (topLineEnd[0] - topLineStart[0])*rightHand[1]) + (topLineEnd[0]*topLineStart[1] - topLineEnd[1]*topLineStart[0])))/fretBoardLength)
		d2 = d ** 2
		b2 = (topLineEnd[1] - rightHand[1])**2 + (topLineEnd[0] - rightHand[0])**2
		a = math.sqrt(b2 - d2)
		noteIndex = a//(fretBoardLength/numberOfNotes)
		if noteIndex < 0:
			noteIndex = 0
		elif noteIndex >= numberOfNotes:
			noteIndex = numberOfNotes - 1
		return int(noteIndex)
	except Exception as e:
		pass


cap = cv2.VideoCapture(cv2.CAP_DSHOW)
midi = mido.open_output()
midi.send(mido.Message('program_change', channel=0, program=26))


def playChord(outputPort, chordNotes):
	for note in chordNotes:
		if note != -1:
			outputPort.send(mido.Message('note_on', channel=0, note=note))
			sleep(0.05)




averageLength = 1
leftHandVelocities = deque([], averageLength)
lastLeftHandPosition = 0

blockStrum = True

while(True):
	startTime = datetime.now()
	ret, frame = cap.read()
	flipped = cv2.flip(frame, 1)

	ret, buffer = cv2.imencode('.png', flipped)
	dataString = base64.b64encode(buffer).decode()
	dataUrl = "data:image/png;base64," + dataString

	response = requests.post('http://localhost:3000/', json={'url': dataUrl})

	json = response.json()

	if json["score"] > 0.3:
		for keypoint in json["keypoints"]:
			if keypoint["score"] > 0.5:
				if keypoint["part"] == "leftWrist":
					leftHandPosition = int(keypoint["position"]["y"])
					leftHandVelocities.append(leftHandPosition - lastLeftHandPosition)
					lastLeftHandPosition = leftHandPosition
				if keypoint["part"] == "rightWrist":
					cv2.circle(flipped, (int(keypoint["position"]["x"]), int(keypoint["position"]["y"])), 5, (0, 0, 255))



	leftHandVelocity = sum(leftHandVelocities)/averageLength


	cv2.circle(flipped, (400, 300), 10, (0, 255, 0))
	cv2.line(flipped, (0, 300), (800, 300), 2, (0, 255, 0))

	noteIndex = drawFretboard(flipped, json)

	if (leftHandVelocity) > 10 and (leftHandVelocities[-1] > 250  or leftHandVelocities[-1] < 350) and not blockStrum:
		if noteIndex is not None:
			print("strum " + chords[noteIndex][0])
			playChord(midi, chords[noteIndex][1])
			blockStrum = True
		
	if blockStrum and leftHandVelocity < 0:
		blockStrum = False

	

	frameTime = datetime.now() - startTime
	frameRate = 1/(frameTime.microseconds / 1000000)

	cv2.putText(flipped, str(int(frameRate)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

	cv2.imshow('webcam', flipped)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
