import cv2
import requests
import base64
from datetime import datetime
import math
from collections import deque
import sys
import mido
from time import sleep, time_ns
import threading


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
fretLineLength = 500
fretTextDistance = 50
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
		sintheta = math.sin(theta)
		costheta = math.cos(theta)

		topLineStart = (int(rightShoulderPos["x"]), int(rightShoulderPos["y"]))
		topLineEnd =  (int(topLineStart[0] - fretBoardLength * costheta), int(topLineStart[1] + fretBoardLength * sintheta))
		cv2.line(frame, topLineStart, topLineEnd, (0, 0, 255))

		bottomLineStart = (int(rightShoulderPos["x"] + fretLineLength * sintheta)), int(rightShoulderPos["y"] + fretLineLength * costheta)
		bottomLineEnd =  (int(bottomLineStart[0] - fretBoardLength * costheta), int(bottomLineStart[1] + fretBoardLength * sintheta))
		cv2.line(frame, bottomLineStart, bottomLineEnd, (0, 0, 255))

		rightHand = (int(rightHandPos["x"]), int(rightHandPos["y"]))
		interDistances = [0] + [(fretBoardLength/2) * math.log10(x) for x in range(fretBoardLength//numberOfNotes, fretBoardLength + 1, fretBoardLength//numberOfNotes)]

		d = (abs((((topLineEnd[1] - topLineStart[1])*rightHand[0] - (topLineEnd[0] - topLineStart[0])*rightHand[1]) + (topLineEnd[0]*topLineStart[1] - topLineEnd[1]*topLineStart[0])))/fretBoardLength)
		d2 = d ** 2
		b2 = (topLineEnd[1] - rightHand[1])**2 + (topLineEnd[0] - rightHand[0])**2
		a = math.sqrt(b2 - d2)
		noteIndex = int(a//(fretBoardLength/numberOfNotes))
		if noteIndex < 0:
			noteIndex = 0
		elif noteIndex >= numberOfNotes:
			noteIndex = numberOfNotes - 1

		for chordIndex, interDistance in enumerate(range(0, fretBoardLength + 1, fretBoardLength//numberOfNotes)):
			interLineStart = (int(topLineEnd[0] + interDistance * costheta), int(topLineEnd[1] + interDistance * -sintheta))
			interLineEnd = (int(interLineStart[0] + fretLineLength * sintheta), int(interLineStart[1] + fretLineLength * costheta))
			cv2.line(frame, interLineStart, interLineEnd, (0, 0, 255))
			if chordIndex < len(chords):
				chordName = chords[chordIndex][0]
				fontScale = 1
				while cv2.getTextSize(chordName, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)[0][0] >= (fretBoardLength / numberOfNotes):
					fontScale *= 0.9

				colour = (0, 0, 255)
				if chordIndex == noteIndex:
					colour = (0, 255, 0)

				chordTextBaseline = (int(interLineStart[0] + fretTextDistance * sintheta), int(interLineStart[1] + fretTextDistance * costheta))
				cv2.putText(frame, chordName, (chordTextBaseline[0] + 5, chordTextBaseline[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale, colour, 2)


		return noteIndex
	except Exception as e:
		pass


cap = cv2.VideoCapture(cv2.CAP_DSHOW)
print(mido.get_output_names())
midi = mido.open_output('VirtualMIDISynth #1 0')

def playChord(outputPort, chordNotes):
	for note in chordNotes:
		if note != -1:
			outputPort.send(mido.Message('note_on', channel=0, note=note))
			sleep(0.05)


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

	leftHandVelocity = 0

	if json["score"] > 0.3:
		for keypoint in json["keypoints"]:
			if keypoint["score"] > 0.5:
				if keypoint["part"] == "leftWrist":
					leftHandPosition = int(keypoint["position"]["y"])
					leftHandVelocity  = (leftHandPosition - lastLeftHandPosition)
					lastLeftHandPosition = leftHandPosition
					cv2.circle(flipped, (int(keypoint["position"]["x"]), int(keypoint["position"]["y"])), 5, (0, 0, 255))
				elif keypoint["part"] == "rightWrist":
					cv2.circle(flipped, (int(keypoint["position"]["x"]), int(keypoint["position"]["y"])), 5, (0, 0, 255))

	cv2.circle(flipped, (400, 300), 10, (0, 255, 0))
	cv2.line(flipped, (0, 300), (800, 300), (0, 255, 0))

	noteIndex = drawFretboard(flipped, json)

	if (leftHandVelocity) > 10 and (leftHandPosition > 280  or leftHandPosition < 320) and not blockStrum:
		if noteIndex is not None:
			print(leftHandVelocity)
			print("strum", chords[noteIndex][0], chords[noteIndex][1])
			threading.Thread(target=playChord, args=(midi, chords[noteIndex][1])).start()
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
midi.close()
