'''
Category r g b
Terrain 210 0 200
Sky 90 200 255
Tree 0 199 0
Vegetation 90 240 0
Building 140 140 140
Road 100 60 100
TrafficSign 255 255 0
TrafficLight 200 200 0
Pole 255 130 0
Misc 80 80 80
Truck 160 60 60
Car 255 127 80
Bus 0 139 139
Human 200 250 200
Window 0 128 0
Door 127 255 212
Sofa 128 0 128
Ceiling 240 230 140
Chair 72 61 139
Floor 0 191 255
Table 255 250 205
Curtain 230 230 250
Bed 205 92 92
Fireplace 233 150 122
Shelf 153 50 204
Lamp 160 82 45
Stair 219 112 147
Bench 245 222 179
Screen 218 165 32
Fridge 255 255 240
Ball 178 34 34
BaseballBat 210 105 30
Bow 95 158 160
Gun 255 248 220
GolfClub 173 255 47
HairBrush 224 255 255
Head 220 20 60
RightUpperArm 255 255 26
RightLowerArm 255 215 0
RightHand 255 140 0
LeftUpperArm 60 179 113
LeftLowerArm 135 206 235
LeftHand 100 149 237
Chest 248 248 255
RightUpperLeg 102 51 153
RightLowerLeg 164 89 58
RightFoot 220 173 116
LeftUpperLeg 0 0 139
LeftLowerLeg 255 182 193
LeftFoot 255 239 213
Neck 152 251 152
LeftShoulder 47 79 79
RightShoulder 85 107 47
LeftElbow 25 25 112
RightElbow 128 0 0
LeftWrist 0 255 255
RightWrist 238 130 238
LeftHip 147 112 219
RightHip 143 188 139
LeftKnee 102 0 102
RightKnee 69 33 84
LeftAnkle 50 205 50
RightAnkle 255 105 180
'''

file = open("colors.txt", "r")
color_list = []
# Repeat for each song in the text file
count = 0
for line in file:
    fields = line.split(" ")
    if count > 0:
        color_list.append([fields[1], fields[2], fields[3]])
    count +=1


