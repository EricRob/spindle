import sys

# Finds the text in the middle of two bracketed clauses
# removeBracketed('<tag>example</tag>') returns example
def getContent(line):
	curr = 0
	beginning = -1
	ending = -1
	for c in line:
		if(beginning == -1):
			if(c == '>'):
				beginning = curr + 1
		else: #beginning set already
			if(c == '<'):
				ending = curr

		curr += 1
	# for c in line
	return line[beginning:ending]
# getContent

# Reads input from the annotation list, then appends stage tuples to
# 	stageList, which is the return value
# Each element of stageList is a tuple, consisting of three numbers:
#	the stage of sleep, the timestamp at which it began, and the one at which it ended
# 0 = awake
# 1 = Stage 1
# 2 = Stage 2
# 3 = Stage 3
# 5 = REM (Stage 4 does not seem to appear in the MrOS dataset)
def readStages(filename):
	flag = 0
	start = 0
	whichStage = -1
	beginning = -1
	ending = -1
	res = []
	with open(filename) as f:
		for line in f:
			if(flag == 3): # Which Stage
				whichStage = int(getContent(line)[-1:])
				flag = 2
			elif(flag == 2): # When Start
				beginning = float(getContent(line))
				flag = 1
			elif(flag == 1): # When End
				ending = float(getContent(line)) + beginning
				# We have all of our pieces, so we need to assemble them now
				# 	before continuing to read
				res.append((whichStage, beginning, ending))
				flag = 0
			elif(line == "<EventType>Stages|Stages</EventType>\n"):
				flag = 3
		# for line in f
	# open filename
	if(res == []):
		print("Error reading file, no stages found.")
	return res
# readStages

stages = readStages("./Annotations/mros-visit1-aa0018-nsrr.xml")
#stages = readStages("./Annotations/chat-baseline-" + fileNum + "-nsrr.xml")

sys.stdout = open("detectionTimes_MrOS.txt", "w+")


newTup = []
for tup in stages:
	if(tup[0] == 2):
		newTup.append(tup);

# newTup full of stage 2 data
        
        

with open("short_dt_MrOS.txt") as f:
    for line in f:
        temp = line.split()
        duration = float(temp[1])
        time = float(temp[0])
        for tup in newTup:
            stageDur = tup[2] - tup[1]
            if time < stageDur:
                time += tup[1]
                break
            else:
                time -= stageDur
        #print(str(time) + " " + str(duration))
        print("%.6f %.6f" % (time, duration))
            
	# for line in f
# open filename
