from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#Q1 word count
def wordCount(fileLoc):
	
	wordList = {}
	wordCount = 0
	f = open(fileLoc, "r")
	lines = f.readlines()
	for line in lines:
		words = line.split()
		for word in words:
			if(word in wordList):
				wordList[word]+=1
			else:
				wordList[word] = 1
	f.close()

	#write into Q1.txt
	f = open("Q1.txt", "w+")
	for wordName in wordList.keys():
		if(wordCount<(len(wordList)-1)):
			content = wordName + " " + str(wordCount) + " " + str(wordList[wordName]) + "\n"
			wordCount+=1
		else:
			content = wordName + " " + str(wordCount) + " " + str(wordList[wordName])
			wordCount+=1
		f.write(content)
	f.close()
	return wordList

#wordList = wordCount("words.txt")

#Q2 image processing 
image = Image.open("westbrook.jpg")
imageArr = np.array(image)
for row in range(imageArr.shape[0]):
	for col in range(imageArr.shape[1]):
		imageArr[row, col] = [int(value/2) for value in imageArr[row, col]]

imagePro = Image.fromarray(imageArr)
imagePro.save("Q2.jpg")

