import PIL
import pytesseract
import argparse
import cv2
import os
from wand.image import Image
from wand.color import Color
import re
import pandas as pd
from string import punctuation, printable
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
from collections import Counter
import pickle


def mode(list):
	hash = Counter()
	for element in list:
		hash[element] += 1
	highest = hash.most_common()[0][1]
	times = int()
	mode = int()
	for ind, i in enumerate(hash.most_common()):
		if ind == 0:
			times = i[1]
			mode = i[0]
		if i[1] != times:
			break
		mode = max(mode, i[0])
	return mode

#image_path = 'Mende/MendeManual-50.png'
#out_path = 'output'

def load(path):
	return cv2.imread(path)

def grey(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def preprocess(img):
	"""
	check to see if we should apply thresholding to preprocess the
	image
	"""
	#if "thresh" in args["preprocess"]:
	image = cv2.threshold(img, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	"""
	make a check to see if median blurring should be done to remove
	noise
	"""
	#if "blur" in args["preprocess"]:
		#image = cv2.medianBlur(gray, 3)
	return image

def tempWrite(img):
	"""
	write the grayscale image to disk as a temporary file so we can
	apply OCR to it
	"""
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, img)
	return filename

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
def raw_Tesseract(temp):
    text = pytesseract.image_to_string(PIL.Image.open(temp), config='--psm 6 -c preserve_interword_spaces=1')
    return text

def tesseractProcess(temp):
    text = pytesseract.image_to_string(PIL.Image.open(temp), config='--psm 6 -c preserve_interword_spaces=1')
    text = re.sub('(?<!\n).*?[\:\)\;] ', '|||', text)
    text = re.sub('[^\w\n][\~\-\,\.]+[^\w\n]', '|||', text)
    text = re.sub('\ \ \ *', '|||', text)
    text = re.sub('(\ )*?(\|)+(\ )*?', '|||', text)
    text = re.sub('[%s\d]+(\|)+'  % punctuation, '|||', text)
    text = re.sub('(\|)+[%s\d]+' % punctuation, '|||', text)
    text = re.sub('(\n).(\|)+', '\n|||', text)
    text = re.sub('(\|)+.\n', '|||\n', text).split('\n')
    return text

def tesseractBoxes(temp):
	# pytesseract.image_to_pdf_or_hocr(PIL.Image.open(filename))
	boxes = pytesseract.image_to_data(PIL.Image.open(temp), config='--psm 6 -c preserve_interword_spaces=1')
	print(boxes)
	return boxes

def makeTable(a_text):
	data_struct = []
	#f = open(out_path + ".txt", "w+")
	for id, line in enumerate(a_text):
		splitline = line.split('|||')
		splitline = [x.strip(punctuation).strip() for x in splitline if x.strip(punctuation) != '']
		#print(splitline)
		#splitline = ["".join(re.split(" [--]*? " % punctuation, x)) for x in splitline]
		#print(splitline)
		#splitline = splitPunc(splitline)
		#print(splitline)
		newtext = '\t'.join(splitline)
		data_struct.append([len(splitline), splitline])
		#data_struct.append()
		#print(len(splitline), newtext)
		#f.write("\t".join([str(len(splitline)), newtext])+ '\n')
	#f.close()
	return data_struct

def findCandidateBlocks(dictionary, modes):
    final_blocks = {}
    for i in dictionary:
        less = modes[i]-1
        exact = modes[i]
        more = modes[i]+1
        if modes[i] >= 2:
            final_blocks.setdefault(i, [])
            for f in dictionary[i]:
                if f[0] in [less, exact, more] and f[1] != ['']:
                    final_blocks[i].append(f)
    for x in final_blocks:
        mark = 0
        for ind, y in enumerate(final_blocks[x]):
            if ind > 1:
                if y[0] == 1 and final_blocks[x][ind-1][0] == 1:
                    del final_blocks[x][ind-1]
                if mark == 1:
                    del final_blocks[x][ind]
            else:
                mark = 0
    return final_blocks

def getModeSplits(paragraphdict):
	modedict = {}
	for i in paragraphdict:
		splitnums = list()
		for f in paragraphdict[i]:
			if f[1] != ['']:
				splitnums.append(f[0])
		modedict[i] = mode(splitnums)
	return modedict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_dir", required=True, help="directory path of input to be OCR'ed")
    ap.add_argument("-o", "--out_dir", default="out", help="output directory for pkl files")
    #ap.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing to be done")
    #ap.add_argument("-s", "--save", required=False, type=str, default=False, help="save preprocessed image")
    args = vars(ap.parse_args())

    in_dir = args["in_dir"]
    out_dir = args["out_dir"]

    for image_path in os.listdir(in_dir):
        try:
            image_name = image_path.strip('.png')
            print(image_path)
            image = load(in_dir + '/' + image_path)
            tempfile = tempWrite(preprocess(grey(image)))
            text = tesseractProcess(tempfile)
            boxes = tesseractBoxes(tempfile)
            hocr_data = pd.read_table(StringIO(boxes), sep='\t', quotechar='\t')
            os.remove(tempfile)
            data = makeTable(text)

            blockindex = 0
            paradict = {}
            for i, line_tuple in enumerate(data):
            	paradict.setdefault(blockindex, [])
            	num_groups = line_tuple[0]
            	if i == 0:
            		#line_tuple.append(blockindex)
            		paradict[blockindex].append(line_tuple)
            	if i == 1:
            		prev1_num_groups = data[i-1][0]
            		if num_groups == prev1_num_groups:
            			#line_tuple.append(blockindex)
            			paradict[blockindex].append(line_tuple)
            		else:
            			blockindex += 1
            			paradict.setdefault(blockindex, [])
            			#line_tuple.append(blockindex)
            			paradict[blockindex].append(line_tuple)
            	elif i == 2:
            		prev1_num_groups = data[i-1][0]
            		prev2_num_groups = data[i-2][0]
            		if num_groups == prev1_num_groups or num_groups == prev2_num_groups:
            			#line_tuple.append(blockindex)
            			paradict[blockindex].append(line_tuple)
            		else:
            			blockindex += 1
            			paradict.setdefault(blockindex, [])
            			#line_tuple.append(blockindex)
            			paradict[blockindex].append(line_tuple)
            	elif i > 2:
            		prev1_num_groups = data[i-1][0]
            		prev2_num_groups = data[i-2][0]
            		prev3_num_groups = data[i-3][0]
            		if num_groups == prev1_num_groups or num_groups == prev2_num_groups:
            			#line_tuple.append(blockindex)
            			paradict[blockindex].append(line_tuple)
            		elif prev1_num_groups != prev2_num_groups and prev3_num_groups == num_groups:
            			#line_tuple.append(blockindex)
            			paradict[blockindex].append(line_tuple)
            		else:
            			blockindex += 1
            			paradict.setdefault(blockindex, [])
            			#line_tuple.append(blockindex)
            			paradict[blockindex].append(line_tuple)

            modedict = getModeSplits(paradict)

            final = findCandidateBlocks(paradict, modedict)

            with open(out_dir + '/' + image_name + '.pkl', 'wb') as handle:
                pickle.dump(final, handle)
            with open(out_dir + '/' + image_name + '_modedict' + '.pkl', 'wb') as handle:
                pickle.dump(modedict, handle)
            with open(out_dir + '/' + image_name + '_pd_hocr' + '.pkl', 'wb') as handle:
                pickle.dump(hocr_data, handle)
        except Exception as e:
            print('failed on ' + image_path)
            print(e)


    #with open('filename.pkl', 'rb') as pickle_file:
    #    content = pickle.load(pickle_file)
