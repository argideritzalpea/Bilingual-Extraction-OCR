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
import json
from tools.column_finder import column_Factory
from tools.image_process import *

in_dir = "/Users/Chris/Documents/Grad School/UW/OCR/Mende"
out_dir = "/Users/Chris/Documents/Grad School/UW/OCR/Bilingual-Extraction-OCR/out"
labeled = "/Users/Chris/Documents/Grad School/UW/OCR/Bilingual-Extraction-OCR/gold_data/Mende/labelbox/tables3.json"

if labeled != None:
    data = {}
with open(labeled) as table:
    data = json.load(table)

tables = {}
for entry in data:
    file = entry["External ID"]
    if entry["Label"] != "Skip":
        for table in entry["Label"]["Table"]:
            tables.setdefault(file, [])
            tables[file].append({"geom": table["geometry"], "colnum": table["select_how_many_columns"]})

class Column():

    def __init__(self, image, bounds, lang="en"):
        self.lang = lang
        self.bounds = bounds
        self.col_image = image[0:image.shape[0], int(bounds[0]):int(bounds[1])]

    def write(self):
        self.col_temp_path = tempWrite(self.col_image)

    def del_temp(self):
        os.remove(self.col_temp_path)

    def processTesseract(self):
        self.raw_tesseract = raw_Tesseract(self.col_temp_path)

    def processTesseractHOCR(self):
        boxes = pytesseract.image_to_data(PIL.Image.open(self.col_temp_path), config='--psm 5 -c preserve_interword_spaces=1')
        self.boxes = pd.read_table(StringIO(boxes), sep='\t', quotechar='\t')

    def getRawTesseractOutput(self):
        return self.raw_tesseract


    """
    TODO: Add function to produce lines to tesseract output that can be compared
    vs other columns'.
    """

    #def runTesseract(self, lang=self.lang):
    #    self.output =

class AnnotatedTable():

    default_language_paradigm = ["source", "target"]

    def __init__(self, image, table_number, geometry, column_format):
        self.image = preprocess(grey(image))
        self.table_number = table_number
        self.geometry = geometry
        self.column_format = column_format
        self.columns = []

    def getImage(self, image):
        return self.image

    def write(self):
        self.temp_path = tempWrite(self.image)

    def del_temp(self):
        os.remove(self.temp_path)

    def processImage(self):
        self.image = preprocess(grey(image))

    def parseColumns(self, pattern="default"):
        matcher = {"bi": 2, "tri": 3, "quad": 4}
        column_num = 0
        for match in matcher:
            if self.column_format.startswith(match):
                column_num = matcher[match]
                break
        spread = column_Factory(self.temp_path, column_num)

        for item, lang in zip(spread, default_language_paradigm*2):
            self.columns.append(Column(self.image, bounds=item, lang=lang))

    def runTesseractOnColumns(self):
        for column in self.columns:
            column.write()
            column.processTesseract()
            column.processTesseractHOCR()
"""
import matplotlib.pyplot as plt
image.shape[0]
plt.imshow(image)
"""
annotable.runTesseractOnColumns()
annotable.columns[0].boxes

column_Factory("../data/page.png", 2)

image_path = "MendeManual-85.png"
image_name = image_path.strip('.png')
print(image_path)
image = load(in_dir + '/' + image_path)
if tables[image_path] != "Skip":
    for num, table in enumerate(tables[image_path]):
        annotable = AnnotatedTable(image, num, table["geom"], table["colnum"])
        annotable.processImage()
        annotable.write()
        annotable.parseColumns()
