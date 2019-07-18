import PIL
import json
import pytesseract
import argparse
import cv2
import os
from wand.image import Image
from wand.color import Color

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--labeled", required=False, help="path to labelbox json label output")
ap.add_argument("-i", "--PDF", required=True, help="path to input PDF to be PNG'ed")
ap.add_argument("-o", "--output", required=True, help="dir path to output PDF to be PNG'ed")
ap.add_argument("-x", "--pages", type=str, help="pages of pdf to convert")
args = vars(ap.parse_args())

def parseIndices():
    try:
        pagerange = [int(i) for i in args['pages'].split("-")]
    except:
        if args['pages'] == 'all':
            pagerange = 'all'
    return pagerange

def convert_pdf(filename, output_path, resolution=350, tables=None):
    """ Convert a PDF into images.

        All the pages will give a single png file with format:
        {pdf_filename}-{page_number}.png

        The function removes the alpha channel from the image and
        replace it with a white background.
    """
    all_pages = Image(filename=filename, resolution=resolution)
    indices = parseIndices()
    if indices == 'all':
        beg = 0
        end = len(all_pages.sequence)
    else:
        beg = indices[0] - 1
        end = indices[1] - 1
    for i, page in enumerate(all_pages.sequence):
        if i == end+1:
            break
        elif i >= beg:
            with Image(page) as img:
                original = img
                image_filename = os.path.splitext(os.path.basename(filename))[0]
                image_filename = '{}-{}.png'.format(image_filename, i+1)
                print(image_filename)
                if image_filename in tables:
                    print(image_filename)
                    for index, table in enumerate(tables[image_filename]):
                        img = original
                        left, top, right, bottom = table[0]['x'], table[0]['y'], table[1]['x'], table[1]['y']
                        img.crop(left, top, right, bottom)
                        img.format = 'png'
                        img.background_color = Color('white')
                        img.alpha_channel = 'remove'
                        image_filename = image_filename.rsplit(".")[0] + "-" + str(index) + image_filename.rsplit(".")[1]
                        image_filename = os.path.join(output_path, image_filename)
                        img.save(filename=image_filename)


inpath = args["PDF"]
outpath = args["output"]
labeled = args["labeled"]

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
                tables[file].append(table["geometry"])

convert_pdf(inpath, outpath, 350, tables)
