# Bilingual-Extraction-OCR
Tools to extract bilingual dictionaries or translations from PDF or other formats.
This repo contains code for OCR table extraction of Latin unicode. It will be further update to supported extraction of character sets according to input language in excess of basic Latin characters.

# Overview

This README describes a pipeline for extracting bilingual dictionaries from scanned documents.

This project uses Tesseract OCR and fastText to extract both glossaries and multi-sentence translations from documents that have have tables of column numbers that are multiples of 2.
1. A dictionary or columnar document of translations in PDF is downloaded to the working directory.
2. `Convert2PNG.py` is run to convert the file into a readable format for Tensorflow.
3. `extract.py` is run taking as input the converted PNG files from the last script. The output of hocr.py are a series of .pkl files that are stored in a user defined directory. This step requires Tensorflow 4.0 for optimal results.
4. The folder containing the .pkl files is passed to `generateDictionaryV2.py`. This script analyzes the output from Tensorflow to form a dictionary of bilingual data contained in columnar format. It emphasizes precision over recall to reduce the influence of erroneous data for downstream tasks.

# Preparation

Create a virtual environmentment and install dependencies contained in `requirements.txt`. fastText and Tesseract 4.0 may require you to go through a hands-on installation process. The repositories and installation instructions are found at the following links:
https://github.com/facebookresearch/fastText/tree/master/python
https://github.com/tesseract-ocr/

You may also need to explicitly set your Python encoding to be UTF-8. From the command line:
`export PYTHONIOENCODING=UTF8`

########

# Usage:

# Convert2PNG.py
This script is run on a PDF image (e.g. digital version of PeaceCorp language manual, etc.) to convert and copy each page into PNG format into a user-defined directory. This step is necessary to run the extract.py preprocessing step.

`python Convert2PNG.py -i [language_resource_PDF] -o [output_folder] -x [pages]`

```
-i : Input PDF file of a bilingual language resource
-o : Output directory path of converted PNG images
-x : Pages to skip in processing (i.e. 15-38)
```

# extract.py
This script calls a directory of PNG images on which to run OCR processing. It accepts the output of `Convert2PNG.py` as input, and returns a folder of files in .pkl format. This script must be run before running the next script, generateDictionaryV2.py. The output should be 3 pkl files for each PNG in the input folder.

`python extract.py -i [input_folder]' -o '[output_folder]`

```
-i : Input folder of PNG images
-o : Output folder of .pkl files
```

# generateDictionaryV2.py
This is the main script that converts Tesseract 4.0 output into a Python dictionary.

`python ./generateDictionaryV2.py -i [input_folder] -b [quantile_threshold] -o [output_dictionary] > [log/error_file]`

```
-i : Input folder of .pkl files containing the output of `extract.py`
-b : Number to pass to the sklearn MeanShift algorithm to determine column alignment (default is .02, values between .02-.03 worked best in testing)
-o : Name of output dictionary in .csv format
-l : Source language in two-character ISO language code format to help align dictionary (default is English)
-t : Target language in two-character ISO language code format to aid post-processing (IN DEVELOPMENT)
-s : List of pages to skip separated by whitespace. Accepts single indices or ranges of indices (i.e. 5-7) that correspond to input file ID numbers
```
