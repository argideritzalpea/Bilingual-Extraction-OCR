# Bilingual-Extraction-OCR
Tools to extract bilingual dictionaries or translations from PDF or other formats.
This repo contains code for OCR table extraction of Latin unicode

#########

# Overview

This README describes a pipeline for extracting bilingual dictionaries from scanned documents.

This project uses Tesseract OCR and fastText to extract both glossaries and multi-sentence translations from documents that have have tables of column numbers that are multiples of 2.
1. A dictionary or columnar document of translations in PDF is downloaded to the working directory.
2. `Convert2PNG.py` is run to convert the file into a readable format for Tensorflow.
3. `extract.py` is run taking as input the converted PNG files from the last script. The output of hocr.py are a series of .pkl files that are stored in a user defined directory. This step requires Tensorflow 4.0 for optimal results.
4. The folder containing the .pkl files is passed to `generateDictionaryV2.py`. This script analyzes the output from Tensorflow to form a dictionary of bilingual data contained in columnar format. It emphasizes precision over recall to reduce the influence of erroneous data for downstream tasks.

########

# Preparation




You may need to explicitly set your Python encoding to be UTF-8. From the command line:
`export PYTHONIOENCODING=UTF8`

########

# Usage:

# Convert2PNG.py
This script is run on a PDF image (e.g. digital version of PeaceCorp language manual, etc.) to convert and copy each page into PNG format into a user-defined directory. This step is necessary to run the extract.py preprocessing step.

`python Convert2PNG.py -i [path_of_PDF_language_resource] -o [path_of_output_folder_of_PNG_images] -x [specify_pages_to_convert (e.g. 5-15)] `

# extract.py
This script calls a directory of PNG images on which to run OCR processing. It accepts the output of `Convert2PNG.py` as input, and returns a folder of files in .pkl format. This script must be run before running the next script, generateDictionaryV2.py. The output should be 3 pkl files for each PNG in the input folder.

`python extract.py -i [input_folder_of_PNG_images]' -o '[output_folder_of_pkl_files]`

# generateDictionaryV2.py
This is the main script that converts Tesseract 4.0 output into a Python dictionary.

`python ./generateDictionaryV2.py -i [input_folder_of_pkl_files] -b [MeanShift_quantile_threshold (experiment with values of .02 and .03)] -o [name_of_output_dictionary] > [path_of_log_and_error_file]`
