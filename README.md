# Bilingual-Extraction-OCR
Tools to extract bilingual dictionaries or translations from PDF or other formats

#########

This README describes a pipeline for extracting bilingual dictionaries from scanned documents.

The pipeline is deterministic. It uses Tesseract OCR and FastText to attempt to extract both glossaries and multi-sentence translations from documents that have have tables of column numbers that are multiples of 2. The workflow is as follows:
1. A dictionary or columnar document of translations in PDF or other image format is downloaded to the working directory.
2. Convert2PNG.py is run to convert the file into a readable format for Tensorflow.
3. hocr.py is run taking as input the converted PNG files from the last script. The output of hocr.py are a series of .pkl files that are stored in a user defined directory. This step requires Tensorflow 4.0 for optimal results.
4. The folder containing the pkl files is passed to generateDictionary.py. This script analyzes the output from Tensorflow to form a dictionary of bilingual data contained in columnar format. It emphasizes precision over recall to reduce the influence of erroneous data for downstream tasks.

