# Bilingual-Extraction-OCR
Tools to extract bilingual dictionaries or translations from PDF or other formats.
This repo contains code for OCR table extraction of Latin unicode.

## Overview

This README describes a pipeline for extracting bilingual dictionaries from scanned documents. It involves using the freeware pdfsandwich, which uses Tesseract v.4.0 to extract multilingual text from PDF image documents and append it to the same PDF. This output can then be used in conjunction with human annotation in Tabula to extract dictionaries from image-only PDFs in languages that are written in multiple different languages.

# Prerequisite Installation

1. Install the pdfsandwich package:
On mac:
`brew install pdfsandwich`

This command will install all necessary dependencies, including Tesseract 4.0.

See the pdfsandwich site for instructions on other operating systems: http://www.tobias-elze.de/pdfsandwich/

2. Install all available Tesseract languages:
`brew install tesseract-lang`
https://formulae.brew.sh/formula/tesseract-lang

The trained models are visible at the repository of the official distribution here:
https://github.com/tesseract-ocr/tessdata_fast/

You should now be able to verify that several language models for Tesseract are installed using the command `pdfsandwich -list_langs`.

# Fine-tuning and training dependencies
Follow the instructions here if you intend to add characters to basic language or script models to Tesseract, or intend to train a new model from scratch:
https://github.com/tesseract-ocr/tesseract/wiki/TrainingTesseract-4.00#fine-tuning-for--a-few-characters

Ensure that any fine-tuning is done from extant "best" models, as opposed to the tesseract "fast" models. These are two different releases from Tesseract, and the larger "best" models are the only ones that will work with fine tuning as of the time of this writing. Otherwise, expect segmentation faults and mismatched vector sizes.

# For a new language

You may add a new language as an installed language in tesseract by adding a new language code to `src/training/language-specific.sh` within the tesseract folder.

Sample training commands:
##Dir structure:
```~/tesseract
~/tesseract/tessdata_best
~/tesseract/tessdata_best/trainmende
```


## Extract the main components and modify the character list to support the new characters you'd like to support.
https://github.com/tesseract-ocr/tesseract/blob/master/doc/combine_tessdata.1.asc
`training/combine_tessdata -e tessdata/tessdata_best/eng.traineddata \
  tessdata_best/trainmende/eng.lstm`

Set the tessdata path to tessdata_best:
`echo $TESSDATA_PREFIX`
`/home/ubuntu/tesseract/tessdata_best`

Added a langcode to `src/training/language-specific.sh` to initialize a new language of your choosing.

Copied the base language (langdata that that you will fine tune from) to new folder of the new language code in "tesseract/langdata_lstm".

For example, if the new language code picked is "men" (Mende):

```src/training/tesstrain.sh --fonts_dir /usr/share/fonts --lang men --linedata_only \
  --noextract_font_properties --langdata_dir ./langdata_lstm \
  --tessdata_dir ./tessdata_best --output_dir ~/tessdata_best/trainmende
```

```src/training/lstmtraining --model_output tessdata_best/trainmende/mende \
  --continue_from tessdata_best/trainmende/eng.lstm \
  --traineddata tessdata_best/trainmende/men/men.traineddata \
  --old_traineddata tessdata_best/eng.traineddata \
  --train_listfile tessdata_best/trainmende/men.training_files.txt \
  --max_iterations 3600
```

## Put checkpoint into traineddata file
```training/lstmtraining --stop_training   --continue_from ~/tesseract/tessdata_best/trainmende/mende0.854_331.checkpoint   --traineddata ~/tesseract/tessdata_best/trainmende/men/men.traineddata   --model_output ~/tesseract/tessdata_best/mendeout/men.traineddata
```

You may then use the traineddata file of the newly fine-tuned language as any other tesseract model.

# Flow

Run `pdfsandwich` on the bilingual document. Using the lang argument and the `+` sign, you can prepare Tesseract to encounter scripts and languages of multiple types.
`pdfsandwich -lang eng+script/Armenian input.pdf`

For a full list of options, see the documentation: http://www.tobias-elze.de/pdfsandwich/

The output produced by pdfsandwich will be a PDF with embedded extracted text.

After running `pdfsandwich` on the PDF, the output PDF can be input into Tabula which automatically extracts tables from text.
https://tabula.technology/

It is highly recommended to manually inspect and define tables as opposed to using the "autodetect" feature of Tabula. The autodetect functionality often misses columns that a human would identify. It is also advisable to create table annotations around the "thinnest" set of columns that comprise a full table. For example, if a page is comprised of two adjacent bi-columnar groups, click and drag an identifying box in Tabula around each of the bi-columnar groups instead of a single box around both. This greatly simplifies downstream data processing.

Boxes around tri-columnar groups is also likely to yield output that does not accurately reflect the exact columnar divides, and may result in output from Tabula that includes "extra" columns. For Tabula unruly tri-columnar output, it may be advantageous to regroup by two columns twice, and match according to the common column later to reconstruct the true tri-columnar output.
