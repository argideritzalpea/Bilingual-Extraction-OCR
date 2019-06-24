import os
import sys
import pandas as pd
import argparse
from jiwer import wer
import numpy as np
import pprint
from CharacTER import cer

"""
ap.add_argument("-e", "--output", required=True, help="path to extraction pipeline output")
ap.add_argument("-x", "--gold_data", required=True, help="path to gold data")
ap.add_argument("-o", "--log", required=True, help="results summary log")
"""

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)

output_data = pd.read_table("../../example_output/Mende_all_V2.txt", header=None)
output_data.columns = ["Mende", "English", "Top", "Page"]
output_data["Matched"] = False
output_data

gold_data = pd.read_excel("../../gold_data/Mende_gold.xlsx", encoding="utf-8")
gold_data["checked"] = False
gold_data.columns

gold_data[gold_data["page"] == 6]


grouped_df = output_data.groupby('Page')
gb = grouped_df.groups

analysis_frame = pd.concat([gold_data["english"], gold_data["mende"], gold_data["checked"]], axis=1, keys=["English", "Mende", "WER", "Checked"])
analysis_frame["WER_English"] = 0.0
analysis_frame["WER_Mende"] = 0.0
analysis_frame["CER_English"] = 0.0
analysis_frame["CER_Mende"] = 0.0
analysis_frame["Matched"] = False
analysis_frame["English_output"] = None
analysis_frame["Mende_output"] = None

gold_data['page'].nunique()

for key, values in gb.items():
    page_group = output_data.iloc[values]
    line_groups = page_group.groupby("Top")
    lines = line_groups.groups
    for line_key, vals in lines.items():
        line_vals = output_data.iloc[vals]
        line_from_top = line_vals.iloc[0]["Top"]
        upper_b = int(line_from_top) + 35
        lower_b = int(line_from_top) - 35
        page_num = str(key.split("-")[1])
        gold_page = gold_data.where(gold_data["page"] == int(page_num))
        gold_matches = gold_page["pixel_height"].between(lower_b, upper_b, inclusive=False)
        if gold_matches.empty:
            print(line_vals.values)
        gold_items = gold_page.loc[gold_matches]
        output_items = line_vals
        if len(gold_items) != 0:
            hey = gold_items
            boo = output_items
            for j, k in zip(gold_items.iterrows(), output_items.iterrows()):
                output_data.at[k[0],"Matched"] = True
                gold_target = str(j[1]["mende"])
                output_target = str(k[1]["Mende"])
                target_WER = wer(gold_target, output_target)
                target_CER = cer(gold_target, output_target)
                analysis_frame.at[j[0],"WER_Mende"] = target_WER
                analysis_frame.at[j[0],"CER_Mende"] = target_CER
                gold_eng = str(j[1]["english"])
                output_eng = str(k[1]["English"])
                eng_WER = wer(gold_eng, output_eng)
                eng_CER = cer(gold_eng, output_eng)
                analysis_frame.at[j[0],"WER_English"] = eng_WER
                analysis_frame.at[j[0],"CER_English"] = eng_CER
                analysis_frame.at[j[0],"Matched"] = True
                analysis_frame.at[j[0],"English_output"] = output_eng
                analysis_frame.at[j[0],"Mende_output"] = output_target

analysis_frame

# Calculate recall
matched = np.sum(analysis_frame["Matched"])
matched
total = len(analysis_frame.index)
recall = matched/total
recall

# Calculate precision
matched = np.sum(output_data["Matched"])
total = len(output_data.index)
total
precision = matched/total
precision

# Calculate F1
F1_score = 2*(recall*precision)/(precision+recall)
F1_score


# Reformat WER to not penalize for
analysis_frame["WER_English_norm"] = analysis_frame["WER_English"]
analysis_frame["WER_Mende_norm"] = analysis_frame["WER_Mende"]
analysis_frame.loc[analysis_frame.WER_English > 1, "WER_English_norm"] = 1
analysis_frame.loc[analysis_frame.WER_Mende > 1, "WER_Mende_norm"] = 1

# Calculate average word error rate
analysis_frame.groupby("Matched").mean().WER_English_norm
analysis_frame.groupby("Matched").mean().WER_Mende_norm

# Calculate average character error rate
analysis_frame.groupby("Matched").mean().CER_English
analysis_frame.groupby("Matched").mean().CER_Mende
