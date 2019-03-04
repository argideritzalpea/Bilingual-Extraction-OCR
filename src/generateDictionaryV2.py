import argparse
import itertools
import re
import operator
import pandas as pd
from string import punctuation, printable
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
from collections import Counter
import pickle
import numpy as np
from collections import OrderedDict, defaultdict
from extract import getModeSplits, mode
from sklearn.cluster import MeanShift, estimate_bandwidth
import traceback
import os
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--in_dir", required=True, help="directory path of input to be OCR'ed")
ap.add_argument("-o", "--out_dict", required=True, help="name / path of output dictionary")
ap.add_argument("-s", "--skip", required=False, nargs='*', help="pages to skip when generating the dictionary")
ap.add_argument("-b", "--bandwidth", required=False, help="determine bandwidth quintile setting for the sklearn.cluster.MeanShift algorithm. values between .02 and .03 are good. some pages fail with different values")
ap.add_argument("-l", "--source_lang", required=False, default="en", help="Select the source language (default is English)")
ap.add_argument("-t", "--target_lang", required=False, help="Select the target language or orthographic relative to give hints about characters for postprocessing")
args = vars(ap.parse_args())

in_dir = args["in_dir"]
out_dict_arg = args["out_dict"]
skip_args = args["skip"]
bandwidth_q = args["bandwidth"]
source_lang = args["source_lang"]
target_lang = args["target_lang"]

skips = set()
if skip_args != None:
    try:
        for num_range in skip_args:
            #print(num_range)
            if re.match("^\d*$",num_range):
                print(num_range)
                skips.add(num_range)
            elif re.match("^(\d*?)\-(\d*?)$",num_range):
                ranged = [int(x) for x in num_range.split("-")]
                for page in range(ranged[0], ranged[1] + 1):
                    skips.add(str(page))
        print("Skipping the following pages: ", skips)
    except:
        print("Error in processing skip args. Pass in arguments as single numbers or hyphenated ranges separated by a space (i.e. '5, 6, 7, 8-10'). The numbers should reflect the file name indices of the pages in the input folder")
else:
    pass

def sortShorties(data):
    """TODO add language threshold to test for titles, add % for capitalized first letters"""
    """Delete from preprocessed table data anything that looks like a title"""
    for i in data:
        data[i] = list(itertools.filterfalse(lambda item: item[0] == 1 and item[1][0][0].isupper(), data[i]))

def deleteBlanks(data):
    """Delete from table data anything that has passed in as empty after cleaning"""
    for i in data:
        for delid, x in enumerate(data[i]):
            if x[1] == []:
                del data[i][delid]

def chunks(l, n):
    """Yield successive n-sized chunks from predicted columns."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class DictionaryFactory:
    """
    Defines a class that processes HOCR and other data output from Tesseract 4.0 using..
    """
    def __init__(self, input_df, basefilename, mean_shift_thresh=bandwidth_q):
        """
        Initializes the factory.
        :input_pkl: pkl file that contains the hocr data in pandas df format
        :mean_shift_thresh: define the quantile parameter to use with the mean shift function
        """
        if mean_shift_thresh is None:
            mean_shift_thresh = .02
        else:
            mean_shift_thresh = float(mean_shift_thresh)
        self._raw_hocr = input_df
        #self._line2members = self._raw_hocr.groupby('line')['text'].apply(lambda g: g.values.tolist()).to_dict()
        self._c_lookup, self._col2members, self._members2col, self._c_lookup_val, self._line2heights = cluster1d(input_df, "left", "column", mean_shift_thresh)
        self._l_lookup, self._line2members, self._members2line, self._l_lookup_val, self._line2heights = cluster1d(input_df, "top", "line", mean_shift_thresh)
        #self._c_lookup, self._col2members, self._members2col, self._c_lookup_val, self._line2heights = cluster1d(input_df, "left", "column", mean_shift_thresh)
        #self._lines = [Line(self, x) for x in self._line2members]
        #self._column = [Column(self, x) for x in self._col2members]
        #self._line_heights = self._raw_hocr.groupby(['line_num'], as_index=False).mean().groupby('line_num')['top'].mean()
        self._basefilename = basefilename

    def countByCol(self):
        data = self._col2members
        dict = Counter()
        for cluster in data:
            dict[cluster] = len(data[cluster])
        #print(pd.DataFrame.from_dict(dict, orient="index"))
        return dict

    #def createLine(self):
    #    print([y._data for y in self._lines[1]._members])

    def removeBeginningNAs(self):
        """ Remove HOCR that contains no text data """
        self._raw_hocr = self._raw_hocr.reset_index(drop=True)
        self._raw_hocr = self._raw_hocr[self._raw_hocr['conf'] != -1]

    def removeAllPunct(self):
        """ Remove any data that contains only punctuation """
        self._raw_hocr = self._raw_hocr.reset_index(drop=True)
        self._raw_hocr = self._raw_hocr[~self._raw_hocr['text'].str.contains("^[^\w]+$")]

    def removeListMarker(self):
        """ Remove data that ends with colons or semicolons (indication of list or non-translation material) """
        self._raw_hocr = self._raw_hocr.reset_index(drop=True)
        self._raw_hocr = self._raw_hocr[~self._raw_hocr['text'].str.contains(".*[:;]$")]

    def removeParens(self):
        """ Remove values that entirely enclosed by parenthesis """
        self._raw_hocr = self._raw_hocr.reset_index(drop=True)
        self._raw_hocr = self._raw_hocr[~self._raw_hocr['text'].contains("\(.*?\)")]

    def printHOCR(self):
        """ Prints the HOCR in a pandas dataFrame format """
        print(self._raw_hocr)

    def assign_Hocr_Line(self, data):
        """
        This funtion attempts to assign "table" output from extract.py to
        lines predicted by a clustering algorithm. Alternatively, the lines
        can be assigned to the lines determined by HOCR output.
        """
        self._oldline2members = self._line2members
        linedict = self._line2members
        new_line2members = OrderedDict()
        for para in data:
            for line in data[para]:
                tokens = " ".join(line[1]).split()
                scores = {}
                for line_hocr in linedict:
                    tot = self._raw_hocr[self._raw_hocr["line"] == line_hocr].shape[0]
                    match = 0
                    for tok in tokens:
                        if tok in linedict[line_hocr]:
                            match += 1
                    try:
                        scores[line_hocr] = match/tot
                    except Exception as e:
                        #print("ERROR in assigning HOCR lines:")
                        #print("Line number: ", line, " : ", "line hocr ", line_hocr, " : ", tokens)
                        scores[line_hocr] = 0
                max_line_num = max(scores.items(), key=operator.itemgetter(1))[0]
                line.append(max_line_num)
                new_line2members[max_line_num] = line[1]
        self._line2members = new_line2members
        
        #print(new_line2members)

        return new_line2members

    def findLikelyColumns(self):
        """
        This attempts to find the likely column locations that have been assigned a 
        number in a clustering algorithm to each "stanza" from the table data output by
        extract.py. It looks up text in the HOCR and guesses the corresponding column
        by finding the left-right cluster value in the HOCR pandas dataFrame of the 
        first word in a phrase in a table format
        """
        linedict = self._line2members
        likelycolumns = {}
        for line in linedict:
            likelycolumns.setdefault(line, [])
            for phrase in linedict[line]:
                #print(phrase)
                col = self.get_wLeft(phrase.split(' ')[0], line)
                #print(type(col))
                likelycolumns[line].append(col)
        self._line2column_pertinence = likelycolumns

        return likelycolumns

    def get_wLeft(self, word, line):
        """
        Helper function to search a particular word's "column" cluster in the pandas dataframe
        """
        hocr = self._raw_hocr
        sample = hocr.loc[hocr['line']==line,:]
        #print("")
        #print(line)
        #print("")
        #print(sample)
        #print(sample.index)
        #print(word, indices)
        try:
            w_Left = sample.loc[sample['text'].str.contains(word, na=False), "column"].values[0]
        except:
            w_Left = None
        return w_Left

    def findTableBlocks(self):
        """
        This function analyzes the "stanzas" found in prior steps and their column clusters,
        looks for those lines with similar custers, and guesses a range of lines that have
        X columns at Y locations.
        TODO: Further testing to improve recall.
        """
        likelycolumns = self._line2column_pertinence
        freq_counter = Counter()
        index_dict = {}
        for linenum in likelycolumns:
            likely = tuple(set(likelycolumns[linenum]))
            index_dict.setdefault(likely, [linenum, None])
            freq_counter[likely] += 1
            if likely in index_dict:
                index_dict[likely][1] = linenum
        combined_ranges = dict()
        for likely, indices in index_dict.items():
            #print(likely, indices)
            x = list(range(min(indices), max(indices)+1))
            for likely2, indices2 in index_dict.copy().items():
                y = list(range(min(indices2), max(indices2)+1))
                #print(likely2, "hey")
                if set(x).intersection(set(y)) != set():
                    inter = set(x).intersection(y)
                    maximum = max(x+y)
                    minimum = min(x+y)
                    x = [minimum, maximum]
                    #print(x)
                combined_ranges[likely] = x
            print("")


        unique_ranges = set(tuple(val) for val in combined_ranges.values())
        final_column_dict = {}
        for un_rang in unique_ranges:
            final_column_dict[un_rang] = [k for k,v in combined_ranges.items() if (v == list(un_rang) and k != None)]

        def returnHighestDict(dict):
            new = {}
            for range, y in dict.items():
                new.setdefault(range, None)
                highest = y[0]
                for tuple in y.copy():
                    if len(tuple) % 2 == 0 and freq_counter[tuple] > freq_counter[highest]:
                        highest = tuple.sort()
                new[range] = highest
            return new

        highestReturnDict = returnHighestDict(final_column_dict)

        def testSubsets(x, y, copy):
            #print(x, y, copy)
            val = False
            add = None
            for g, w in copy.items():
                if set(y).issubset(set(w)):
                    val = True
                elif set(list(range(x[0], x[1]+1))).issubset(set(list(range(g[0], g[1]+1)))):
                    val = True
            return val, add

        for f, g in highestReturnDict.copy().items():
            subs = highestReturnDict.copy()
            del subs[f]
            subsetmatch, append = testSubsets(f, g, subs)
            if subsetmatch == True:
                del highestReturnDict[f]
            else:
                try:
                    highestReturnDict[f] = tuple(sorted(g))
                except:
                    print("Couldn't add a NoneType")

        self._highestReturnDict = highestReturnDict

        return highestReturnDict

    def returnDictionaryItems(self):
        """
        Attempts to pull all data from the HOCR that aligns with predictions of column locations found
        in previous functions. Recursive calls to combineMultiColumn descend lines according to a pre-
        defined logic. It is currently set to look for lines whose first letter is capitalized and are
        followed by a line that has a first letter that is lowercase.
        """
        outdict = {}
        data = self._raw_hocr
        rangedict = self._highestReturnDict

        def combineMultiColumn(first, second, deletion_index, firstcol, secondcol, stopcolumn, group_id):
            seconddelete = False
            print("deletion coming in: ", deletion_index)
            deletion_index += 1
            print("deletion current: ", deletion_index)
            if deletion_index in self._line2members:
                n_first = data.loc[(data['line'] == deletion_index) & (data['column'] >= firstcol) & (data['column'] < secondcol), 'text'].str.cat(sep=' ')
                if stopcol != False:
                    n_second = data.loc[(data['line'] == deletion_index) & (data['column'] >= secondcol) & (data['column'] <= stopcol), 'text'].str.cat(sep=' ')
                else:
                    n_second = data.loc[(data['line'] == deletion_index) & (data['column'] >= secondcol), 'text'].str.cat(sep=' ')
                if len(first) == 0:
                    first = " "
                if len(second) == 0:
                    second = " "
                if len(n_first) == 0:
                    n_first = " "
                if len(n_second) == 0:
                    n_second = " "
                print(first, n_first)
                print(second, n_second)
                if first[0].isupper() and n_first[0].islower():
                    combined_first = first + " " + n_first
                    if second[0].isupper() and n_second[0].islower():
                        combined_second = second + ' ' + n_second
                    else:
                        combined_second = second
                    print("At this stage : ", combined_first, combined_second)
                    combined_first, combined_second = combineMultiColumn(combined_first, combined_second, deletion_index, firstcol, secondcol, stopcol, group_id)
                    delete.add((deletion_index, group_id))
                    print("deleted added: ", deletion_index)
                #elif first[0].isupper() and (n_first == " " or n_first == ""):
                #    combined_first = first
                #    if second[0].isupper() and (n_second == " " or n_second == ""):
                #        
                else:
                    combined_first = first
                    if second[0].isupper() and n_second[0].islower():
                        combined_first = first + ' ' + n_first
                        combined_second = second + ' ' + n_second
                        print("At this stage : ", combined_first, combined_second)
                        combined_first, combined_second = combineMultiColumn(combined_first, combined_second, deletion_index, firstcol, secondcol, stopcol, group_id)
                        delete.add((deletion_index, group_id))
                        print("deleted added: ", deletion_index)
                    else:
                        combined_second = second
                        print("At this stage : ", combined_first, combined_second)
            else:
                print("Line index not found")
                combined_first, combined_second = first, second

            return combined_first, combined_second

        deletionList = set()
        delete = set()
        for linenum in self._line2column_pertinence:
            outdict.setdefault(linenum, {})
            #print("")
            #print(linenum)
            #print("RANGEDICT")
            #print(rangedict)
            for range in rangedict:
                if linenum >= range[0] and linenum <= range[1]:
                    #print(list(chunks(rangedict[range], 2)))
                    for idx, group in enumerate(list(chunks(rangedict[range], 2))):
                        ### Find upper limit if chunk not in last column table (group)
                        try:
                            #print("rangedict value: ", rangedict[range], "index: ", idx)
                            if idx == 0:
                                stopcol = rangedict[range][(idx+1) * 2] - 2
                            elif idx == len(list(chunks(rangedict[range], 2))):
                                stopcol = False
                            else:
                                stopcol = rangedict[range][(idx+1) * 2 + 1] - 2
                            #print("Obtained stopcol: ", stopcol)
                        except:
                            stopcol = False
                            #print("Couldn't get stopcol - going to the end")
                        try:
                            firstcol = group[0]
                            secondcol = group[1]
                            #print("got columns: ", firstcol, secondcol)
                        except:
                            
                            print("couldn't get columns")
                        try:
                            first = data.loc[(data['line'] == linenum) & (data['column'] >= firstcol) & (data['column'] < secondcol), 'text'].str.cat(sep=' ')#, self._raw_hocr['text']].str.join(' ')
                            if stopcol != False:
                                second = data.loc[(data['line'] == linenum) & (data['column'] >= secondcol) & (data['column'] <= stopcol - 2), 'text'].str.cat(sep=' ')
                            else:
                                second = data.loc[(data['line'] == linenum) & (data['column'] >= secondcol), 'text'].str.cat(sep=' ')
                            #print("Succeeded in getting column words: ", first, second)
                            #print("Trying to descend with:", first, second, linenum, firstcol, secondcol, stopcol, idx)
                            try:
                                #n_firstcol = data.loc[(data['line'] == linenum+1) & (data['column'] >= firstcol) & (data['column'] < secondcol), 'text'].str.cat(sep=' ')#, self._raw_hocr['text']].str.join(' ')
                                #n_secondcol = data.loc[(data['line'] == linenum+1) & (data['column'] >= secondcol), 'text'].str.cat(sep=' ')
                                #print("Descending...")
                                fir, sec = combineMultiColumn(first, second, linenum, firstcol, secondcol, stopcol, idx)
                                #print("Final output is : ", [fir, sec])
                                if fir != ' ' and sec != ' ':
                                    outdict[linenum].setdefault(idx, None)
                                    outdict[linenum][idx] = [fir, sec]
                            except Exception as e:
                                #print(e)
                                traceback.print_exc()
                                #print("Couldn't descend...")
                                if first != ' ' and second != ' ':
                                    #print("Final output is : ", [first, second])
                                    outdict[linenum].setdefault(idx, None)
                                    outdict[linenum][idx] = [first, second]
                        except:
                            print('Failed for line: ', linenum)

        for deletion_i in reversed(sorted(list(delete))):
            line = deletion_i[0]
            group = deletion_i[1]
            try:
                print("deleting: ", deletion_i, outdict[line])
                del outdict[line][group]
            except Exception as e:
                print("Couldn't delete ", deletion_i, " from delete list")
                print(e)


        #print(self._line_heights)
        finaldictionary = []
        for listicle in outdict:
            print(listicle, " : ", outdict[listicle])
            for group in outdict[listicle]:
                print(group, " : ", outdict[listicle][group])
                try:
                    first = outdict[listicle][group][0]
                    second = outdict[listicle][group][1]
                except:
                    pass
                if first != "" and second != "":
                    finaldictionary.append([first.strip(), second.strip(), self._line2heights[listicle], self._basefilename]) 
        
        self._pre_dictionary = finaldictionary
        #print("final_dictionary____")
        #print(finaldictionary)
        
        return finaldictionary


def cluster1d(data, pixel_array_name, new_col_name, quant):
    """
    This employs the MeanShift algorith from sklearn to find "clusters" from a vector. It is used
    to find the columns and lines.
    """
    data[new_col_name] = 999
    x = data[pixel_array_name]
    X = np.array(list(zip(x, np.zeros(len(x)))), dtype=np.float32)
    bandwidth = estimate_bandwidth(X, quantile=quant)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    cluster_centers = np.argsort(ms.cluster_centers_[:,0])

    newlabelindex = {}
    for index, index2 in zip(list(labels_unique), list(cluster_centers)):
        newlabelindex[index] = index2

    lookup = {}
    line2members = OrderedDict()
    members2line = OrderedDict()
    line2heights = OrderedDict()
    lookup_val = {}
    for k in range(n_clusters_):
        my_members = labels == newlabelindex[k]
        #print("cluster {0}: {1}".format(k, X[my_members, 0]))
        members = [v for v in hocr.loc[my_members, "text"]]
        heights = [v for v in hocr.loc[my_members, "top"]]
        line2members[k] = members
        line2heights[k] = np.median(heights)
        lookup[k] = my_members
        lookup_val[k] = X[my_members, 0]
        joined = " ".join([b for b in members if isinstance(b, str)])
        members2line[joined] = k
        #print(x.shape)
        data.loc[my_members, new_col_name] = k

    return lookup, line2members, members2line, lookup_val, line2heights

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)

newDictionary = []
fulfilled = 0
for f in sorted(os.listdir(in_dir)):
    basefilename = f.strip('.pkl')
    index = str()
    if f.endswith('.pkl'):
        name = re.sub("(\_modedict\.pkl|\_pd\_hocr\.pkl|\.pkl)", "", f)
        index = re.match('.*?([0-9]+)$',name)
    if f.endswith('.pkl') and not f.endswith('_modedict.pkl') and not f.endswith('_pd_hocr.pkl') and index.group(1) not in skips:
        with open(in_dir + '/' + f, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        with open(in_dir + '/' + basefilename + '_pd_hocr.pkl', 'rb') as hocrfile:
            hocr = pickle.load(hocrfile)
        print(f)
        
        try:
            sortShorties(content)
            deleteBlanks(content)
            #print(content)
            h = DictionaryFactory(hocr, basefilename)
            h.removeBeginningNAs()
            h.removeListMarker()
            h.removeAllPunct()
            h.countByCol()
            h.assign_Hocr_Line(content)
            h.findLikelyColumns()
            h.findTableBlocks()
            outdict = h.returnDictionaryItems()
            fulfilled += 1
            #h.printHOCR()
            #print(h._line2heights)
            for x in outdict:
                print(x)
            print('\n\n\n')
            newDictionary.extend(outdict)
        except:
            print(h._line2heights)
            print('didn"t work')
            traceback.print_exc()
        print('')
        print('')
        print(str(fulfilled) + ' have been fulfilled')


with open(out_dict_arg+'.csv', 'w', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file, dialect="excel-tab")
    for entry in newDictionary:
        writer.writerow(entry)
