
## Calculating accuracy here in the same file.
import glob, os
import ast
import numpy as np
from statistics import mean

from Preprocess_ARC import Preprocess_Arc, Preprocess_KB_sentences, Write_ARC_KB, get_IDF_weights, Query_boosting_sent

def evals(scores, candidates, Correct_ans, outfile1, write1 = 0):
    if write1 == 1:
        outfile = open(outfile1,"w")
    ind_score=[]
    All_score=[]
    Correct_predicted_ques=[]
    Accuracy = 0
    score_index=0
    print (len(candidates))
    new_line = ""
    for cindex,cand1 in enumerate(candidates):
        num_options=len(cand1)
        upper_limit=score_index + num_options
        while score_index < upper_limit:
              # print (cand1, score_index, scores[score_index])

              final_score=0

              ind_score.append((scores[score_index][1]))
              # ind_score.append(mean(scores[score_index]))

              # if scores[score_index].index(max(scores[score_index]))==1:
              #    # ind_score.append(max(scores[score_index]))
              #    ind_score.append((scores[score_index][1]))
              # else:
              #
              #    ind_score.append((scores[score_index][1]))
              #    # ind_score.append(max(scores[score_index]))
              #    # ind_score.append(0)

              # ind_score.append(max(scores[score_index]))
              new_line+= " " + str(final_score)
              # All_score.append(sum(scores[score_index]))
              score_index+=1
        ind_score=np.asarray(ind_score)

        # Predicted_ans.append(np.argmax(ind_score))
        new_line+= "\t" + str(Correct_ans[cindex]) + "\n"
        if write1==1:
           outfile.write(new_line)
        new_line=""
        if Correct_ans[cindex]==np.argmax(ind_score):
           Accuracy+=1
           Correct_predicted_ques.append(cindex)
        ind_score = []
    print(cindex)
    return (Accuracy/float(cindex)), Correct_predicted_ques

def rearranged(orig_file, new_file):
    orig_lines = open(orig_file,"r")
    sequence1 = {}
    for line in orig_lines:
        data1 = line.strip().split("\t")
        sequence1.update({data1[1]:data1[0]})
    out_file = open(new_file,"w")

    for ind1 in range(len(sequence1)):
        out_file.write(sequence1[str(ind1)] + "\n")

def rearranged_CLARK(orig_file, new_file, candidates):
    orig_lines = open(orig_file,"r")
    sequence1 = {}
    for line in orig_lines:
        data1 = line.strip().split("\t")
        sequence1.update({data1[1]:data1[0]})
    out_file = open(new_file,"w")

    for ind1, ques_cand in enumerate(candidates):
        for cind1, cand1 in enumerate(ques_cand):
            if (str(ind1)+"_"+str(cind1)) in sequence1.keys():
               out_file.write(sequence1[(str(ind1)+"_"+str(cind1))] + "\n")
            else:
               out_file.write(str(np.array([-100,-100]))+"\n")

def calculate_overlap(All_correct_quest_list):
    list1 = All_correct_quest_list[0]
    list2 = All_correct_quest_list[1]
    overlap_val = (len(set(list1).intersection(set(list2)))/float(len(list2)))
    # overlap_val = (len(set(list1).intersection(set(list2)))/float(len(set(list1).union(set(list2)))))
    print ("overlap is around ",overlap_val)
    return overlap_val
def sort_files_WRT_epochs(All_file_names):
    epochs = []
    for file_name in All_file_names:
        epochs.append(int( file_name.split("_")[-1].split(".")[0]))
    sorted_indices = np.argsort(np.array(epochs))

    sorted_files = []
    for sind1 in sorted_indices:
        sorted_files.append(All_file_names[sind1])

    return sorted_files



# ARC_file = "/Users/vikasy/SEM_5/ARC_Challenge_FLAIR/ARC_corpus/ARC-Easy/ARC-Easy-Test.csv"
# ARC_file = "/Users/vikasy/SEM_5/ARC_Challenge_FLAIR/ARC_corpus/ARC-Challenge/ARC-Challenge-Test.csv"
ARC_file = "/Users/vikasy/SEM_5/ARC_Challenge_FLAIR/ARC_corpus/ARC-Challenge/ARC-Challenge-Train_dev.csv"
cols_sizes, questions, candidates, algebra, All_words, correct_ans, negative_ques=Preprocess_Arc("ARC",ARC_file).preprocess()

# pred_file = "Only_Easy_perf_1JUST_Balanced_SIGIR.txt"
# pred_files = ["CHECK_Easy_perf_1JUST_CLARK_P2.txt", "CHECK_Easy_perf_1JUST_CLARK_P2_20Epoch.txt"]
# files_directory = "EASY_P2_epoch10_150/"
files_directory = "Challenge_P2_P3_P4/Challenge_P2_epoch10_150_train/"
all_pred_files1 = os.listdir(files_directory)
all_pred_files = sort_files_WRT_epochs(all_pred_files1)
All_correct_question_set = []
for pfileind, pred_file in enumerate(all_pred_files):
    new_rearranged_file = "Challenge_P2_P3_P4/train_.txt"
    rearranged_CLARK(files_directory+pred_file, new_rearranged_file, candidates)
    new_arranged_pred = open(new_rearranged_file,"r")
    Score_matrix = []
    len_mat = []
    for line in new_arranged_pred:
        words = line.split("[")[1].split("]")[0].split()
        Score_matrix.append([float(words[0]), float(words[1])])
    Ensemble_file = "abc.txt"
    Accuracy, correct_question_set=evals(Score_matrix, candidates, correct_ans, Ensemble_file,0)
    All_correct_question_set.append(correct_question_set)
    print(Accuracy, "for pred file, ", pred_file)
    if len(All_correct_question_set)==2:
       overlap_score = calculate_overlap(All_correct_question_set)
       All_correct_question_set = All_correct_question_set[1:]