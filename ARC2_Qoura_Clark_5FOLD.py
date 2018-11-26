import ast, os
import numpy as np
import math

from Preprocess_ARC import Preprocess_Arc, Preprocess_KB_sentences, Write_ARC_KB, get_IDF_weights, Query_boosting_sent
file_set = "train"
portion = "Easy"
P_val = 2 ## is you want to filter candidates based on P_vals
if portion == "Challenge":
    if file_set == "test":
        justification_file = "/Users/vikasy/SEM_5/ARC_CHALLENGE_Clark_Just/ARC-Challenge-Test_with_hits_default_8.jsonl"
        # justification_file = "/Users/vikasy/SEM_5/ARC_CHALLENGE_Clark_Just/ARC-Challenge-Test_with_hits_default_8.jsonl"
        ARC_file = "/Users/vikasy/SEM_5/ARC_Challenge_BM25/ARC_corpus/ARC-Challenge/ARC-Challenge-Test.csv"
        Out_file = "ARC_test_Quora.tsv"

    elif file_set == "train_dev":
        justification_file = "/Users/vikasy/SEM_5/ARC_Challenge_BM25/Challenge_BIDAF_train_dev_3_60_explanations_BM25.txt"
        ARC_file = "/Users/vikasy/SEM_5/ARC_Challenge_BM25/ARC_corpus/ARC-Challenge/ARC-Challenge-Train_dev.csv"
        Out_file = "ARC_train_dev_Quora.tsv"

elif portion=="Easy":
    if file_set == "test":
        justification_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/ARC-Easy-Test-Clark-Justifications.txt"
        ranking_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/Easy_test_ranking.txt"
        # justification_file = "/Users/vikasy/SEM_5/ARC_EASY/SIGIR_easy_test_justification_4.txt"
        ARC_file = "/Users/vikasy/SEM_5/ARC_EASY/ARC_corpus/ARC-Easy/ARC-Easy-Test.csv"
        Out_file = "EASY_ARC_test_Quora_CLARK_P2.tsv"

    elif file_set == "train_dev":
        justification_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/ARC-Easy-Train_dev-Clark-Justifications.txt"
        ranking_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/Easy_train_dev_ranking.txt"
        # justification_file = "/Users/vikasy/SEM_5/ARC_EASY/SIGIR_easy_train_dev_justification_4.txt"
        ARC_file = "/Users/vikasy/SEM_5/ARC_EASY/ARC_corpus/ARC-Easy/ARC-Easy-Train_dev.csv"
        Out_file = "EASY_ARC_train_dev_Quora_CLARK_P2.tsv"
    elif file_set == "train":
        justification_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/ARC-Easy-Train-Clark-Justifications.txt"
        ranking_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/Easy_train_ranking.txt"
        # justification_file = "/Users/vikasy/SEM_5/ARC_EASY/SIGIR_easy_train_dev_justification_4.txt"
        ARC_file = "/Users/vikasy/SEM_5/ARC_EASY/ARC_corpus/ARC-Easy/ARC-Easy-Train.csv"
        Out_file = "EASY_ARC_JUST_train_Quora_CLARK_P2.tsv"
    elif file_set == "dev":
        justification_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/ARC-Easy-Dev-Clark-Justifications.txt"
        ranking_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/Easy_dev_ranking.txt"
        # justification_file = "/Users/vikasy/SEM_5/ARC_EASY/SIGIR_easy_train_dev_justification_4.txt"
        ARC_file = "/Users/vikasy/SEM_5/ARC_EASY/ARC_corpus/ARC-Easy/ARC-Easy-Dev.csv"
        Out_file = "EASY_ARC_JUST_dev_Quora_CLARK_P2.tsv"


# cols_sizes, questions, candidates, algebra, All_words, correct_ans, negative_ques=Preprocess_Arc("ARC",ARC_file).preprocess()

"""
print (questions[0], candidates[0], len(questions), len(correct_ans), set(correct_ans))

All_justification_sets = open(justification_file,"r").readlines()
ranked_list = open(ranking_file,"r").readlines()

num_of_justifications = 1

just_counter = 0

Cal_P_val = 4 - P_val  ## assuming there are 4 candidate answers per question
write_file = open(Out_file,"w")

for ind, ques in enumerate(questions):
    for cindex1, cand1 in enumerate(candidates[ind]):
        if cindex1 in ast.literal_eval(ranked_list[ind])[Cal_P_val:]:
            # print ("ok, this is working: ",cindex1,ast.literal_eval(ranked_list[ind])[Cal_P_val:])
            if len(All_justification_sets[just_counter].split("\t"))>2:
                if cindex1 == int(correct_ans[ind]):
                   new_line = str(1)
                else:
                   new_line = str(0)

                ques_ans_sent=questions[ind].lower().strip() + " " + str(cand1).lower().strip()
                new_line+= "\t" + ques_ans_sent

                justification_set = All_justification_sets[just_counter]
                just_sentences = ""
                for just1 in justification_set.split("\t")[2:2+num_of_justifications]: ## because first two tab values are blank in justification files.
                    just2 = " ".join(just1.split()[1:])
                    just_sentences += just2 + ". "

                new_line += "\t" + just_sentences
                new_line += "\t" + str(ind)+"_"+str(cindex1)
                write_file.write(new_line + "\n")
        just_counter+=1


print(just_counter)
"""

def write_fold_data(data1, out_file1):
    for d1 in data1:
        out_file1.write(d1)




data = open(Out_file,"r").readlines()
candidates = []
ques_cand = []
for line in data:
    if len(ques_cand)>0:
       if line.split("\t")[-1].split("_")[0]==ques_cand[-1].split("_")[0]:
          ques_cand.append(line.split("\t")[-1])
       else:
          candidates.append(ques_cand)
          ques_cand = []
          ques_cand.append(line.split("\t")[-1])
    else:
        ques_cand.append(line.split("\t")[-1])

candidates.append(ques_cand)
ques_cand = []


print ("ques len is: ", len(candidates))

folds = 5
interval = len(candidates) / float(folds)
interval = math.floor(interval)
total_performance = []

for i in range(folds):
    fold_train_out_file = open("5fold_CV/"+"train_"+str(i)+Out_file, "w")
    fold_test_out_file = open("5fold_CV/"+"test_"+str(i)+Out_file,"w")

    if i == 0:
        start = 0

        for cand12 in candidates[0:interval * (i + 1)]:
            start += len(cand12)
        train_data = data[start:]
        write_fold_data(train_data, fold_train_out_file)

        test_data = data[0:start]
        write_fold_data(test_data, fold_test_out_file)

        print("fold number is ", i, str(interval * (i + 1)))
    elif i == folds - 1:
        end = 0
        for cand12 in candidates[0:interval * (i)]:
            end += len(cand12)

        train_data = data[0:end]
        write_fold_data(train_data, fold_train_out_file)

        test_data = data[end:]
        print ("the len of final fold is ", len(test_data))
        write_fold_data(test_data, fold_test_out_file)

        print("fold number is ", i, str(interval * (i)))

    else:
        start = 0
        end = 0

        for cand12 in candidates[0:interval * (i + 1)]:
            start += len(cand12)

        for cand12 in candidates[0:interval * (i)]:
            end += len(cand12)

        train_data = np.concatenate((data[0:end], data[start:]), axis=0)
        write_fold_data(train_data, fold_train_out_file)

        test_data = data[end: start]
        write_fold_data(test_data, fold_test_out_file)

        print("fold number is ", i, str(interval * (i)), str(interval * (i + 1)))
