import ast, os
import numpy as np
import math

from Preprocess_ARC import Preprocess_Arc, Preprocess_KB_sentences, Write_ARC_KB, get_IDF_weights, Query_boosting_sent
file_set = "train_dev"
portion = "Challenge"
P_val = 3 ## is you want to filter candidates based on P_vals
if portion == "Challenge":
    if file_set == "test":
        justification_file = "/Users/vikasy/SEM_5/ARC_Challenge_BM25/Challenge_BIDAF_test_3_60_explanations_BM25.txt"
        ranking_file = "/Users/vikasy/SEM_5/ARC_Challenge_BM25/Challenge_test_ranking_FLAIR.txt"
        # justification_file = "/Users/vikasy/SEM_5/ARC_CHALLENGE_Clark_Just/ARC-Challenge-Test_with_hits_default_8.jsonl"
        ARC_file = "/Users/vikasy/SEM_5/ARC_Challenge_BM25/ARC_corpus/ARC-Challenge/ARC-Challenge-Test.csv"
        Out_file = "Challenge_P2_P3_P4/ARC_Challenge_test_P3.tsv"

    elif file_set == "train_dev":
        justification_file = "/Users/vikasy/SEM_5/ARC_Challenge_BM25/Challenge_BIDAF_train_dev_3_60_explanations_BM25.txt"
        ranking_file = "/Users/vikasy/SEM_5/ARC_Challenge_BM25/Challenge_train_dev_ranking_FLAIR.txt"
        ARC_file = "/Users/vikasy/SEM_5/ARC_Challenge_BM25/ARC_corpus/ARC-Challenge/ARC-Challenge-Train_dev.csv"
        Out_file = "Challenge_P2_P3_P4/ARC_Challenge_train_dev_P3.tsv"

elif portion=="Easy":
    if file_set == "test":
        justification_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/ARC-Easy-Test-Clark-Justifications.txt"
        ranking_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/Easy_test_ranking.txt"
        # justification_file = "/Users/vikasy/SEM_5/ARC_EASY/SIGIR_easy_test_justification_4.txt"
        ARC_file = "/Users/vikasy/SEM_5/ARC_EASY/ARC_corpus/ARC-Easy/ARC-Easy-Test.csv"
        Out_file = "EASY_ARC_test_Quora_CLARK_P3.tsv"

    elif file_set == "train_dev":
        justification_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/ARC-Easy-Train_dev-Clark-Justifications.txt"
        ranking_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/Easy_train_dev_ranking.txt"
        # justification_file = "/Users/vikasy/SEM_5/ARC_EASY/SIGIR_easy_train_dev_justification_4.txt"
        ARC_file = "/Users/vikasy/SEM_5/ARC_EASY/ARC_corpus/ARC-Easy/ARC-Easy-Train_dev.csv"
        Out_file = "EASY_ARC_train_dev_Quora_CLARK_P3.tsv"
    elif file_set == "train":
        justification_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/ARC-Easy-Train-Clark-Justifications.txt"
        ranking_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/Easy_train_ranking.txt"
        # justification_file = "/Users/vikasy/SEM_5/ARC_EASY/SIGIR_easy_train_dev_justification_4.txt"
        ARC_file = "/Users/vikasy/SEM_5/ARC_EASY/ARC_corpus/ARC-Easy/ARC-Easy-Train.csv"
        Out_file = "EASY_ARC_JUST_train_Quora_CLARK_P3.tsv"
    elif file_set == "dev":
        justification_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/ARC-Easy-Dev-Clark-Justifications.txt"
        ranking_file = "/Users/vikasy/SEM_5/ARC_EASY_Clark_Just/Easy_dev_ranking.txt"
        # justification_file = "/Users/vikasy/SEM_5/ARC_EASY/SIGIR_easy_train_dev_justification_4.txt"
        ARC_file = "/Users/vikasy/SEM_5/ARC_EASY/ARC_corpus/ARC-Easy/ARC-Easy-Dev.csv"
        Out_file = "EASY_ARC_JUST_dev_Quora_CLARK_P3.tsv"


cols_sizes, questions, candidates, algebra, All_words, correct_ans, negative_ques=Preprocess_Arc("ARC",ARC_file).preprocess()

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






