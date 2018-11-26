import json

dataset = "Challenge"  ## "Challenge"

if dataset == "Easy":
    json_data=open("ARC-Easy/ARC-Easy-Train_just.jsonl","r")
    json_data_write=open("ARC-Easy/ARC-Easy-Train_Good_just.jsonl","w")
    json_data_dev = open("ARC-Easy/ARC-Easy-Dev_just.jsonl", "r")
    json_data_write_dev = open("ARC-Easy/ARC-Easy-Dev_Good_just.jsonl", "w")
    # justification_file = open("/Users/vikasy/SEM_5/ARC_multi-hop/Multihop_optimization_Easy/Easy_train_dev_Good_justifications.txt", "r")
    # justification_file = open("/Users/vikasy/SEM_5/ARC_multi-hop/Multihop_optimization_Easy/Easy_train_Dev_Cand_3_60_explanations_BM25.txt", "r")
    justification_file = open("/Users/vikasy/SEM_5/ARC_EASY/SIGIR_easy_train_dev_justification.txt", "r")

else:
    json_data = open("ARC-Challenge/ARC-Challenge-Train_just.jsonl", "r")
    json_data_write = open("ARC-Challenge/ARC-Challenge-Train_Good_just.jsonl", "w")
    json_data_dev=open("ARC-Challenge/ARC-Challenge-Dev_just.jsonl","r")
    json_data_write_dev=open("ARC-Challenge/ARC-Challenge-Dev_Good_just.jsonl","w")
    justification_file = open("/Users/vikasy/SEM_5/ARC_Challenge_BM25/SIGIR_Challenge_train_dev_justification.txt","r")

    # justification_file = open("/Users/vikasy/SEM_5/ARC_Challenge_BM25/Challenge_train_Dev_Cand_3_60_explanations_BM25.txt","r")

Good_justifications = []
for line in justification_file:
    all_just = line.strip().split("\t")
    good_just1 = " ".join(all_just[0].split()[1:])
    # good_just1 += " " + " ".join(all_just[1].split()[1:])

    Good_justifications.append(good_just1)


print(len(Good_justifications))
####### for train
count =0
tot_len = 0
for line in json_data:

    # print(count)
    data_dict = json.loads(line)

    tot_len+=len(data_dict["question"]["choices"])
    for choice_text in data_dict["question"]["choices"]:
        choice_text["text"] += " " + Good_justifications[count]
        count+=1

    json.dump(data_dict, json_data_write)
    json_data_write.write('\n')


###### for DEV
for line in json_data_dev:

    # print(count)
    data_dict = json.loads(line)

    tot_len+=len(data_dict["question"]["choices"])
    for choice_text in data_dict["question"]["choices"]:
        # print(count)
        choice_text["text"] += " " + Good_justifications[count]
        count+=1

    json.dump(data_dict, json_data_write_dev)
    json_data_write_dev.write('\n')



print("total len is: ", tot_len)

