from Feedforward_PMI_multi_class import feedforward_keras
from numpy import array
import ast

def normalize_values(All_x):
    normalized_All_x = []
    for ques_data in All_x:
        sorted_index = sorted(range(len(ques_data)), key=lambda k: ques_data[k])

        new_ques_data = [0 for i in range(len(ques_data))]
        for curr_ind, sorted_ind1 in enumerate(sorted_index):
            if curr_ind==0:
               new_ques_data[sorted_ind1]=0
            elif curr_ind ==1:
               new_ques_data[sorted_ind1]=0.25
            elif curr_ind ==2:
               new_ques_data[sorted_ind1] = 0.5
            elif curr_ind ==3:
               new_ques_data[sorted_ind1] = 1
            elif curr_ind ==4:
               new_ques_data[sorted_ind1] = 1.25
            else:
               print("this case shouldnt be there, there is some error: ")
        # new_ques_data+=ques_data
        normalized_All_x.append(new_ques_data)
    return normalized_All_x

def steve_difference_feature(All_x):
    diff_feature = []

    for ques_data in All_x:
        max_val = max(ques_data)
        diff_data = [max_val-val for val in ques_data]

        diff_feature.append(diff_data)
    return diff_feature



def Merge_features(data1, data2):
    for ind, data_point in enumerate(data1):
        data1[ind] += data2[ind]

    return data1

def get_features_labels(file):
    ques_label = []
    Final_labels = []
    data = []
    data_ques = []
    for line in file:
        features = line.strip().split("\t")[0]
        labels = line.strip().split("\t")[1]
        ind_features = features.split()
        for f1 in ind_features:
            data_ques.append(float(f1))

        while len(data_ques)<5:
            data_ques.append(float(-10))

        data.append(data_ques)
        data_ques = []

        labels = labels.split()

        Final_labels.append(int(labels[0]))
    new_data1 = normalize_values(data)
    new_data2 = steve_difference_feature(data)

    # final_new_data = Merge_features(new_data1, new_data2)  ## adding ranking and steve difference suggestion
    # final_new_data = Merge_features(new_data1, data)  #### adding original data
    return new_data1, Final_labels

def ranked_normalized_value(All_x):
    pass

def get_ranked_features(file):

    data = []
    data_ques = [0 for i in range(5)]
    for line in file:
        ind_features = ast.literal_eval(line)
        while len(ind_features)<5: ## then we need to pad
            ind_features = [len(ind_features)] + ind_features

        for f1, val1 in enumerate(ind_features):
            if f1 == 0: ## ranked 4th
               data_ques[val1] = 0
            elif f1 == 1:
               data_ques[val1]=0.25
            elif f1 == 2:
               data_ques[val1] = 0.50
            elif f1 == 3:
               data_ques[val1] = 1
            elif f1 == 4:
               data_ques[val1] = 1.25


        data.append(data_ques)
        # data.append(ind_features)
        data_ques = [0 for i in range(5)]


    new_data1 = data
    new_data2 = steve_difference_feature(data)

    # final_new_data = Merge_features(new_data1, new_data2)  ## adding ranking and steve difference suggestion
    # final_new_data1 = Merge_features(final_new_data, data)  #### adding original data
    return new_data1

########### Train Files
train_file = open("Challenge_P2_P3_P4/BIMPM_P2_train_dev_predictions.txt","r")
train_data_PN, train_label = get_features_labels(train_file)

train_file = open("Challenge_P2_P3_P4/BIMPM_P3_train_dev_predictions.txt","r")
train_data_P3, dummy1 = get_features_labels(train_file)


train_file = open("/Users/vikasy/SEM_5/ARC_Challenge_BM25/Challenge_train_dev_ranking_FLAIR.txt","r")
train_data_PMI = get_ranked_features(train_file)

train_data1 = Merge_features(train_data_PN,train_data_P3)
train_data = Merge_features(train_data1, train_data_PMI)

############ Dev Files


############# Test files

test_file = open("Challenge_P2_P3_P4/BIMPM_P2_test_predictions.txt","r")
test_data_PN, test_label = get_features_labels(test_file)

test_file = open("Challenge_P2_P3_P4/BIMPM_P3_test_predictions.txt","r")
test_data_P3, dummy1 = get_features_labels(test_file)

test_file = open("/Users/vikasy/SEM_5/ARC_Challenge_BM25/Challenge_test_ranking_FLAIR.txt","r")

test_data_PMI = get_ranked_features(test_file)

print ("test data PMI looks like ", test_data_PMI[0:10])

test_data1 = Merge_features(test_data_PN, test_data_P3)
test_data = Merge_features(test_data1, test_data_PMI)

############

print("train data is: ", len(train_data), len(train_data[1]))
print("test data is: ", test_data[0:10])
print("train data is: ", train_data[0:10])

predictions = feedforward_keras(array(train_data), train_label, array(test_data), test_label,  0, test_label)

print(predictions)

