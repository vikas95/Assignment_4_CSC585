
# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import numpy
from numpy import array
import keras
# fix random seed for reproducibility
numpy.random.seed(7)

def multi_class(data, labels):
    multiclass_label = []

    for lind, label in enumerate(labels):
        dum_lab = [0 for i in range(5)]
        dum_lab[label] = 1

        multiclass_label.append(dum_lab)


    Y = array(multiclass_label)
    return Y

def normalize_values(All_x):
    All_x = All_x.tolist()

    normalized_All_x = []
    for ques_data in All_x:
        new_ques_data = [cand_score/float(sum(ques_data)) for cand_score in ques_data]
        normalized_All_x.append(new_ques_data)
    return array(normalized_All_x)


def feedforward_keras(data, Y, data_test, Y_test, dummy, Correct_ans): # , number_epochs):
    print ("max label is ", max(Y), len(Y))
    Y = multi_class(data, Y)
    Y_test = multi_class(data_test, Y_test)
    print(Y)
    print ("Y len is:  ", len(Y), len(Y[1]))


    number_epochs = 400

    print("Input length is: ", data[0].size)
    model = Sequential()

    model.add(Dense(32, input_dim=data[0].size, activation="sigmoid"))

    # model.add(batch_norm)

    # model.add(Dense(10, activation="relu"))

    model.add(Dense(5, activation='softmax'))

    # sgd = optimizers.SGD(decay=0.000001)
    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])
    # Fit the model
    model.fit(data, Y, epochs=number_epochs, batch_size=10, validation_data = (data_test, Y_test))  # validation_split=0.5
    # evaluate the model
    scores = model.evaluate(data_test, Y_test)
    predictions1 = model.predict(data_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    for layer in model.layers:
        weight = layer.get_weights()
        # print ("the weight vector is: ", weight)


    Question_accuracy=0

    print ("predictions are: ", predictions1)
    if len(predictions1) == len(Correct_ans):
        print ("yes, we can calculate question level accuracy")

        for ind, pred1 in enumerate(predictions1):
            predicted_val = numpy.argmax(pred1)
            if int(predicted_val) == int(Correct_ans[ind]):

               Question_accuracy+=1
    else:
        print ("something is wrong in dimensions, please cross check ")
        print(len(predictions1), len(Correct_ans))


    print ("The final question number is: ", Question_accuracy )
    return (Question_accuracy)/ float( len(Correct_ans))




