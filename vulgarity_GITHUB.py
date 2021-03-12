from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.datasets import imdb
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Bidirectional
from keras.layers import GRU
from SupervisedAlgorithm import  Performance
import os

os.environ["CUDA_VISIBLE_DEVICES"]='0'

dataset = "dvd"

def write_to_CSV(filename, data, labels, predictions):
    
    import csv
    
    csvfile=open(filename,'w', newline='')
    obj=csv.writer(csvfile)
    
    for element in zip(data, labels, predictions):
        obj.writerow(element)
        
    csvfile.close()
    

#------------- Model --------------
def model(X_train, X_test, y_train, y_test, num_of_words):
    

    #random_state = 10 for Cricket
   
    #print(X_train)
    #print(type(X_train))
    #return
    max_sentence_length = 200
    X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)

    from imblearn.over_sampling import SMOTE

    #sampler = SMOTE()
    #X_train, y_train = sampler.fit_sample(X_train, y_train)


    model = Sequential()
    model.add(Embedding(num_of_words + 1, 64, input_length=max_sentence_length))
    model.add(Dropout(0.2))
    #model.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2))
    model.add(Bidirectional(LSTM(300, dropout=0.2, recurrent_dropout=0.2)))
    #model.add(GRU(500, dropout=0.5, recurrent_dropout=0.5))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))#'sigmoid'))
    #model.add(Dense(1, activation='sigmoid'))#'sigmoid'))

    '''
    model = Sequential()
    model.add(Embedding(num_of_words + 1, 500, input_length=max_sentence_length))

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
    '''

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0

    from keras.optimizers import SGD
    #opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #LRSEn 
    #opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])


    #model.fit(X_train, y_train, batch_size=128, epochs=20)
    #model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=128,shuffle=True)

    model.fit(X_train, y_train, validation_data = (X_test, y_test),  epochs=5, batch_size=64, shuffle=True)

    print('\nTesing Accuracy: {}'. format(model.evaluate(X_test, y_test)[1]))
    #print('\nTraining Accuracy: {}'. format(model.evaluate(X_train, y_train)[1]))

    acc = model.evaluate(X_test, y_test)
    #import tensorflow as tf
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix

    predictions = model.predict(X_test)
    print( confusion_matrix( y_test.argmax(axis=1), predictions.argmax(axis=1)))

    predictions_argmax = predictions.argmax(axis=1)
    y_test_argmax= y_test.argmax(axis=1)
    #write_to_CSV("/Users/russell/Downloads/" + dataset + "_BiLSTM.csv", X_test, y_test_argmax, predictions_argmax)
    write_to_CSV("/Users/russell/Downloads/transliterated_BiLSTM.csv", X_test, y_test_argmax, predictions_argmax)
   
    conf_matrix = confusion_matrix( y_test.argmax(axis=1), predictions.argmax(axis=1))
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis = 0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis = 1)
    f1_scores = (2 * np.mean(precision)*  np.mean(recall)) / (np.mean(precision) + np.mean(recall))
    print ("F1 \n",f1_scores, np.mean(precision),  np.mean(recall))

    total_accuracy += acc
    total_precision += precision
    total_recall += recall
    total_f1_score += f1_scores
    

def get_vulgar_data_n_label():
    
    
    reviews = []
    label = []
    
    
    path = "/Users/russell/Documents/NLP/Paper-8/data/reviews/final/final_vulgar_1331.txt"
    
    #path = "/Users/russell/Documents/NLP/Paper-7/data/vulgar_samples.txt"
    #path = "final_vulgar_1331.txt"
  
    with open(path) as file:
        for line in file:
           reviews.append(line)
           
    num_vulger = len(reviews)

   
       
    path = "/Users/russell/Documents/NLP/Paper-8/data/reviews/final/final_non_vulgar_2608.txt"
 
        
    #path = "final_non_vulgar_2608.txt"
    
    #path = "/Users/russell/Documents/NLP/Paper-7/data/non_vulgar_samples.txt"
 
    with open(path) as file:
        for line in file:
           reviews.append(line)
    
        
    num_non_vulger = len(reviews) - num_vulger
    
    print(num_vulger, num_non_vulger)
        
    label = [0] *  num_vulger  + [1] * num_non_vulger
    
    
    reviews = reviews[: 2 * num_vulger]
    label = label[: 2 * num_vulger]
    
    
    c = list(zip(reviews,label))
    import random
    random.shuffle(c)
    reviews, label= zip(*c)
    

    X = np.array(reviews)
    Y_0 = np.array(label)

    
    from keras.preprocessing.text import Tokenizer
    documents = X #get_data()
    # create the tokenizer
    #print(documents[0])
    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(documents)
    sequences = t.texts_to_sequences(documents)

    # summarize what was learned
    print("Number of Words in Corpus: ", len(t.word_counts))
    print("Number of Samples", t.document_count)
    print(" Word Index Length: " , len(t.word_index))
 

    Y = np.zeros((len(Y_0), 2), dtype=np.int)

    i = 0

    num_of_vulgar = 0
    num_of_non_vulgar = 0

    for value in Y_0:

        #value = str(value).strip()
        #print("-----",value)
        if value == 0:
            num_of_vulgar += 1
            Y[i][0] = 1

        else :
            num_of_non_vulgar += 1
            Y[i][1] = 1

        i += 1


    print("Vulgar: ", num_of_vulgar)
    print("Non-Vulgar: ", num_of_non_vulgar)

    return sequences, Y, len(t.word_counts)

def get_processed_drama_data_n_label():
    from pandas import read_excel
    my_sheet_name = 'drama_review'
    data = read_excel('Drama_review_Russell.xlsx', sheet_name = my_sheet_name)
    numpy_array = data.values
    X = numpy_array[:,1]
    Y_0 = numpy_array[:,2]

    print(X[0])
    from keras.preprocessing.text import Tokenizer
    documents = X #get_data()
    # create the tokenizer
    print(documents[0])
    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(documents)
    sequences = t.texts_to_sequences(documents)

    # summarize what was learned
    print("Number of Words in Corpus: ", len(t.word_counts))
    print("Number of Samples", t.document_count)
    print(" Word Index Length: " , len(t.word_index))
   # print(t.word_docs)
    # integer encode documents
   # print(documents[1400])


    Y = np.zeros((len(Y_0), 3), dtype=np.int)

    i = 0

    num_of_positive = 0
    num_of_negative = 0
    num_of_neutral = 0
    for value in Y_0:
        if value == 1:
            num_of_positive += 1
            Y[i][2] = 1
            #Y[i] = 1
        elif value == -1:
            num_of_negative += 1
            #Y[i] = -1
            Y[i][0] = 1
        else :
            #Y[i] = 0
            num_of_neutral += 1
            Y[i][1] = 1

        i += 1

    print("Pos: ", num_of_positive)
    print("Neg: ", num_of_negative)
    print("Neu: ", num_of_neutral)

    return sequences, Y, len(t.word_counts)

def get_processed_data():

    from keras.preprocessing.text import Tokenizer
    documents = get_data()
    # create the tokenizer
    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(documents)
    sequences = t.texts_to_sequences(documents)

    # summarize what was learned
    print(len(t.word_counts))
    print(t.document_count)
    print(" Word Index Length: " , len(t.word_index))
   # print(t.word_docs)
    # integer encode documents
    print(documents[1400])

    return sequences, len(t.word_counts)
    '''
    from keras.preprocessing.text import one_hot
    encoded_docs = [one_hot(d, len(t.word_counts)) for d in documents]
    return encoded_docs
    '''

    #return t.word_docs
    '''
    encoded_docs = t.texts_to_matrix(documents, mode='freq')
    for i in encoded_docs[1400]:

        if i != 0:
            print(i)
    return encoded_docs
    '''

def main():

    #train_data, train_label, num_of_words = get_processed_drama_data_n_label()


    data, label, num_of_words = get_vulgar_data_n_label()
    
    #data, label, num_of_words = get_transliterated_reviews()

  
    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state = None,stratify = label, test_size=0.1)

    #test_review_length = len(test_review)
    #print("test_review_length :", test_review_length)
    #X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, random_state = None, shuffle= False , test_size=test_review_length)
    
    model(X_train, X_test, y_train, y_test, num_of_words)
    
    #random_state = 10 for Cricket
    #import random
    #X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, random_state = None,stratify = train_label, test_size=0.1)

    #get__testing_label(y_test)

    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, random_state=0, stratify = None, test_size=0.3)
    #y_train = y_train.astype('int')
    #y_test = y_test.astype('int')
    return 0



if __name__ == '__main__':
    main()
