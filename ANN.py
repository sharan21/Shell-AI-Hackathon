from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras import regularizers
from keras import optimizers
from input_pipeline import *

def create_model():

    print ("Creating model...")
    model = Sequential()

    BatchNormalization(
        axis=-1, momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None)


    model.add(Dense(units=102, activation='relu', input_dim=102))
    model.add(Dropout(0.3))
    model.add(Dense(units=200, activation='relu')) # (102,400)
    model.add(Dropout(0.3))
    model.add(Dense(units=100, activation='relu')) # (400,200)
    model.add(Dropout(0.3))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    adam = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['accuracy'])

    print("Done!")

    return(model)



def train_model(pathtojson, pathtoh5):

    
    trained_model = create_model()

    print("Begin Training...")
    trained_model.fit(x_train, y_train, epochs=400, batch_size=16)
    print("Done!")

    print("Saving model...")
    model_json = trained_model.to_json()

    with open(pathtojson, "w") as json_file:
        json_file.write(model_json)

    trained_model.save_weights(pathtoh5)
    print("Done!")

    print("Evaluating model...")
    score, acc = trained_model.evaluate(x_test, y_test, batch_size=16)
    print ("Scores for Test set: {}".format(score))
    print ("Accuracy for Test set: {}".format(acc))

    return trained_model


def test_model(pathtojson, pathtoh5, data, labels ):

    print ("Testing model")
    print("using model: {}".format(pathtojson))

    # load json and create model
    json_file = open(pathtojson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(pathtoh5)
    print("Loaded model from disk")

    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
    print(data.shape)

    labels = to_categorical(labels)
    score, acc = loaded_model.evaluate(data, labels, batch_size=16)

    print ("Scores for Test set: {}".format(score))
    print ("Accuracy for Test set: {}".format(acc))


def load_model(pathtojson, pathtoh5):

    json_file = open(pathtojson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(pathtoh5)

    return loaded_model



def load_and_predict(pathtojson, pathtoh5, data):

    print("using model: {}".format(pathtojson))

    # load json and create model
    json_file = open(pathtojson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    #load weights into new model
    loaded_model.load_weights(pathtoh5)
    print("Loaded model from disk")

    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print ("compiled the loaded model with cat. cross entropy with adam optim...")
    print ("shape of data {}".format(data.shape))

    classes = loaded_model.predict(data)

    print ("done predicting, printing")

    for instance in classes:
        print (instance)

    return classes



if __name__ == '__main__':


    x_train, y_train, x_test, y_test = get_final_data()

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # exit(0)
    
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    path_to_json = './models/average10.json'
    path_to_h5 = './models/average10.h5'
    
    train_model(path_to_json, path_to_h5)
    # testmodel(pathtojson, pathtoh5, data, labels)
    # loadandpredict('./model.json','./model.h5',data)

    model = load_model(path_to_json, path_to_h5)







