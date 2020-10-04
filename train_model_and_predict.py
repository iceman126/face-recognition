from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def main():
    # load dataset
    data = load('5-celebrity-faces-embeddings.npz')
    train_X, train_y, val_X, val_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print(f"# of train dataset: {train_X.shape[0]}, # of test dataset: {val_X.shape[0]}")

    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    train_X = in_encoder.transform(train_X)
    val_X = in_encoder.transform(val_X)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(train_y)
    train_y = out_encoder.transform(train_y)
    val_y = out_encoder.transform(val_y)
    # fit model
    print("fitting model for prediction")
    # model = SVC(kernel='rbf', probability=True)
    # model = SVC(probability=True)
    # model.fit(train_X, train_y)
    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(train_X, train_y)

    # predict
    yhat_train = model.predict(train_X)
    yhat_val = model.predict(val_X)
    print(yhat_train)
    print(yhat_val)
    # score
    score_train = accuracy_score(train_y, yhat_train)
    score_test = accuracy_score(val_y, yhat_val)
    # summarize
    print(f"Accuracy: train={score_train*100}%, test={score_test*100}%")

if __name__ == "__main__":
    main()