from sklearn.preprocessing import LabelEncoder, StandardScaler
from functions import readCSV, printMetatdata, splitData, trainOnDecTree, trainOnNeurNet, scaleData, predictions, accuracyCalc

def main():
    le = LabelEncoder()
    data = readCSV()
    #printMetatdata(data)
    X_train, X_test, Y_train, Y_test = splitData(data, le)

    userIn = input("Enter D for Decision Tree or N for Neural Network: ")

    if userIn == "D":
        print("\nDecision Tree:\n")
        decTree = trainOnDecTree(X_train, Y_train)
        Y_pred_decTree, labels_Enc_decTree, labels_Name_decTree  = predictions(X_test, decTree, le)
        accuracyCalc(Y_test, Y_pred_decTree, labels_Enc_decTree)

    elif userIn == "N":
        print("\nNeural Networks:\n")
        scaler = StandardScaler()
        X_train_scaled, X_test_scaled = scaleData(X_train, X_test, scaler)
        mlpNeurlNet = trainOnNeurNet(X_train_scaled, Y_train)
        Y_Pred_mlpNeurlNet, labels_Enc_mlpNeurlNet, labels_Name_mlpNuerlNet = predictions(X_test_scaled, mlpNeurlNet, le, X_test)
        accuracyCalc(Y_test, Y_Pred_mlpNeurlNet, labels_Enc_mlpNeurlNet)

    else:
        print("Invalid Input")
        exit()

if __name__ == "__main__":
   main()