# Importing essential libraries
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import re
import csv
import warnings

# Filtering out DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Loading and preprocessing data
training = pd.read_csv('Medical_Data/Training.csv')
testing = pd.read_csv('Medical_Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

# Grouping data by 'prognosis'
reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers using LabelEncoder
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# Preprocessing testing data
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# Training Decision Tree Classifier with train and test
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

# here Cross-validation on Decision Tree Classifier
scores = cross_val_score(clf, x_test, y_test, cv=3)
print("Decision Tree Cross-validation Score:", scores.mean())

# Training Support Vector Classifier
model = SVC()
model.fit(x_train, y_train)
print("Support Vector Classifier Score:", model.score(x_test, y_test))

# Extracting feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Function to read text using pyttsx3
def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

# Dictionaries for symptom information
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

# Dictionary to map symptoms to indices
symptoms_dict = {symptom: index for index, symptom in enumerate(x)}

# Function to calculate condition based on symptoms and days
def calc_condition(exp, days):
    sum_severity = sum(severityDictionary[item] for item in exp)
    if (sum_severity * days) / (len(exp) + 1) > 13:
        print("Please consult with a healthcare professional")
    else:
        print("It might not be that bad, but you should take precautions.")

# Function to load symptom descriptions from CSV
def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

# Function to load severity information from CSV
def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                # Check if the row has the expected number of elements
                if len(row) >= 2:
                    _diction = {row[0]: int(row[1])}
                    severityDictionary.update(_diction)
        except:
            pass

# Function to load precaution information from CSV
def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# Function to get user information
def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t", end="->")
    name = input("")
    print("Hello, ", name)

# Function to check pattern in disease list
def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []

# Function to predict using a secondary Decision Tree model
def sec_predict(symptoms_exp):
    df = pd.read_csv('Medical_Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

# Function to print the predicted disease
def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

# Function to convert decision tree to code
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nEnter the symptom you are experiencing  \t\t", end="->")
        disease_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days = int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms, "? : ", end='')
                while True:
                    inp = input("")
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ", end="")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                print("You may have ", present_disease[0])
                print(description_list.get(present_disease[0], "Description not available"))
            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list.get(present_disease[0], "Description not available"))
                print(description_list.get(second_prediction[0], "Description not available"))

            precution_list = precautionDictionary.get(present_disease[0], [])
            print("Take following measures : ")
            for i, j in enumerate(precution_list):
                print(i + 1, ")", j)

    recurse(0, 1)


# Calling functions to get symptom dictionaries and user information
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()

# Calling the tree_to_code function to interact with the user
tree_to_code(clf, cols)

print("----------------------------------------------------------------------------------------")
