#Rae Tasker
#April 2022
#Coursework for Ethics of Machine Learning

#Ridge Regression Analysis of Community/Crime Data


from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict


#assign the particular values of interest to a list
def assign_data(desired_features, headers, instance_data):
    #create list
    collected = []
 
    for instance in instance_data:
        sample = []
        
        for item in desired_features:
            feature_index = headers.index(item)
            sample.append(float(instance[feature_index]))
            
        collected.append(sample)
        
    return collected

#determines the majority racial group for each area in the dataset
def majority_define(data):
    maj_list = []

    #print(data)
    
    for item in data:
        #find the index of the maximum value in the list
        num = item.index(max(item))

        if num == 0:
            maj_list.append("Majority Black")
        elif num == 1:
            maj_list.append("Majority White")
        elif num == 2:
            maj_list.append("Majority Asian")
        elif num == 3:
            maj_list.append("Majority Hispanic")
        else:
            print("error in maj_def")
                

    return maj_list


#breaks down data based on racial majority
def data_breakdowns(data, racedata):
    wlist = []
    blist = []
    alist = []
    hlist = []
    
    for i in range(len(data)):
        if (racedata[i] == "Majority White"):
            wlist.append(conv_prediction(data[i]))
        elif (racedata[i] == "Majority Black"):
            blist.append(conv_prediction(data[i]))
        elif (racedata[i] == "Majority Asian"):
            alist.append(conv_prediction(data[i]))
        else:
            hlist.append(conv_prediction(data[i]))

    wh, wl = high_low(wlist)
    bh, bl = high_low(blist)
    ah, al = high_low(alist)
    hh, hl = high_low(hlist)

    
    print("White:",high_low(wlist),wh + wl)
    print("Black:",high_low(blist),bh+bl)
    print("Asian:",high_low(alist),ah+al)
    print("Hispanic:",high_low(hlist),hh+hl)

    #print(wh+wl+bh+bl+ah+al+hh+hl)

    return wh, wl, bh, bl, ah, al, hh, hl
    

def high_low(data):
    high = 0
    low = 0

    for item in data:
        if(item == "high violence"):
            high += 1
        else:
            low += 1
            
    return high, low

def separate_by_racial_majority(data, racedata, targetrace):
    maj = []
    for i in range(len(data)):
        if racedata[i] == targetrace:
            maj.append(data[i])

    return maj

#10-fold cross validation -- use predict(X)
    #how many data points in each set
def cval_prediction(features, target, racedata):
    #split features and target into 10 parts
    num = len(features)/10
    num2 = len(target)/10

    results = []
    compare_sum = 0

    comp_hh_sum = 0
    comp_hl_sum = 0
    comp_lh_sum = 0
    comp_ll_sum = 0

    bcorrect = 0
    wcorrect = 0
    acorrect = 0
    hcorrect = 0
    
    w_hh = 0
    w_hl = 0
    w_lh = 0
    w_ll = 0

    b_hh = 0
    b_hl = 0
    b_lh = 0
    b_ll = 0

    a_hh = 0
    a_hl = 0
    a_lh = 0
    a_ll = 0

    h_hh = 0
    h_hl = 0
    h_lh = 0
    h_ll = 0
    

    for i in range(10):
        trainingfeat = []
        trainingsample = []
        
        for item in features:
            trainingfeat.append(item)

        for item in target:
            trainingsample.append(item)

        current = int(i*num)
        subfeats = features[current:current+int(num)]

        #everything except sub-categories needs to go in as training data
        #remove the test data from the training set
        for item in subfeats:
            trainingfeat.remove(item)

        subtargets = target[current:current+int(num)]

        #remove the test targets from the training data
        for item in subtargets:
            trainingsample.remove(item)
        

        print("num samples:",len(subfeats), len(subtargets))

        #how many of each racial group are in each sample
        wh, wl, bh, bl, ah, al, hh, hl = data_breakdowns(subtargets, racedata)



        apply_results = apply_ridge(trainingfeat, trainingsample, subfeats)
        
        correct, incorrect, hh, hl, lh, ll = compare_results(subtargets, apply_results)


        #Correctness of Black Group
        bres = separate_by_racial_majority(apply_results, racedata, "Majority Black")
        bpred = separate_by_racial_majority(subtargets, racedata, "Majority Black")

        bcor, bincor, bhh, bhl, blh, bll = compare_results(bpred, bres)

        b_hh += bhh
        b_hl += bhl
        b_lh += blh
        b_ll += bll

        #Correctness of White Group
        wres = separate_by_racial_majority(apply_results, racedata, "Majority White")
        wpred = separate_by_racial_majority(subtargets, racedata, "Majority White")

        wcor, wincor, whh, whl, wlh, wll = compare_results(wpred, wres)

        w_hh += whh
        w_hl += whl
        w_lh += wlh
        w_ll += wll
        
        #Correctness of Asian Group
        ares = separate_by_racial_majority(apply_results, racedata, "Majority Asian")
        apred = separate_by_racial_majority(subtargets, racedata, "Majority Asian")

        acor, aincor, ahh, ahl, alh, All = compare_results(apred, ares)

        a_hh += ahh
        a_hl += ahl
        a_lh += alh
        a_ll += All
        
        #Correctness of Hispanic Group
        hres = separate_by_racial_majority(apply_results, racedata, "Majority Hispanic")
        hpred = separate_by_racial_majority(subtargets, racedata, "Majority Hispanic")

        hcor, hincor, hhh, hhl, hlh, hll = compare_results(hpred, hres)

        h_hh += hhh
        h_hl += hhl
        h_lh += hlh
        h_ll += hll


##        bhigh += bh
##        blow += bl
##        ahigh += ah
##        alow += al
##        hhigh += hh
##        hlow += hl
        
        compare_sum += correct
        bcorrect += bcor
        wcorrect += wcor
        acorrect += acor
        hcorrect += hcor
        
        comp_hh_sum += hh
        comp_hl_sum += hl
        comp_lh_sum += lh
        comp_ll_sum += ll

        for item in apply_results:
            results.append(item)


        print("\n")

    wsum, bsum, asum, hsum = racetally(racedata)
    #data_breakdowns(results, racedata)

    print("\n")
    
    print("Average results (out of 199):")
    print("avg correct:",compare_sum/10)
    print("avg predicted high, calculated high:",comp_hh_sum/10)
    print("avg predicted high, calculated low:",comp_hl_sum/10)
    print("avg predicted low, calculated high:",comp_lh_sum/10)
    print("avg predicted low, calculated low:",comp_ll_sum/10)

    print("\n")

    print("Avg results by race:")
    print("White:",wcorrect/10)
    print("Black:",bcorrect/10)
    print("Asian:",acorrect/10)
    print("Hispanic:",hcorrect/10)

    print("\n")

    print("Breakdown of White Majority Group Results")
    print("avg predicted high, calculated high:",w_hh/10)
    print("avg predicted high, calculated low:",w_hl/10)
    print("avg predicted low, calculated high:",w_lh/10)
    print("avg predicted low, calculated low:",w_ll/10)

    print("\n")

    print("Breakdown of Black Majority Group Results")
    print("avg predicted high, calculated high:",b_hh/10)
    print("avg predicted high, calculated low:",b_hl/10)
    print("avg predicted low, calculated high:",b_lh/10)
    print("avg predicted low, calculated low:",b_ll/10)

    print("\n")

    print("Breakdown of Asian Majority Group Results")
    print("avg predicted high, calculated high:",a_hh/10)
    print("avg predicted high, calculated low:",a_hl/10)
    print("avg predicted low, calculated high:",a_lh/10)
    print("avg predicted low, calculated low:",a_ll/10)

    print("\n")

    print("Breakdown of Hispanic Majority Group Results")
    print("avg predicted high, calculated high:",h_hh/10)
    print("avg predicted high, calculated low:",h_hl/10)
    print("avg predicted low, calculated high:",h_lh/10)
    print("avg predicted low, calculated low:",h_ll/10)
            
    return results

def racetally(races):
    wsum = 0
    bsum = 0
    asum = 0
    hsum = 0
    
    for i in range(len(races)):
        if races[i] == "Majority White":
            wsum += 1
        if races[i] == "Majority Black":
            bsum += 1
        if races[i] == "Majority Asian":
            asum += 1
        else:
            hsum +=1

    return wsum, bsum, asum, hsum


#applies the ridge regression machine learning model from scikit-learn
#and returns the predicted values
def apply_ridge(trainfeats, trainsamples, testfeats):
    clf = Ridge(alpha=1.0)
    clf.fit(trainfeats, trainsamples)
    return clf.predict(testfeats)
    

#convert from predicted numbers to categories
def conv_prediction(samples):
    
    for item in samples:
        if item >= 0.25:
            return "high violence"
        elif item < 0.25:
            return "low violence"
        else:
            return "error"


#compares expected results to those calculated by the algorithm
def compare_results(predicted, calculated):
    conv_pred = []
    conv_cal = []
    counter_correct = 0
    counter_incorrect = 0
    
    #compare expected results to actual results
    for element in predicted:
        conv_pred.append(conv_prediction(element))
                               
    for element in calculated:
        conv_cal.append(conv_prediction(element))


    for i in range(len(conv_cal)):
        if conv_pred[i] == conv_cal[i]:
            counter_correct += 1
        else:
            counter_incorrect += 1
        #print("predicted:",conv_pred[i],"\tcalculated:",conv_cal[i])

    print("correct:", counter_correct, "incorrect:", counter_incorrect)

    hh, hl, lh, ll = comp_by_class(conv_pred, conv_cal)

    return counter_correct, counter_incorrect, hh, hl, lh, ll


#breaks down comparison futher by high or low prediction
def comp_by_class(pred, cal):
    high_high_counter = 0
    high_low_counter = 0
    low_high_counter = 0
    low_low_counter = 0

    for i in range(len(cal)):
        if pred[i] == "high violence" and cal[i] == "high violence":
            high_high_counter += 1
        elif pred[i] == "high violence" and cal[i] == "low violence":
            high_low_counter += 1
        elif pred[i] == "low violence" and cal[i] == "high violence":
            low_high_counter += 1
        else:
            low_low_counter += 1

    return high_high_counter, high_low_counter, low_high_counter, low_low_counter
            
        
def main():
    dataset = open("dataset.csv", 'r')
    alldataHeaders = dataset.readline().split(",")
    #removing bogus symbols at the start of the file
    alldataHeaders[0] = alldataHeaders[0][3:]
    alldata = dataset.readlines()

    splitdata = []

    for item in alldata:
        splitdata.append(item.strip().split(","))
        
    #headers that will be protected data
    protHeaders = [
        "racepctblack",
        "racePctWhite",
        "racePctAsian",
        "racePctHisp"
        ]
    #all headers, including protected ones
    relevantHeaders = [
        "householdsize",
        "racepctblack",
        "racePctWhite",
        "racePctAsian",
        "racePctHisp",
        "medIncome",
        "PctPopUnderPov",
        "PctLess9thGrade",
        "PctNotHSGrad",
        "PctBSorMore",
        "PctUnemployed",
        "PctEmploy",
        'PersPerFam',
        'PctFam2Par',
        'PctKids2Par',
        'PctYoungKids2Par',
        'PctTeen2Par',
        'PctWorkMomYoungKids',
        'PctWorkMom',
        'PctLargHouseFam',
        'PctLargHouseOccup',
        'PersPerOccupHous',
        'PersPerOwnOccHous',
        'PersPerRentOccHous',
        'HousVacant',
        'PctHousOccup',
        'MedYrHousBuilt',
        'PctHousNoPhone',
        'PctWOFullPlumb',
        'RentLowQ',
        'RentMedian',
        'RentHighQ',
        'MedRent',
        'MedRentPctHousInc',
        'MedOwnCostPctInc',
        'MedOwnCostPctIncNoMtg',
        'NumInShelters',
        'NumStreet'
        ]
    
    #excludes protected variables
    noprotectedHeaders = [
        "householdsize",
        "medIncome",
        "PctPopUnderPov",
        "PctLess9thGrade",
        "PctNotHSGrad",
        "PctBSorMore",
        "PctUnemployed",
        "PctEmploy",
        'PersPerFam',
        'PctFam2Par',
        'PctKids2Par',
        'PctYoungKids2Par',
        'PctTeen2Par',
        'PctWorkMomYoungKids',
        'PctWorkMom',
        'PctLargHouseFam',
        'PctLargHouseOccup',
        'PersPerOccupHous',
        'PersPerOwnOccHous',
        'PersPerRentOccHous',
        'HousVacant',
        'PctHousOccup',
        'MedYrHousBuilt',
        'PctHousNoPhone',
        'PctWOFullPlumb',
        'RentLowQ',
        'RentMedian',
        'RentHighQ',
        'MedRent',
        'MedRentPctHousInc',
        'MedOwnCostPctInc',
        'MedOwnCostPctIncNoMtg',
        'NumInShelters',
        'NumStreet'
        ]
    
    relevantdata = []
    protdata = []
    noprotecteddata = []
    targetdata = []

    #collect the list of relevant non-protected test data
    relevantdata = assign_data(relevantHeaders, alldataHeaders, splitdata)

    #collect protected data separately for ease
    protdata = assign_data(protHeaders, alldataHeaders, splitdata)
        
    #collect the list of target data - the last item in the list
    targetdata = assign_data(["ViolentCrimesPerPop\n"], alldataHeaders, splitdata)

    noprotecteddata = assign_data(noprotectedHeaders, alldataHeaders, splitdata)

    
    maj = majority_define(protdata)
    data_breakdowns(targetdata, maj)
    
    caldata = cval_prediction(relevantdata, targetdata, maj)


    print("________NO PROTECTED DATA RUN_________")

    npcaldata = cval_prediction(noprotecteddata, targetdata, maj)



main()



