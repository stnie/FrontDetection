import os
import numpy as np

def getParamsFromInfo(folder):
    infoFiles = ["criterion", "test_criterion", "test_loader", "train_loader", "opimizer", "test_data_set", "train_data_set", "weight"]
    for idx, f in enumerate(infoFiles):
        infoFiles[i] = os.path.join(folder, f+"_info.txt")
    
    criterion = getCriterionFromInfo(infoFiles[0])
    test_criterion = getCriterionFromInfo(infoFiles[1])

    test_loader = getLoaderFromInfo(infoFiles[2])
    train_loader = getLoaderFromInfo(infoFiles[3])

    optimizer = getOptimizerFromInfo(infoFiles[4])

    test_data_set = getDataSetFromInfo(infoFiles[5])
    train_data_set = getDataSetFromInfo(infoFiles[6])
    
    weight = getWeightFromInfo(infoFiles[7])

def getVariablesFromDataSetInfo(filename):
    variables = []
    with open(filename, "r") as f:
        for line in f:
            if("variables" in line):
                parts = line.split("variables")
                parts2 = parts[1].split("[")[1].split("]")[0].split(",")
                for var in parts2:
                    var = var.strip()[1:-1]
                    variables.append(var)
    return variables

def getNormTypeFromDataSetInfo(filename):
    with open(filename, "r") as f:
        for line in f:
            if("normalize_type" in line):
                parts = line.split("normalize_type")
                parts2 = parts[1].split(":")[1].split(",")[0]
                return parts2

def getLevelRangeFromDataSetInfo(filename):
    levelrange = []
    with open(filename, "r") as f:
        for line in f:
            if("levelrange" in line):
                parts = line.split("levelrange")
                parts2 = parts[1].split("[")[1].split("]")[0].split(",")
                for var in parts2:
                    var = var.strip()
                    levelrange.append(int(var))
    return np.array(levelrange)

def getDataSetInformationFromInfo(filename):
    dataInfo = dict()
    with open(filename, "r") as f:
        datasetType = f.readline()
        print(datasetType)
        for line in f:
            parts = line.split(" :: ")
            try:
                key,datatype,value,end = parts[0], parts[1], parts[2:-1], parts[-1]
                dataInfo[key] = formatValue(datatype, value)
            except:
                print("wrongly formatted line. Potentially a multiline object")
    return dataInfo

def formatValue(datatype, valueString):
    if(isinstance(valueString, list)):
        valueString = ' :: '.join(valueString)
    valueString = valueString.strip()
    datatype = datatype.strip()
    # handle basic datatypes
    if(datatype == 'str'):
        valueString = valueString.strip("'")
        valueString = valueString.strip()
        return valueString
    elif(datatype == 'int'):
        return int(valueString)
    elif(datatype == 'float'):
        return float(valueString)
    # handle more complex basic datatypes (lists, tuples, dictionaries)
    else:

        typeparts = datatype.split("(")
        if(len(typeparts) <= 1):
            print("Wrong format, exiting now")
            return "NonenoN"
        elif(len(typeparts) == 2):
            nested = False
        else:
            nested = True
        if(nested == False):
            if(typeparts[0] == "list"):
                values = valueString[1:-1].split(',')
                types = typeparts[1][:-1]
                return [formatValue(types, value) for value in values]
            elif(typeparts[0] == "tuple"):
                values = valueString[1:-1].split(',')
                types = typeparts[1][:-1].split(',')
                return tuple([formatValue(types[idx], value) for idx,value in enumerate(values)])
            elif(typeparts[0] == "dict"):
                entries = valueString[1:-1].split(',')
                keys, values = list(zip([entry.split(':') for entry in entries]))
                keytype,valuetype = typeparts[1][:-1].split(': ')
                return {formatValue(keytype, keys[idx]): formatValue(valuetype, values[idx]) for idx in range(len(keys))}
            else:
                print("unknown Type encountered, line will be ignored")
        else:
            # a list nested with other complex types
            if(typeparts[0] == "list"):
                values = getValueListFromNested(valueString[1:-1])
                # get the inner type
                types = "(".join(typeparts[1:])[:-1]
                return [formatValue(types, value) for value in values]
            # a tuple nested with other complex types
            elif(typeparts[0] == "tuple"):
                values = getValueListFromNested(valueString[1:-1])
                typeList = "(".join(typeparts[1:])[:-1]
                types = getValueListFromNested(typeList)
                return tuple([formatValue(types[idx], value) for idx,value in enumerate(values)])
            # a dict nested with other complex types as values
            elif(typeparts[0] == "dict"):
                entries = getValueListFromNested(valueString[1:-1])
                print(entries)
                keys, values = list(zip(*[entry.split(': ') for entry in entries]))
                types = "(".join(typeparts[1:])[:-1]
                typeParts = types.split(': ')
                keytype = typeParts[0]
                valuetype = ":".join(typeParts[1:])
                print(keytype, valuetype)
                print(values)
                return {formatValue(keytype, keys[idx]): formatValue(valuetype, values[idx]) for idx in range(len(keys))}
            else:
                print("unknown Type encountered, line will be ignored")
    return "NonenoN"

def getValueListFromNested(valueString):
    AllLevelValues = valueString.split(',')
    values = []
    level = 0
    for value in AllLevelValues:
        if(level == 0):
            values.append(value)
        elif(level > 0):
            values[-1] += ","+value
        level+=value.count("(")+value.count("{")+value.count("[")
        level-=value.count(")")+value.count("}")+value.count("]")
    return values


        


#getVariablesFromDataSetInfo(sys.argv[1])
    
