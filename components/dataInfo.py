def dataInfo(data):
    
    #data information
    print(data.info())

    #checking out null values in the dataset
    print(data.isna().sum())

    #percentage null values
    print(data.isna().mean(),"\n\n")

    #checking if any percentage of null values is greater than 10 percent
    print(data.isna().mean()>=0.1)
