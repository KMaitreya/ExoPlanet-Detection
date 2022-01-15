def dataInfo(data):
    
    #data information
    print("DATA INFORMATION:\n",data.info(),"\n\n")

    #checking out null values in the dataset
    print("NULL VALUES CHECK:\n",data.isna().sum(),"\n\n")

    #percentage null values
    print("PERCENTAGE NULL VALUES:\n",data.isna().mean(),"\n\n")

    #checking if any percentage of null values is greater than 10 percent
    print("PERCENTAGE NULL VALUES GREATER THAN 10:\n",data.isna().mean()>=0.1, "\n\n")