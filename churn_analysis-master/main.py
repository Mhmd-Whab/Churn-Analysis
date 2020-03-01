import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import beta


def prob(dataframe, col):
    group = dataframe.groupby(col)
    num_group = group.size()
    return num_group / len(dataframe)


def joint_prob(dataframe, col1, col2):
    p = dataframe.groupby([col1, col2])
    return (p.size()) / len(dataframe)


def cond_prob(dataframe, col1, col2):
    join = (joint_prob(dataframe, col1, col2))
    join_arr = np.array(join)
    p1 = np.array(prob(dataframe, col1))
    p2 = np.array(prob(dataframe, col2))
    for i in range(len(p1)):
        for k in range(len(p2)):
            join_arr[(i * len(p2)) + k] = join_arr[(i * len(p2)) + k] / p2[k]
    cond = pd.DataFrame(join)
    cond.rename(columns={0: "Joint"}, inplace=True)
    cond.insert(1, "Conditional", join_arr)
    return cond

def histo(dataframe,columns):
    sns.set()
    for col in columns:
        dataframe.hist(col,bins=25)
        plt.show()
        pdf, bins = np.histogram(data[col], bins=25, density=True)
        bin_centers = (bins[1:] + bins[:-1]) * 0.5
        plt.plot(bin_centers, pdf)
        plt.show()
        plt.hist(dataframe[col], cumulative=True, density=1, label='CDF',
            histtype='step', alpha=1, color='k')
        plt.show()

def mean_var(df,columns):
    res = []
    for col in columns:
        n = np.array(df[col])
        arr = [n.mean(),n.var()]
        res.append(arr)
    return res

def normal_fit(data,col):
    plt.hist(data[col], bins=10000, density=True, alpha=0.6, color='g')
    mu, std = norm.fit(data[col])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 10000)
    # p = norm.pdf(x, mu, std)
    # plt.plot(x, p, 'y', linewidth=2)
    # title = "Normal fitting of "+str(col)+" with Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    # plt.title(title)
    # plt.show()
    return x,mu,std


def exponential_fit(data,col):
    plt.hist(data[col], bins=10000, density=True, alpha=0.6, color='g')
    loc, scale = expon.fit(data[col])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 10000)
    # plt.plot(x, p, 'k', linewidth=2)
    # title = "Exponential fitting of "+str(col)
    # plt.title(title)
    # plt.show()
    return x,loc,scale

def beta_fit(data,col):
    plt.hist(data[col], bins=10000, density=True, alpha=0.6, color='g')
    a,b,loc,scale = beta.fit(data[col])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 10000)
    # plt.title("Beta fitting of "+str(col))
    # plt.plot(x, p, 'r', linewidth=2)
    # plt.show()
    return x,a,b,loc,scale

def pdf(data,col):
    pdf, bins = np.histogram(data[col], bins=10000, density=True)
    bin_centers = (bins[1:] + bins[:-1]) * 0.5
    # plt.plot(bin_centers, pdf)
    # plt.show()
    return pdf,bin_centers

def bayes(data,col,given):
    pr_chu = pd.Series(data.groupby(col).size())
    pr1 = pd.Series(data.groupby(given).size())
    pr_c1 = pd.Series(data.groupby([given, col]).size() / pr_chu)
    pr_chu = pr_chu / len(data)
    pr1 = pr1 / len(data)
    return (pr_c1*pr_chu)/pr1


# milestone1
# 1 task
data = pd.read_csv('cell2celltrain.csv')
# 2,3 tasks
data.dropna(inplace=True)
columns = list(data)
total = len(data)
number_data = data.select_dtypes(exclude=['object'])
string_data = data.select_dtypes(exclude=['int','float'])
# data frame for the probabilty of string columns
# task 4
# print(prob(data,"CreditRating"))
# task 5
# print(joint_prob(data,"Churn","ChildrenInHH"))
# task 6
# y = data.groupby("CreditRating").size()
# cond = data.groupby(["PrizmCode","CreditRating"]).size()/y
# print(cond)
sns.set()
# task 7,8,9
# histo(number_data,list(number_data))
# task 10
# join_pdf = plt.hist2d(data["MonthlyRevenue"],data["MonthlyMinutes"],bins=50,range=[[0,150],[0,1500]])
# plt.show()
# task 11
# mv = mean_var(number_data,list(number_data))

# milestone 2 ...
# task 1
data_corr= data.corr()
data_cov = data.cov()

# task 2
data_corr=data_corr.drop(columns="CustomerID")
data_corr=data_corr.drop("CustomerID",axis=0)
columns=list(number_data)
n=len(data_corr)
list = [columns[1]]

for i in range(n-1):
    cor=data_corr.iloc[0,i+1]
    if abs(cor)<=0.5:
        col=columns[i+2]
        list.append(col)


toremove=[]
for i in range(len(list)-1):
    for j in range(len(columns)-2):
        cor=data_corr.loc[list[i+1],columns[j+2]]
        if abs(cor)>0.5 and cor!=1.0:
            col=columns[j+2]
            if list.__contains__(col) and not toremove.__contains__(col):
                toremove.append(col)

for i in range(len(toremove)):
    list.remove(toremove[i])


# task 3

# c=number_data.iloc[0,1]
# m=expon.pdf(c,loc,scale)
array_pdf=[]
for i in range(len(list)):
    x1, loc, scale = exponential_fit(number_data, list[i])
    x2, mu, std = normal_fit(number_data, list[i])
    p1 = expon.pdf(x1, loc, scale)
    pdf1, b = pdf(number_data, list[i])
    p2 = norm.pdf(x2, mu, std)
    pdf2, b = pdf(number_data, list[i])
    mse1 = np.square(np.subtract(p1, pdf1)).mean()
    mse2 = np.square(np.subtract(p2, pdf2)).mean()
    tmp=[]
    if mse1<mse2 :
        tmp=["expon",loc,scale]
    else:
        tmp=["norm",mu,std]
    array_pdf.append(tmp)

bays_data=data[data.Churn!="No"]
array_cond_pdf=[]
for i in range(len(list)):
    x1, loc, scale = exponential_fit(bays_data, list[i])
    x2, mu, std = normal_fit(bays_data, list[i])
    p1 = expon.pdf(x1, loc, scale)
    pdf1, b = pdf(bays_data, list[i])
    p2 = norm.pdf(x2, mu, std)
    pdf2, b = pdf(bays_data, list[i])
    mse1 = np.square(np.subtract(p1, pdf1)).mean()
    mse2 = np.square(np.subtract(p2, pdf2)).mean()
    tmp=[]
    if mse1<mse2 :
        tmp=["expon",loc,scale]
    else:
        tmp=["norm",mu,std]
    array_cond_pdf.append(tmp)

pr_chu = pd.Series((data.groupby("Churn").size())/total).take(indices=[1])
pr_chu = float(pr_chu)
churn_bays=[]
data.reset_index(inplace=True)
for i in range(len(data)):
    result=1
    for j in range(len(list)):
        stat_c=array_cond_pdf[j][0]
        stat=array_pdf[j][0]
        c = data[list[j]][i]
        if stat=="norm":
            p=norm.pdf(c,array_pdf[j][1],array_pdf[j][2])
        else:
            p = expon.pdf(c, array_pdf[j][1], array_pdf[j][2])
        if stat_c == "norm":
            pc = norm.pdf(c, array_cond_pdf[j][1], array_pdf[j][2])
        else:
            pc = expon.pdf(c, array_cond_pdf[j][1], array_pdf[j][2])
        result=result*(pc/p)
    result=result*pr_chu
    churn_bays.append(result)
churn_bays=pd.DataFrame(churn_bays)
churn_bays.dropna(inplace=True)
churn_bays.reset_index(inplace=True)
sum=0
for i in range(len(churn_bays)):
    if churn_bays[0][i] >=0.7:
        sum+=1

actual_sum=0
for i in range(len(data)):
    if data["Churn"][i] == "Yes":
        actual_sum+=1
print("sum of churn calculated = "+str(sum))
print("true sum of churn = "+str(actual_sum))
acc=actual_sum-sum
print("accuracy is "+str(acc/100)+"%")
