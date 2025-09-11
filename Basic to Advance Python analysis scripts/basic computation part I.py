#-----------------------------
# Basic Co mputation Part I
#-----------------------------
import numpy as np
import pandas as pd
import scipy.stats as s1

# Basic 1
2-1
# Basic 2
4+5
# Basic 3
5*5
# Basic 4
25/5
# Basic 5
np.exp(4)
# Basic 6
np.log(23)
# Basic 7
np.sin(34)
# Basic 8
data1=[23,34,45,34,12,45,65,34,23,46,45,23,24,35,
       43,23,34,45,36,34,45,23,24,45,45,24,35,45,34,34]
df = pd.DataFrame(data1)
# Basic 9
df.describe()
# Basic 10
np.average(df)
# Basic 11
np.std(df)
np.var(df)
# Basic 12
import matplotlib.pyplot as plt
plt.hist(df)
# Basic 13
plt.boxplot(df)
# Basic 14
plt.subplot(221)
plt.hist(df)
plt.title('histogram for data 1')
plt.xlabel("Values in x axis")

plt.subplot(222)
plt.boxplot(df)
plt.title('boxplot for data 1')
# Basic 15
# qualitative data set
data2=['l','l','n','n','a','a','a','e','e','e','e','e','e','e','l','l','l','n','n','a']
import collections
frequency = collections.Counter(data2)
# Basic 16
dict(frequency)
# Basic 17
x=['LT','NA','AS','EU']
y=[5,2,3,7]
plt.bar(x,y)
plt.title("Bar Chart for continet Variable")
plt.xlabel("Continents")
plt.ylabel("Frequency")
# # Basic 18
cols=['r','g','y','b']
labels=['LT','NA','AS','EU']
sizes=[5,2,3,7]
explode = [0,0.1,0.2,0.3]
plt.pie(sizes,explode=explode, labels=labels,colors=cols)
#plt.pie(sizes, labels=labels,colors=cols)
plt.title("Pie Chart of Race Variable")
plt.axis('equal')
# Basic 19
plt.subplot(221)
x=['LT','NA','AS','EU']
y=[5,2,3,7]
plt.bar(x,y)
plt.title("Bar Chart for continet Variable")
plt.xlabel("Continents")

plt.subplot(222)
cols=['r','g','y','b']
labels=['LT','NA','AS','EU']
sizes=[5,2,3,7]
#explode = [0,0,0,0]
plt.pie(sizes, labels=labels,colors=cols)
plt.title("Pie Chart of Race Variable")
plt.axis('equal')

# Basic 20
df = pd.DataFrame(data1)
df.sample(n=15)

# Basic 21
# simulation for empirical rule
mu = 50 # mean
sd1= 10 # standard deviation
B=100
x = np.random.normal(mu,sd1,B)
sorted(x)
plt.hist(x)
# within 1 standard deviation
[50-1*10,50+1*10] 
# within 2 standard deviation
[50-2*10,50+2*10]
# within 3 standard deviation
[50-3*10,50+3*10]
np.std(data1)

# Basic 22
# simulation for chebysheb's rule
beta = 5 # beta is the scale parameter, which equal to beta=1/lambda
B=100
y = np.random.exponential(beta,B)
a=np.average(y)
b=np.std(y)
sorted(y)
plt.hist(y)
# within 1 standard deviation
[a-b,a+b]
# within 2 standard deviation
[a-2*b,a+2*b]
# within 3 standard deviation
[a-3*b,a+3*b]

# Basic 23
np.cov(x,y)
np.var(x)
np.var(y)

# Basic 24
7**(2)

