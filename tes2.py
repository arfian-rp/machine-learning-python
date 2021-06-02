import pandas as pd
apel = pd.Series([5,2,0,1])
jeruk = pd.Series([0,3,8,2], index=[1,2,3,4])
jeruk.index=['q','w','e','r']
df1 = pd.DataFrame([3,2,0,1], columns=['apel'], index=[1,2,3,4])
dict1 ={
    'apel':[5,2,0,1],
    'jeruk':[0,3,8,2]
}
df1 = pd.DataFrame(dict1)
print(jeruk['q'])