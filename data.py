# Script to reshape data using pandas stack function.
import pandas as pd
from pandas import DataFrame
import numpy as np
import xlrd
import xlwt
import matplotlib.pyplot as plt

# Test data stack using simple dataframe
data 			= DataFrame(np.arange(6).reshape((2,3)), index=pd.Index(['Helsinki', 'Jyväskylä'], name='Kaupunki'), columns=pd.Index(['Yksi','Kaksi','Kolme'],name='Järjestys'))
print(data);

result 			= data.stack()
print(result);


# Use data stack to reshape excel sheet data
xls_file		= pd.ExcelFile('tyottomat.xls')
table 			= xls_file.parse('Sheet1')
print(table)

pivotoitu 		= table.stack();
exceliin 		= DataFrame(pivotoitu)
print(pivotoitu)

# Write output into a new Excel file:
exceliin.to_excel('tyottomat2.xls')