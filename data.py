import pandas as pd
from pandas import DataFrame
import numpy as np

data = DataFrame(np.arange(6).reshape((2,3)), index=pd.Index(['Ohio', 'Colorado'], name='state'), columns=pd.Index(['one','two','three'],name='number'))

print(data);

result = data.stack()

print(result);