import plotly
plotly.tools.set_credentials_file(username='AnttiVaisanen', api_key='6tkhotzvfm')

import plotly.plotly as py
import plotly.graph_objs as go




import numpy as np

y0 = np.random.randn(50)-1
y1 = np.random.randn(50)+1

trace0 = go.Box(
    y=y0
)
trace1 = go.Box(
    y=y1
)
data = [trace0, trace1]
py.iplot(data)
