import numpy as np
from numpy.typing import NDArray
import pandas as pd
from matplotlib import pyplot as plt
from scipy.linalg import eig, eigh
import scipy.stats as ss
from plotly import express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
from multidomain_MMD import L, H, KSE, kde
from steeringassistant.config import load_config
import pmdarima as pm
from pmdarima.model_selection import train_test_split


df1 = pd.read_csv('TBM-A/tunnel-1/preprocessed_0.5.csv').head(2720)
df2 = pd.read_csv('TBM-A/tunnel-2/preprocessed_0.5.csv').head(1660)
df3 = pd.read_csv('TBM-A/tunnel-3/preprocessed_0.5.csv').head(1900)
df4 = pd.read_csv('TBM-B/tunnel-1/preprocessed_0.5.csv').head(3520)

x1 = df1[['cot_ver','ac_curve_ver']]
x2 = df2[['cot_ver','ac_curve_ver']]
x3 = df3[['cot_ver','ac_curve_ver']]
x4 = df4[['cot_ver','ac_curve_ver']]


### performing mulidomain tca 
x1234 = pd.concat([x1, x2, x3, x4], ignore_index= True, axis = 0)

S = 4
n1 = len(x1)
n2 = len(x2)
n3 = len(x3)
n4 = len(x4)
print(n1, n2, n3, n4)


N = n1+n2+n3+n4
Sigma =  N**(-2./(2+4))  

Sigma *= x1234.cov().values

k = KSE((Sigma))
x1234_values = x1234.values
K_multi = k(x1234_values[:,None,:],x1234_values[None,:,:])
L_ = L(n1, n2, n3, n4)
H_ = H(N)
# formulae (KLK + μI)−1KHK 
KLK = K_multi@L_@K_multi
KHK = K_multi@H_@K_multi
i = -6  # Start to regularize from 10^-6 on 
while True:
        try: 
               # Try to calculate eigenvalues and eigenvectors 
                vals, vecs = eigh(KLK, KHK)
        except np.linalg.linalg.LinAlgError:
                KHK += 10**i*np.identity(N)  # Regularize by adding identity 
                print(f'added {10**i} of identity to regularize KHK')
        else:
                break
        i += 1 # Increase the order of magnitude 

m = 6
# solve the 'reverse' gEVP
vals, vecs = eigh(KLK, KHK)

## Finding the leading eigen vectors, which is the map \phi
ind = np.argsort(abs(vals))
vecs_sort = vecs[:,ind]
p = vecs[:,-m:]

## Reconstruction of the features
x1_trans = ((p.T@K_multi) [:, 0:n1]).T #transposing for the sake of training. the original data shape(500,2), the reconstructed is (500,5)
x2_trans = ((p.T@K_multi) [:, n1:n1+n2]).T
x3_trans = ((p.T@K_multi) [:, n1+n2:n1+n2+n3]).T
x4_trans = ((p.T@K_multi) [:, n1+n2+n3:n1+n2+n3+n4]).T

##Training for original data
# Tunnel_1
#declaring exogenous and endogeneous
exod_1 = pd.DataFrame(x1).to_numpy()
endog_1 = pd.DataFrame(df1, columns = ['QDV_VMT_elevation_1_reference_point'])

# train test split
train_y, test_y = train_test_split(endog_1, train_size=2700)
train_x, test_x = train_test_split(exod_1, train_size= 2700)

# training and predicting with the same tunnel [original data]
arima_1 = pm.ARIMA(order=(10,2,0), seasonal_order=(0, 0, 0, 0))
arima_1.fit(train_y, train_x)
pred_h_1_org = arima_1.predict(test_y.shape[0], test_x, return_conf_int=True, alpha=0.05)
conf_int= pd.DataFrame(pred_h_1_org[1])

## to save model:
# import pickle
# with open('arima_1','wb') as f:
#     pickle.dump(arima_1,f)

# plot observation, prediction and confidence interval
fig = go.Figure()
fig.add_scatter(x=df1['QDV_VMT_tunnel_distance'][2600:2700], y=df1['QDV_VMT_elevation_1_reference_point'][2600:2700], name='Training Data', mode = 'markers')
fig.add_scatter(x=df1['QDV_VMT_tunnel_distance'][2700:2720], y=df1['QDV_VMT_elevation_1_reference_point'][2700:2720], name='Observation', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df1['QDV_VMT_tunnel_distance'][2700:2720], y=pred_h_1_org[0], name='Forecasts', mode = 'markers', opacity=0.7)
fig.update_xaxes(title_text="Tunnel Distance [m]")
fig.update_yaxes(title_text="Elevation[m]")
fig.update_layout(width=800, height = 400, legend=dict(yanchor="top", y=1, xanchor="left", x=0), margin=dict(b=0, t=0, l=0, r=0))
fig.show()

## Tunnel_2
#declaring exogenous and endogeneous
exod_2 = pd.DataFrame(x2).to_numpy()
endog_2 = pd.DataFrame(df2, columns = ['QDV_VMT_elevation_1_reference_point'])

# train test split
train_y, test_y = train_test_split(endog_2, train_size=1640)
train_x, test_x = train_test_split(exod_2, train_size= 1640)

# training and predicting with the same tunnel [original data]
arima_2 = pm.ARIMA(order=(10,2,0), seasonal_order=(0, 0, 0, 0))
arima_2.fit(train_y, train_x)
pred_h_2_org = arima_2.predict(test_y.shape[0], test_x, return_conf_int=True, alpha=0.05)
conf_int= pd.DataFrame(pred_h_2_org[1])

# ## to save model:
# with open('arima_2','wb') as f:
#     pickle.dump(arima_2,f)

# plot observation, prediction and confidence interval
fig = go.Figure()
fig.add_scatter(x=df2['QDV_VMT_tunnel_distance'][1500:1640], y=df2['QDV_VMT_elevation_1_reference_point'][1500:1640], name='Training Data', mode = 'markers')
fig.add_scatter(x=df2['QDV_VMT_tunnel_distance'][1640:1660], y=df2['QDV_VMT_elevation_1_reference_point'][1640:1660], name='Observation', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df2['QDV_VMT_tunnel_distance'][1640:1660], y=pred_h_2_org[0], name='Forecasts', mode = 'markers', opacity=0.7)
fig.update_xaxes(title_text="Tunnel Distance [m]")
fig.update_yaxes(title_text="Elevation[m]")
fig.update_layout(width=800, height = 400, legend=dict(yanchor="top", y=1, xanchor="left", x=0), margin=dict(b=0, t=0, l=0, r=0))
fig.show()

## Tunnel_3
#declaring exogenous and endogeneous
exod_3 = pd.DataFrame(x3).to_numpy()
endog_3 = pd.DataFrame(df3, columns = ['QDV_VMT_elevation_1_reference_point'])

# train test split
train_y, test_y = train_test_split(endog_3, train_size=1880)
train_x, test_x = train_test_split(exod_3, train_size= 1880)

# training and predicting with the same tunnel [original data]
arima_3 = pm.ARIMA(order=10,2,0), seasonal_order=(0, 0, 0, 0))
arima_3.fit(train_y, train_x)
pred_h_3_org = arima_3.predict(test_y.shape[0], test_x, return_conf_int=True, alpha=0.05)
conf_int= pd.DataFrame(pred_h_3_org[1])

# ## to save model:
# with open('arima_3','wb') as f:
#     pickle.dump(arima_3,f)

# plot observation, prediction and confidence interval
fig = go.Figure()
fig.add_scatter(x=df3['QDV_VMT_tunnel_distance'][1800:1880], y=df3['QDV_VMT_elevation_1_reference_point'][1800:1880], name='Training Data', mode = 'markers')
fig.add_scatter(x=df3['QDV_VMT_tunnel_distance'][1880:1900], y=df3['QDV_VMT_elevation_1_reference_point'][1880:1900], name='Observation', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df3['QDV_VMT_tunnel_distance'][1880:1900], y=pred_h_3_org[0], name='Forecasts', mode = 'markers', opacity=0.7)
fig.update_xaxes(title_text="Tunnel Distance [m]")
fig.update_yaxes(title_text="Elevation[m]")
fig.update_layout(width=800, height = 400, legend=dict(yanchor="top", y=1, xanchor="left", x=0), margin=dict(b=0, t=0, l=0, r=0))
fig.show()

## Tunnel_4
#declaring exogenous and endogeneous
exod_4 = pd.DataFrame(x4).to_numpy()
endog_4 = pd.DataFrame(df4, columns = ['QDV_VMT_elevation_1_reference_point'])

# train test split
train_y, test_y = train_test_split(endog_4, train_size=3500)
train_x, test_x = train_test_split(exod_4, train_size= 3500)

# training and predicting with the same tunnel [original data]
arima_4 = pm.ARIMA(order=(10,2,0), seasonal_order=(0, 0, 0, 0))
arima_4.fit(train_y, train_x)
pred_h_4_org = arima_4.predict(test_y.shape[0], test_x, return_conf_int=True, alpha=0.05)
conf_int= pd.DataFrame(pred_h_4_org[1])

# ## to save model:
# with open('arima_4','wb') as f:
#     pickle.dump(arima_4,f)

# plot observation, prediction and confidence interval
fig = go.Figure()
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][3400:3500], y=df4['QDV_VMT_elevation_1_reference_point'][3400:3500], name='Training Data', mode = 'markers')
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][3500:3520], y=df4['QDV_VMT_elevation_1_reference_point'][3500:3520], name='Observation', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][3500:3520], y=pred_h_4_org [0], name='Forecasts', mode = 'markers', opacity=0.7)
fig.update_xaxes(title_text="Tunnel Distance [m]")
fig.update_yaxes(title_text="Elevation[m]")
fig.update_layout(width=800, height = 400, legend=dict(yanchor="top", y=1, xanchor="right", x=1), margin=dict(b=0, t=0, l=0, r=0))
fig.show()

params_org = pd.concat([arima_1.params(), arima_2.params(), arima_3.params(), arima_4.params()], axis=1)
params_org.columns = ['arima1_org', 'arima2_org','arima3_org', 'arima4_org']
## plot coeff
fig = go.Figure()
fig.add_trace(go.Bar(
        x=params_org.index[3:13],
        y=params_org.arima1_org[3:13], name = 'TBM A tunnel 1'))
fig.add_trace(go.Bar(
        x=params_org.index[3:13],
        y=params_org.arima2_org[3:13], name = 'TBM A tunnel 2'))
fig.add_trace(go.Bar(
        x=params_org.index[3:13],
        y=params_org.arima3_org[3:13], name = 'TBM A tunnel 3'))
fig.add_trace(go.Bar(
        x=params_org.index[3:13],
        y=params_org.arima4_org[3:13], name = 'TBM B tunnel 1'))
fig.show()

fig = go.Figure()
fig.add_trace(go.Bar(
        x=params_org.index[1:2],
        y=params_org.arima1_org[1:2], name = 'TBM A tunnel 1'))
fig.add_trace(go.Bar(
        x=params_org.index[1:2],
        y=params_org.arima2_org[1:2], name = 'TBM A tunnel 2'))
fig.add_trace(go.Bar(
        x=params_org.index[1:2],
        y=params_org.arima3_org[1:2], name = 'TBM A tunnel 3'))
fig.add_trace(go.Bar(
        x=params_org.index[1:2],
        y=params_org.arima4_org[1:2], name = 'TBM B tunnel 1'))
fig.show()

fig = go.Figure()
fig.add_trace(go.Bar(
        x=params_org.index[2:3],
        y=params_org.arima1_org[2:3], name = 'TBM A tunnel 1'))
fig.add_trace(go.Bar(
        x=params_org.index[2:3],
        y=params_org.arima2_org[2:3], name = 'TBM A tunnel 2'))
fig.add_trace(go.Bar(
        x=params_org.index[2:3],
        y=params_org.arima3_org[2:3], name = 'TBM A tunnel 3'))
fig.add_trace(go.Bar(
        x=params_org.index[2:3],
        y=params_org.arima4_org[2:3], name = 'TBM B tunnel 1'))
fig.show()

### Exchanging model coefficients to make predictions [trans]

# ar4 to ar2
arima_4.params()[:] = arima_2.params()

pred_h_42_org = arima_4.arima_res_.get_prediction(start = 10, end = 30, exog = exod_4[9:30], dynamic = True)

# plot observation, prediction and confidence interval
fig = go.Figure()
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][:120], y=df4['QDV_VMT_elevation_1_reference_point'][:120], name='Observation', mode = 'markers')
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][10:30], y=pred_h_42_org.predicted_mean, name='Forecasts', mode = 'markers')
fig.update_xaxes(title_text="Tunnel Distance [m]")
fig.update_yaxes(title_text="Elevation[m]")
fig.update_layout(width=800, height = 400, legend=dict(yanchor="top", y=1, xanchor="right", x=1), margin=dict(b=0, t=0, l=0, r=0))
fig.show()

################################################################################################################################################
###Training for transformed data
## Tunnel_1
#declaring exogenous and endogeneous
exod_1_trans = pd.DataFrame(x1_trans).to_numpy()
endog_1 = pd.DataFrame(df1, columns = ['QDV_VMT_elevation_1_reference_point'])

# train test split
train_y, test_y = train_test_split(endog_1, train_size=2700)
train_x, test_x = train_test_split(exod_1_trans, train_size= 2700)

# training and predicting with the same tunnel [transformed data]
arima_1_trans = pm.ARIMA(order=(10,2,0), seasonal_order=(0, 0, 0, 0))
arima_1_trans.fit(train_y, train_x)
pred_h_1_trans = arima_1_trans.predict(test_y.shape[0], test_x, return_conf_int=True, alpha=0.05)
conf_int= pd.DataFrame(pred_h_1_trans[1])

# ## to save model:
# import pickle
# with open('arima_1_trans','wb') as f:
#     pickle.dump(arima_1_trans,f)

# plot observation, prediction and confidence interval
fig = go.Figure()
fig.add_scatter(x=df1['QDV_VMT_tunnel_distance'][2600:2700], y=df1['QDV_VMT_elevation_1_reference_point'][2600:2700], name='Training Data', mode = 'markers')
fig.add_scatter(x=df1['QDV_VMT_tunnel_distance'][2700:2720], y=df1['QDV_VMT_elevation_1_reference_point'][2700:2720], name='Observation', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df1['QDV_VMT_tunnel_distance'][2700:2720], y=pred_h_1_org[0], name='Forecasts before Transfer', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df1['QDV_VMT_tunnel_distance'][2700:2720], y=pred_h_1_trans[0], name='Forecasts after Transfer', mode = 'markers', opacity=0.7)
fig.update_xaxes(title_text="Tunnel Distance [m]")
fig.update_yaxes(title_text="Elevation[m]")
fig.update_layout(width=800, height = 400, legend=dict(yanchor="top", y=1, xanchor="left", x=0), margin=dict(b=0, t=0, l=0, r=0))
fig.show()

## Tunnel_2
#declaring exogenous and endogeneous
exod_2_trans = pd.DataFrame(x2_trans).to_numpy()
endog_2 = pd.DataFrame(df2, columns = ['QDV_VMT_elevation_1_reference_point'])

# train test split
train_y, test_y = train_test_split(endog_2, train_size=1640)
train_x, test_x = train_test_split(exod_2_trans, train_size=1640)

# training and predicting with the same tunnel [transformed data]
arima_2_trans = pm.ARIMA(order=(10,2,0), seasonal_order=(0, 0, 0, 0))
arima_2_trans.fit(train_y, train_x)
pred_h_2_trans = arima_2_trans.predict(test_y.shape[0], test_x, return_conf_int=True, alpha=0.05)
conf_int= pd.DataFrame(pred_h_2_trans[1])

# ## to save model:
# import pickle
# with open('arima_2_trans','wb') as f:
#     pickle.dump(arima_2_trans,f)

# plot observation, prediction and confidence interval
fig = go.Figure()
fig.add_scatter(x=df2['QDV_VMT_tunnel_distance'][1500:1640], y=df2['QDV_VMT_elevation_1_reference_point'][1500:1640], name='Training Data', mode = 'markers')
fig.add_scatter(x=df2['QDV_VMT_tunnel_distance'][1640:1660], y=df2['QDV_VMT_elevation_1_reference_point'][1640:1660], name='Observation', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df2['QDV_VMT_tunnel_distance'][1640:1660], y=pred_h_2_org[0], name='Forecasts before Transfer', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df2['QDV_VMT_tunnel_distance'][1640:1660], y=pred_h_2_trans[0], name='Forecasts after Transfer', mode = 'markers', opacity=0.7)
fig.update_xaxes(title_text="Tunnel Distance [m]")
fig.update_yaxes(title_text="Elevation[m]")
fig.update_layout(width=800, height = 400, legend=dict(yanchor="top", y=1, xanchor="left", x=0), margin=dict(b=0, t=0, l=0, r=0))
fig.show()

## Tunnel_3
#declaring exogenous and endogeneous
exod_3_trans = pd.DataFrame(x3_trans).to_numpy()
endog_3 = pd.DataFrame(df3, columns = ['QDV_VMT_elevation_1_reference_point'])

# train test split
train_y, test_y = train_test_split(endog_3, train_size=1880)
train_x, test_x = train_test_split(exod_3_trans, train_size=1880)

# training and predicting with the same tunnel [transformed data]
arima_3_trans = pm.ARIMA(order=(10,2,0), seasonal_order=(0, 0, 0, 0))
arima_3_trans.fit(train_y, train_x)
pred_h_3_trans = arima_3_trans.predict(test_y.shape[0], test_x, return_conf_int=True, alpha=0.05)
conf_int= pd.DataFrame(pred_h_3_trans[1])

# ## to save model:
# import pickle
# with open('arima_3_trans','wb') as f:
#     pickle.dump(arima_3_trans,f)

# plot observation, prediction and confidence interval
fig = go.Figure()
fig.add_scatter(x=df3['QDV_VMT_tunnel_distance'][1800:1880], y=df3['QDV_VMT_elevation_1_reference_point'][1800:1880], name='Training Data', mode = 'markers')
fig.add_scatter(x=df3['QDV_VMT_tunnel_distance'][1880:1900], y=df3['QDV_VMT_elevation_1_reference_point'][1880:1900], name='Observation', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df3['QDV_VMT_tunnel_distance'][1880:1900], y=pred_h_3_org[0], name='Forecasts before Transfer', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df3['QDV_VMT_tunnel_distance'][1880:1900], y=pred_h_3_trans[0], name='Forecasts after Transfer', mode = 'markers', opacity=0.7)
fig.update_xaxes(title_text="Tunnel Distance [m]")
fig.update_yaxes(title_text="Elevation[m]")
fig.update_layout(width=800, height = 400, legend=dict(yanchor="top", y=1, xanchor="left", x=0), margin=dict(b=0, t=0, l=0, r=0))
fig.show()

## Tunnel_4
#declaring exogenous and endogeneous
exod_4_trans = pd.DataFrame(x4_trans).to_numpy()
endog_4 = pd.DataFrame(df4, columns = ['QDV_VMT_elevation_1_reference_point'])

# train test split
train_y, test_y = train_test_split(endog_4, train_size=3500)
train_x, test_x = train_test_split(exod_4_trans, train_size=3500)

# training and predicting with the same tunnel [transformed data]
arima_4_trans = pm.ARIMA(order=(10,2,0), seasonal_order=(0, 0, 0, 0))
arima_4_trans.fit(train_y, train_x)
pred_h_4_trans = arima_4_trans.predict(test_y.shape[0], test_x, return_conf_int=True, alpha=0.05)
conf_int= pd.DataFrame(pred_h_4_trans[1])

# ## to save model:
# import pickle
# with open('arima_4_trans','wb') as f:
#     pickle.dump(arima_4_trans,f)

# plot observation, prediction and confidence interval
fig = go.Figure()
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][3400:3500], y=df4['QDV_VMT_elevation_1_reference_point'][3400:3500], name='Training Data', mode = 'markers')
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][3500:3520], y=df4['QDV_VMT_elevation_1_reference_point'][3500:3520], name='Observation', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][3500:3520], y=pred_h_4_org[0], name='Forecasts before Transfer', mode = 'markers', opacity=0.7)
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][3500:3520], y=pred_h_4_trans[0], name='Forecasts after Transfer', mode = 'markers', opacity=0.7)
fig.update_xaxes(title_text="Tunnel Distance [m]")
fig.update_yaxes(title_text="Elevation[m]")
fig.update_layout(width=800, height = 400, legend=dict(yanchor="top", y=1, xanchor="right", x=1), margin=dict(b=0, t=0, l=0, r=0))
fig.show()

params_trans = pd.concat([arima_1_trans.params(), arima_2_trans.params(), arima_3_trans.params(), arima_4_trans.params()], axis=1)
params_trans.columns = ['arima1_trans', 'arima2_trans','arima3_trans','arima4_trans']

# plot coeff
fig = go.Figure()
fig.add_trace(go.Bar(
        x=params_trans.index[7:17],
        y=params_trans.arima1_trans[7:17], name = 'TBM A tunnel 1'))
fig.add_trace(go.Bar(
        x=params_trans.index[7:17],
        y=params_trans.arima2_trans[7:17], name = 'TBM A tunnel 2'))
fig.add_trace(go.Bar(
        x=params_trans.index[7:17],
        y=params_trans.arima3_trans[7:17], name = 'TBM A tunnel 3'))
fig.add_trace(go.Bar(
        x=params_trans.index[7:17],,
        y=params_trans.arima4_trans[7:17], name = 'TBM B tunnel 1'))
fig.show()

fig = go.Figure()
fig.add_trace(go.Bar(
        x=params_trans.index[1:7],
        y=params_trans.arima1_trans[1:7], name = 'TBM A tunnel 1'))
fig.add_trace(go.Bar(
        x=params_trans.index[1:7],
        y=params_trans.arima2_trans[1:7], name = 'TBM A tunnel 2'))
fig.add_trace(go.Bar(
        x=params_trans.index[1:7],
        y=params_trans.arima3_trans[1:7], name = 'TBM A tunnel 3'))
fig.add_trace(go.Bar(
        x=params_trans.index[1:7],
        y=params_trans.arima4_trans[1:7], name = 'TBM B tunnel 1'))
fig.show()

##########################################################################################################
### Exchanging model coefficients to make predictions [trans]
# ar4 to ar2
arima_4_trans.params()[:] = arima_2_trans.params()

pred_h = arima_4_trans.arima_res_.get_prediction(start = 10, end = 30, exog = exod_4_trans[9:30], dynamic = True)

# plot observation, prediction and confidence interval
fig = go.Figure()
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][:120], y=df4['QDV_VMT_elevation_1_reference_point'][:120], name='Observation', mode = 'markers')
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][10:30], y=pred_h_42_org.predicted_mean, name='Forecasts before Transfer', mode = 'markers')
fig.add_scatter(x=df4['QDV_VMT_TBM_station'][10:30], y=pred_h.predicted_mean, name='Forecasts after Transfer', mode = 'markers')
fig.update_xaxes(title_text="Tunnel Distance [m]")
fig.update_yaxes(title_text="Elevation[m]")
fig.update_layout(width=800, height = 400, legend=dict(yanchor="top", y=1, xanchor="right", x=1), margin=dict(b=0, t=0, l=0, r=0))
fig.show()

# saving the transformed features
# pd.DataFrame(x1_trans).to_excel('TBM-A/tunnel-1/trans.xlsx')
# pd.DataFrame(x2_trans).to_excel('TBM-A/tunnel-2/trans.xlsx')
# pd.DataFrame(x3_trans).to_excel('TBM-A/tunnel-3/trans.xlsx')
# pd.DataFrame(x4_trans).to_excel('TBM-B/tunnel-1/trans.xlsx')
