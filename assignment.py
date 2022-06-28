import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# the function below helps to create a data based on the (longitude, latitude) of a city
def dataframe_builder(longitude, latitude):
    ozone_emmision = requests.get(
        'https://api.v2.emissions-api.org/api/v2/ozone/average.json?interval=day&begin=2020-01-01&end=2020-01-31&point={},{}'.format(
            longitude, latitude)).json()
    carbonmonoxide_emmision = requests.get(
        'https://api.v2.emissions-api.org/api/v2/carbonmonoxide/average.json?interval=day&begin=2020-01-01&end=2020-01-31&point={},{}'.format(
            longitude, latitude)).json()
    nitrogendioxide_emmision = requests.get(
        'https://api.v2.emissions-api.org/api/v2/nitrogendioxide/average.json?interval=day&begin=2020-01-01&end=2020-01-31&point={},{}'.format(
            longitude, latitude)).json()
    methane_emmision = requests.get(
        'https://api.v2.emissions-api.org/api/v2/methane/average.json?interval=day&begin=2020-01-01&end=2020-01-31&point={},{}'.format(
            longitude, latitude)).json()

    emmision = {}  # final dictionary

    for it in ozone_emmision:
        date = it['end'].split('T')[0].split('-')[-1]
        if date not in emmision.keys():
            emmision[date] = {}
            emmision[date]['ozone'] = it['average']
        else:
            emmision[date]['ozone'] = it['average']

    for it in carbonmonoxide_emmision:
        date = it['end'].split('T')[0].split('-')[-1]
        if date not in emmision.keys():
            emmision[date] = {}
            emmision[date]['carbonmonoxide'] = it['average']
        else:
            emmision[date]['carbonmonoxide'] = it['average']

    for it in nitrogendioxide_emmision:
        date = it['end'].split('T')[0].split('-')[-1]
        if date not in emmision.keys():
            emmision[date] = {}
            emmision[date]['nitrogendioxide'] = it['average']
        else:
            emmision[date]['nitrogendioxide'] = it['average']

    for it in methane_emmision:
        date = it['end'].split('T')[0].split('-')[-1]
        if date not in emmision.keys():
            emmision[date] = {}
            emmision[date]['methane'] = it['average']
        else:
            emmision[date]['methane'] = it['average']

    emmision_dataframe = pd.DataFrame(emmision)
    # transposing the dataframe
    emmision_dataframe = emmision_dataframe.T
    # sorting the dataframe
    emmision_dataframe = emmision_dataframe.sort_index()
    # storing days list
    day_list = emmision_dataframe.index
    day_list = [int(itr) for itr in day_list]

    emmision_dataframe['day'] = day_list
    return emmision_dataframe



# JANUARY 2020 EMMISIONS REPORT
january_emmision = {}

# New Delhi - 28.7041° N, 77.1025° E
january_emmision['delhi_emmision_data'] = dataframe_builder(77.1025, 28.7041)
# Tokyo - 35.6762° N, 139.6503° E
january_emmision['tokyo_emmision_data'] = dataframe_builder(139.6503, 35.6762)
# Jakarta - 6.2088° S, 106.8456° E
january_emmision['jakarta_emmision_data'] = dataframe_builder(106.8456, -6.2088)
# Manila - 14.5995° N, 120.9842° E
january_emmision['manila_emmision_data'] = dataframe_builder(120.9842, 14.5995)
# Seoul - 37.5665° N, 126.9780° E
january_emmision['seoul_emmision_data'] = dataframe_builder(126.9780, 37.5665)
# Sanghai - 31.2304° N, 121.4737° E
january_emmision['sanghai_emmision_data'] = dataframe_builder(121.4737, 31.2304)
# Karachi - 24.8607° N, 67.0011° E
january_emmision['karachi_emmision_data'] = dataframe_builder(67.0011, 24.8607)
# Beijing - 39.9042° N, 116.4074° E
january_emmision['beijing_emmision_data'] = dataframe_builder(116.4074, 39.9042)
# NYC - 40.7128° N, 74.0060° W
january_emmision['nyc_emmision_data'] = dataframe_builder(-74.0060, 40.7128)
# Guangzhou - 23.1291° N, 113.2644° E
january_emmision['guangzhou_emmision_data'] = dataframe_builder(113.2644, 23.1291)
# Sao Paulo - 23.5558° S, 46.6396° W
january_emmision['saopaulo_emmision_data'] = dataframe_builder(-46.6396, -23.5558)
# Mexico city - 19.4326° N, 99.1332° W
january_emmision['mexico_emmision_data'] = dataframe_builder(-99.1332, 19.4326)
# Mumbai - 19.0760° N, 72.8777° E
january_emmision['mumbai_emmision_data'] = dataframe_builder(72.8777, 19.0760)
# Kyoto - 35.0116° N, 135.7681° E
january_emmision['kyoto_emmision_data'] = dataframe_builder(135.7681, 35.0116)
# Moscow - 55.7558° N, 37.6173° E
january_emmision['moscow_emmision_data'] = dataframe_builder(37.6173, 55.7558)
# Dhaka - 23.8103° N, 90.4125° E
january_emmision['dhaka_emmision_data'] = dataframe_builder(90.4125, 23.8103)
# Cairo - 30.0444° N, 31.2357° E
january_emmision['cairo_emmision_data'] = dataframe_builder(31.2357, 30.0444)
# LA - 34.0522° N, 118.2437° W
january_emmision['la_emmision_data'] = dataframe_builder(-118.2437, 34.0522)
# Bangkok - 13.7563° N, 100.5018° E
january_emmision['bangkok_emmision_data'] = dataframe_builder(100.5018, 13.7563)
# Kolkata - 22.5726° N, 88.3639° E
january_emmision['kolkata_emmision_data'] = dataframe_builder(88.3639, 22.5726)

# CITY LIST
city_list = ['delhi', 'tokyo', 'jakarta', 'manila', 'seoul', 'sanghai', 'karachi', 'beijing', 'nyc', 'guangzhou',
             'saopaulo', 'mexico', 'mumbai', 'kyoto', 'moscow', 'dhaka', 'cairo', 'la', 'bangkok', 'kolkata']



methane_emmision_january = {}
for itr in city_list :
    city = itr + '_emmision_data'
    if january_emmision[city].shape[1] == 5 :
        methane_emmision_january[itr] = january_emmision[city]['methane'].mean()

ozone_emmision_january = {}
for itr in city_list :
    city = itr + '_emmision_data'
    ozone_emmision_january[itr] = january_emmision[city]['ozone'].mean()

carbonmonoxide_emmision_january = {}
for itr in city_list :
    city = itr + '_emmision_data'
    carbonmonoxide_emmision_january[itr] = january_emmision[city]['carbonmonoxide'].mean()

nitrogendioxide_emmision_january = {}
for itr in city_list :
    city = itr + '_emmision_data'
    nitrogendioxide_emmision_january[itr] = january_emmision[city]['nitrogendioxide'].mean()

# CREATING DATASET CITY WISE AVERAGE
def top_x_content(gas_emmision_january, fname, top_x=5):
    gas_emmision_keys = gas_emmision_january.keys()
    gas_emmision_values = gas_emmision_january.values()

    gas_emmision_january_reverse = {}
    for key in gas_emmision_january.keys():
        gas_emmision_january_reverse[gas_emmision_january[key]] = key

    gas_emmision_keys_new = sorted(gas_emmision_january_reverse.keys(), reverse=True)[:top_x]
    gas_emmision_cities = []

    for it in gas_emmision_keys_new:
        gas_emmision_cities.append(gas_emmision_january_reverse[it])

    gas_emmision_keys_new = [it - min(gas_emmision_values) for it in gas_emmision_keys_new]

    gas_emmision_cities.reverse()
    gas_emmision_keys_new.reverse()
    plt.figure(figsize=(10, 5))
    plt.barh(gas_emmision_cities, gas_emmision_keys_new)
    plt.savefig('{}.jpeg'.format(fname))
    plt.show()


top_x_content(ozone_emmision_january, 'ozone', -1)
top_x_content(carbonmonoxide_emmision_january, 'carbonmonoxide', -1)
top_x_content(nitrogendioxide_emmision_january, 'nitrogendioxide', -1)
top_x_content(methane_emmision_january, 'methane', -1)

# Unsupervised Clustering

# making dataset


data_diction = {}
for key in city_list :
    if key not in data_diction.keys() :
        data_diction[key] = {}
        data_diction[key]['ozone'] = ozone_emmision_january[key]
        data_diction[key]['carbonmonoxide'] = carbonmonoxide_emmision_january[key]
        data_diction[key]['nitrogendioxide'] = nitrogendioxide_emmision_january[key]
        if key in methane_emmision_january.keys() :
            data_diction[key]['methane'] = methane_emmision_january[key]
dataframe2 = pd.DataFrame(data_diction)
dataframe2 = dataframe2.T

# Preprocessing using minmax scaler

scaler = MinMaxScaler()

scaler.fit(dataframe2[['ozone']])
dataframe2['ozone'] = scaler.transform(dataframe2[['ozone']])

scaler.fit(dataframe2[['carbonmonoxide']])
dataframe2['carbonmonoxide'] = scaler.transform(dataframe2[['carbonmonoxide']])

scaler.fit(dataframe2[['nitrogendioxide']])
dataframe2['nitrogendioxide'] = scaler.transform(dataframe2[['nitrogendioxide']])

scaler.fit(dataframe2[['methane']])
dataframe2['methane'] = scaler.transform(dataframe2[['methane']])

# Calculating sum of squared error and choosing optimum number of clusters

sse = []
k_rng = range(1,21)
for k in k_rng :
    km = KMeans(n_clusters = k)
    km.fit(dataframe2[['ozone','carbonmonoxide', 'nitrogendioxide']]) # methane data is incomplete
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

# 5 clusters

kmeans = KMeans(n_clusters = 5)
y_predicted = kmeans.fit_predict(dataframe2[['ozone','carbonmonoxide', 'nitrogendioxide']])
dataframe2['cluster'] = y_predicted
dataframe2['city'] = dataframe2.index

cluster0 = dataframe2[dataframe2['cluster'] == 0]
cluster1 = dataframe2[dataframe2['cluster'] == 1]
cluster2 = dataframe2[dataframe2['cluster'] == 2]
cluster3 = dataframe2[dataframe2['cluster'] == 3]
cluster4 = dataframe2[dataframe2['cluster'] == 4]

# saving csv clusters
cluster0.to_csv('cluster0', index=False)
cluster1.to_csv('cluster1', index=False)
cluster2.to_csv('cluster2', index=False)
cluster3.to_csv('cluster3', index=False)
cluster4.to_csv('cluster4', index=False)