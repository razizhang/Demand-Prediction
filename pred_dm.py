#coding=utf-8
from keras.models import model_from_json

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

import os
import sys
import gc
from io import StringIO,BytesIO

from datetime import datetime, timedelta



# input reshape with multiple locations
def inReshapeMulLoc(data, nTs, nPred, nF):
    data_wide = data.pivot_table(index=['lng', 'lat'], columns='time', fill_value=0)
    data_wide = data_wide.reset_index().values
    del data
    gc.collect()

    nL = data_wide.shape[0]
    nS = int((data_wide.shape[1] - 2) / nF)
    # uncomment if more than one day
    # nD = int(nS / nTs)

    X = np.zeros([nL, nTs, nF + 2], dtype='float32')

    # extract features from wide table
    for i in range(data_wide.shape[0]):
        row = data_wide[i]

        X_sub = np.zeros([nS, nF + 2], dtype='float32')
        X_sub[:, 0] = row[0]
        X_sub[:, 1] = row[1]
        for j in range(nF):
            X_sub[:, 2 + j] = row[2 + nS * j:2 + nS * (j + 1)]

        X[i:(i + 1)] = X_sub[:24]

    return X

# save prediction with loc and time columns
def print_pred(data, nDay, pred):
    data_wide = data.pivot_table(index=['lng', 'lat'], columns='time', fill_value=0)

    del data
    gc.collect()

    #dm = data_wide.iloc[:, 24:24 + nDay * 24].values.reshape(-1, 1)
    #sp = data_wide.iloc[:, (1 + nDay) * 24 * 2 + 24:].values.reshape(-1, 1)

    df_time = data_wide.columns.values[:24]
    df_time = pd.DataFrame(df_time, columns=['time'])
    df_time = df_time['time'].apply(lambda x: x[1]).values

    df_loc = data_wide.reset_index().iloc[:, 0:2].values

    len_time = 24
    len_loc = len(df_loc)

    df_time = np.tile(df_time, len_loc).reshape(len_time * len_loc, 1)
    df_loc = np.concatenate(np.tile(df_loc, len_time).reshape(len_loc, len_time, 2))

    df_out = np.concatenate((df_loc, df_time, pred.reshape(-1, 1)), axis=1)
    del df_loc, df_time
    gc.collect()
    #df_out = pd.DataFrame(df_out, columns=['lng', 'lat', 'time', 'dm_pred', 'dm', 'sp'])
    df_out = pd.DataFrame(df_out, columns=['lng', 'lat', 'time', 'dm_pred'])

    #df_out.to_csv('pred_result_s.csv', index=False)
    return df_out



# params
test_set_dir = '/data/zhangruizhe/Python/DemandPrediction/DMData/test_set/'
model_dir = '/data/zhangruizhe/Python/DemandPrediction/Model/'
model_num = '1'
#city = 'cd'


nTs = 24           # number of timesteps to lookback
nPred = 24         # number of timesteps to predict
nF = 4             # number of features

# city area bound
area = {'bj':['116.170', '116.610', '39.794', '40.030'],
        'cd':['103.818', '104.280', '30.554', '30.859'],
        'sh':['121.300', '121.703', '31.127', '31.346'],
        'gz':['113.179', '113.473', '22.982', '23.222'],
        'sz':['113.800', '114.176', '22.456', '22.643']}

# date of prediction
date = datetime.strptime(sys.argv[1], '%Y%m%d') - timedelta(days=1)
date_st = date.strftime('%Y%m%d')
date_ed = (date + timedelta(days=1)).strftime('%Y%m%d')
date_cons = (date + timedelta(days=1)-timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')


def pred_dm_one_city(city):

    # load json and create model
    model_path = model_dir + 'model_' + city + '_' + model_num + '.json'
    weight_path = model_dir + 'model_' + city + '_' + model_num + '.h5'
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_path)
    print("Loaded model from disk")


    # TODO: Re-implement below, Extract test data from ops.dm_bike_index_one_hour
    bound = area[city]

    hive_qry = """
    hive -e \
    "select     lng, lat, 
                index_datetime as time, 
                sum(start_num) + sum(end_num) as flow,
                sum(demand_num) as dm 
     from       ops.dm_bike_index_one_hour 
     where 	    date_time = '{0}' and 
                index_datetime < '{1}' and 
                lng between '{l}' and '{r}' and
                lat between '{d}' and '{u}' 
     group by   lng,lat,index_datetime 
     order by   lng,lat,index_datetime" -S
    """.format(date_st, date_cons, l=bound[0], r=bound[1], d=bound[2], u=bound[3])


    f = os.popen('''{cmd}'''.format(cmd=hive_qry))
    df_data = pd.read_csv(StringIO(f.read()),
                          sep='\t',
                          names=['lng', 'lat', 'time', 'flow', 'dm'])
    f.close()
    print('Data Loaded')
    nLoc = int(df_data['lng'].unique().shape[0] * df_data['lat'].unique().shape[0])
    nDay = int(np.ceil(df_data['time'].unique().shape[0] / 24)) - 1
    print('Area Size:', nLoc, 'Blocks')
    print('Period:', nDay+1, 'Days')


    '''
    # load test data
    test_set_path = test_set_dir+city+'_dm_block_l_test.csv'
    df_data = pd.read_csv(test_set_path, usecols= ['lng','lat','time', 'sp','flow', 'dm'])
    print('Data Loaded')
    print(df_data.shape)
    nLoc = int(df_data['lng'].unique().shape[0] * df_data['lat'].unique().shape[0])
    nDay = int(np.ceil(df_data['time'].unique().shape[0] / 24)) - 1
    print('Area Size:', nLoc, 'Blocks')
    print('Period:', nDay+1, 'Days')
    
    '''


    # partition feature
    print('Start to Extract Features...')

    X = inReshapeMulLoc(df_data, nTs, nPred, nF-2)
    #del df_data
    #gc.collect()

    print('Done! Extracted ' + str(X.shape[0]) + ' samples.')


    # make prediction and print MAE
    pred = np.abs(loaded_model.predict(X[:,:,0:4]))

    # floor the prediciton value
    print('Predicting Demand...')
    pred = np.round(pred,0)
    print('Done!')

    df_result = print_pred(df_data, 1, pred)

    return df_result



# Predict demand for five big cities ----------------------------------------------------------------------

df_result = pd.DataFrame(columns=['lng', 'lat', 'time', 'dm_pred'])

for city in area.keys():
    df_result = df_result.append(pred_dm_one_city(city), ignore_index=True)
    print(df_result.shape)

#df_result = df_result.append(pred_dm_one_city('sz'), ignore_index=True)

df_result['dm_pred'] = df_result['dm_pred'].astype('float64')

#TODO: write prediction back to hive bi_compass.dm_bike_index_one_hour
#pred_dm_one_city('sh')


# Overwrite prediction results back to Hive dataset -------------------------------------------------------
#read data from Hive, citytype = 0
bound = area['cd']
hive_qry = """
hive -e \
"select     * 
 from       bi_compass.dm_bike_index_one_hour 
 where 	    citytype = 0 and
            date_time = '{0}' and
            biketype = '0' " -S
""".format(date_ed,l=bound[0], r=bound[1], d=bound[2], u=bound[3])

f = os.popen('''{cmd}'''.format(cmd=hive_qry))
df_data = pd.read_csv(StringIO(f.read()),
                      sep='\t',
                      dtype={'adcode': np.str,'citycode': np.str},
                      names=['lng', 'lat', 'index_datetime', 'biketype',
                             'start_num', 'end_num','supply_num', 'demand_num',
                             'country', 'province', 'city', 'district','adcode',
                             'township', 'towncode', 'neighborhood', 'street',
                             'streetnumber', 'citycode', 'date_time', 'citytype'])
f.close()


# change the name of time column
df_result.rename(columns={'time':'index_datetime'}, inplace = True)
# change datetime to a week ago
data_prev = (date + timedelta(days=1)).strftime('%Y-%m-%d')
df_result['index_datetime'] = df_result['index_datetime'].apply(lambda x: data_prev+ x[10:])

#df_result left join with the read
df_out = df_data.merge(df_result, on=['lng', 'lat', 'index_datetime'], how='left')

df_out['lng'] = np.round(df_out['lng'], 3)
df_out['lat'] = np.round(df_out['lat'], 3)

# replace demand_num if prediction exists
has_no_pred = np.isnan(df_out['dm_pred'])
df_out['dm_pred'] = df_out['dm_pred'].fillna(0)
df_out['dm'] = df_out['demand_num'] * has_no_pred + df_out['dm_pred'] * np.logical_not(has_no_pred)

# change column names as shown in the original table
df_out.drop(['demand_num', 'dm_pred'],axis=1, inplace=True)
df_out.rename(columns={'dm':'demand_num'}, inplace = True)

#
df_out['adcode'] = df_out['adcode'].astype('str')

print("Update " + str(sum(np.logical_not(has_no_pred))) + "/" + str(df_data.shape[0]) + " records")



#save as a file...
df_out.to_csv('/data/zhangruizhe/Python/DemandPrediction/updated_pred.csv',
              columns=['lng', 'lat', 'index_datetime', 'biketype', 'start_num', 'end_num',
                       'supply_num', 'demand_num', 'country', 'province', 'city', 'district',
                       'adcode', 'township', 'towncode', 'neighborhood', 'street', 'streetnumber',
                       'citycode'],
              header=False,
              index=False)
print('Saved to csv file')

#insert overwrite, write back to hive
db = 'bi_compass'
tb = 'dm_bike_index_one_hour'
date_time = date_ed
city_type = '0'
fn = '/data/zhangruizhe/Python/DemandPrediction/updated_pred.csv'

cmd1 = 'hive -e "use {0}; alter table {1} drop partition (date_time="{2}", citytype="{3}");"'.format(db,tb,date_time,city_type)
cmd2 = 'hadoop fs -mkdir hdfs://nameservice1/user/hive/warehouse/{0}.db/{1}/date_time={2}/citytype={3}'.format(db,tb,date_time,city_type)
cmd3 = 'hadoop fs -put {0} hdfs://nameservice1/user/hive/warehouse/{1}.db/{2}/date_time={3}/citytype={4}'.format(fn,db,tb,date_time,city_type)
cmd4 = 'hive -e "use {0}; alter table {1} add partition (date_time="{2}", citytype="{3}");"'.format(db,tb,date_time,city_type)

os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
os.system(cmd4)