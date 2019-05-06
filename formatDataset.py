#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import math
with open("finaltrain1.json") as json_file:  
    data = json.load(json_file)


# ## Create a temporary dataframe with one row of zeros
# 
# * Also create a list of the features

# In[ ]:


zerodataset = pd.DataFrame(0,index=np.arange(1,2),columns=np.arange(43)) 
zerodataset.columns = ['image','scene','time','weather','alternate_drive','direct_drive',
                         'traff_lights_dist','traff_lights_angle','traff_lights_area','traff_lights_color',
                        'traff_sign_dist','traff_sign_angle','traff_sign_area',
                        'car_dist','car_angle','car_area',
                         'bike_dist','bike_angle','bike_area',
                         'bus_dist','bus_angle','bus_area',
                         'motor_dist','motor_angle','motor_area',
                         'person_dist','person_angle','person_area',
                         'rider_dist','rider_angle','rider_area',
                         'train_dist','train_angle','train_area',
                         'truck_dist','truck_angle','truck_area',
                         'left_lane_direction','left_lane_style','left_lane_type',
                         'right_lane_direction','right_lane_style','right_lane_type']


# ## Format the nested json data
# 
# * There is a lot of formatting going on here. This is mainly because the dataset is very complicated and nested. 
# * If you want to see the orignal unformatted dataset, open "bdd100k_labels_images_train.json"
# * The variables starting with "final..." are the final features I'm using

# In[ ]:



for countiter,i in enumerate(data):
    #print(countiter)
    firstDataFrame = pd.DataFrame(json_normalize(i))
    
    # labels has all the detections 
    labels1 = firstDataFrame['labels'].tolist()
    labelscorrect = [value for sublist in labels1 for value in sublist]
    df = pd.DataFrame(json_normalize(labelscorrect))
    
    # don't need these features
    df = df.drop(columns=['manualAttributes','manualShape','attributes.truncated','attributes.occluded'])
    
    firstDataFrame = firstDataFrame.drop(columns=['labels','timestamp'])
    
    firstDataFrame = firstDataFrame.reindex(columns= ['name','attributes.scene','attributes.timeofday',
                                                      'attributes.weather'])
    
    # for each object (there are 10 types of object detections), instead of having virtually useless coordinates,
    # condense to basically 3 features - object area, object angle, object distance all with resepect to the
    # camera frame of reference
    
    
    df['object_area']=abs(df['box2d.x1']-df['box2d.x2'])*abs(df['box2d.y1']-df['box2d.y2'])
    df['midx']=(df['box2d.x1']+df['box2d.x2'])/2
    df['midy']=(df['box2d.y1']+df['box2d.y2'])/2
    
    # frame of reference of the camera
    df['refx'] = 1280/2
    df['refy'] = 720
    df['object_dist']=(abs(df['midx']-df['refx'])**2 + abs(df['midy']-df['refy'])**2)**0.5

    df['object_angle']=np.arctan2(df['refy']-df['midy'],df['midx']-df['refx'])*180/math.pi
    
    row = df[df['category'].str.contains('drivable area')]
    row['poly2d']=0
    df[df['category'].str.contains('drivable area')] = row

    vertex=[]
    minvertex=[]
    try:
        for j in df['poly2d']:
           # print(type(i))
            if type(j)!=float and type(j)!=int:
                #print(i)

                for ii in j:
                    #print(ii.get('vertices'))
                    vertex.append(ii.get('vertices'))
                    minvertex.append(min(ii.get('vertices')))
            else:
                vertex.append(np.NaN)
        #print(minvertex)
        #print('---')
        #print(vertex)
        df['VERTEX']=pd.Series(vertex)
        dirr=[]
        minivertex=[]
        for k in vertex:
            #print(i[0])
            #print(i[0]-1280/2)
            if type(k) != float:

                if (min(k)[0]-640)<0:
                    dirr.append("left")
                else:
                    dirr.append("right")
                minivertex.append(abs(min(k)[0]-640))
            else:
                dirr.append(np.NaN)
                minivertex.append(np.NaN)

        df['direction'] = pd.Series(dirr)
        #minivertex
        df['minvert'] = pd.Series(minivertex)


        lanedf = df[df['category'].str.contains('lane')]

        # use try and except because not all images have the same number or type of detections
        try:
            left_lanesdf = lanedf[lanedf['direction'].str.contains('left')]
            final_left_lanesdf = left_lanesdf.loc[left_lanesdf['minvert']==min(left_lanesdf['minvert'])]
            final_left_lanesdf = final_left_lanesdf[['attributes.laneDirection','attributes.laneStyle',
                                                           'attributes.laneType']]
        except:
            final_left_lanesdf = pd.DataFrame({'left_lane_direction' : [np.nan],'left_lane_style':[np.NaN],'left_lane_type':[np.NaN]})


        try:
            right_lanesdf = lanedf[lanedf['direction'].str.contains('right')]
            final_right_lanesdf = right_lanesdf.loc[right_lanesdf['minvert']==min(right_lanesdf['minvert'])]
            final_right_lanesdf = final_right_lanesdf[['attributes.laneDirection','attributes.laneStyle',
                                                           'attributes.laneType']]
        except:
            final_right_lanesdf = pd.DataFrame({'right_lane_direction' : [np.nan],'right_lane_style':[np.NaN],'right_lane_type':[np.NaN]})

    #finalleftlanesdf
    except KeyError:
        final_right_lanesdf = pd.DataFrame({'right_lane_direction' : [np.nan],'right_lane_style':[np.NaN],'right_lane_type':[np.NaN]})
        final_left_lanesdf = pd.DataFrame({'left_lane_direction' : [np.nan],'left_lane_style':[np.NaN],'left_lane_type':[np.NaN]})

    try:
        drive = df['attributes.areaType'].value_counts()
        drive = drive.to_dict()
        direct = drive.get('direct')
    except:
        direct = np.NaN
    try:
        drive = df['attributes.areaType'].value_counts()
        drive = drive.to_dict()
        alternative = drive.get('alternative')
    except:
        alternative = np.NaN
    #print(alternative)

    try:
        traffic_lightsdf = df[df['category'].str.contains('traffic light')]
        final_traffic_lightsdf = traffic_lightsdf.loc[traffic_lightsdf['object_dist']==min(traffic_lightsdf['object_dist'])]
        final_traffic_lightsdf=final_traffic_lightsdf[['object_dist','object_angle','object_area',
                                                      'attributes.trafficLightColor']]
    except:
        final_traffic_lightsdf = pd.DataFrame({'object_dist' : [np.NaN],'object_angle':[np.NaN],'object_area':[np.NaN],
                                             'traff_lights_color':[np.NaN]})



    try:
        traffic_signsdf = df[df['category'].str.contains('traffic sign')]
        final_traffic_signsdf = traffic_signsdf.loc[traffic_signsdf['object_dist']==min(traffic_signsdf['object_dist'])]
        final_traffic_signsdf=final_traffic_signsdf[['object_dist','object_angle','object_area']]
    except:
        final_traffic_signsdf = pd.DataFrame({'object_dist' : [np.nan],'object_angle':[np.NaN],'object_area':[np.NaN]})


    try:    
        car_df = df[df['category'].str.contains('car')]
        final_car_df = car_df.loc[car_df['object_dist']==min(car_df['object_dist'])]
        final_car_df = final_car_df[['object_dist','object_angle','object_area']]
    except:
        final_car_df = pd.DataFrame({'object_dist' : [np.nan],'object_angle':[np.NaN],'object_area':[np.NaN]})


    try:
        bus_df = df[df['category'].str.contains('bus')]
        final_bus_df = bus_df.loc[bus_df['object_dist']==min(bus_df['object_dist'])]
        final_bus_df = final_bus_df[['object_dist','object_angle','object_area']]
    except:
        final_bus_df = pd.DataFrame({'object_dist' : [np.nan],'object_angle':[np.NaN],'object_area':[np.NaN]})


    try:
        person_df = df[df['category'].str.contains('person')]
        final_person_df = person_df.loc[person_df['object_dist']==min(person_df['object_dist'])]
        final_person_df = final_person_df[['object_dist','object_angle','object_area']]
    except:
        final_person_df = pd.DataFrame({'object_dist' : [np.nan],'object_angle':[np.NaN],'object_area':[np.NaN]})


    try:
        bike_df = df[df['category'].str.contains('bike')]
        final_bike_df = bike_df.loc[bike_df['object_dist']==min(bike_df['object_dist'])]
        final_bike_df = final_bike_df[['object_dist','object_angle','object_area']]
    except:
        final_bike_df = pd.DataFrame({'object_dist' : [np.nan],'object_angle':[np.NaN],'object_area':[np.NaN]})


    try:
        truck_df = df[df['category'].str.contains('truck')]
        final_truck_df = truck_df.loc[truck_df['object_dist']==min(truck_df['object_dist'])]
        final_truck_df = final_truck_df[['object_dist','object_angle','object_area']]
    except:
        final_truck_df = pd.DataFrame({'object_dist' : [np.nan],'object_angle':[np.NaN],'object_area':[np.NaN]})


    try:
        motor_df = df[df['category'].str.contains('motor')]
        final_motor_df = motor_df.loc[motor_df['object_dist']==min(motor_df['object_dist'])]
        final_motor_df = final_motor_df[['object_dist','object_angle','object_area']]
    except:
        final_motor_df = pd.DataFrame({'object_dist' : [np.nan],'object_angle':[np.NaN],'object_area':[np.NaN]})


    try:
        train_df = df[df['category'].str.contains('train')]
        final_train_df = train_df.loc[train_df['object_dist']==min(train_df['object_dist'])]
        final_train_df = final_train_df[['object_dist','object_angle','object_area']]
    except:
        final_train_df = pd.DataFrame({'object_dist' : [np.nan],'object_angle':[np.NaN],'object_area':[np.NaN]})

    try:
        rider_df = df[df['category'].str.contains('rider')]
        final_rider_df = rider_df.loc[rider_df['object_dist']==min(riders_df['object_dist'])]
        final_rider_df = final_rider_df[['object_dist','object_angle','object_area']]
    except:
        final_rider_df = pd.DataFrame({'object_dist' : [np.nan],'object_angle':[np.NaN],'object_area':[np.NaN]})

    firstDataFrame.reset_index(drop=True,inplace=True)

    final_bike_df.reset_index(drop=True,inplace=True)

    final_bus_df.reset_index(drop=True,inplace=True)

    final_left_lanesdf.reset_index(drop=True,inplace=True)

    final_motor_df.reset_index(drop=True,inplace=True)

    final_person_df.reset_index(drop=True,inplace=True)

    final_rider_df.reset_index(drop=True,inplace=True)

    final_car_df.reset_index(drop=True,inplace=True)
    final_right_lanesdf.reset_index(drop=True,inplace=True)
    final_traffic_lightsdf.reset_index(drop=True,inplace=True)
    final_traffic_signsdf.reset_index(drop=True,inplace=True)
    final_train_df.reset_index(drop=True,inplace=True)
    final_truck_df.reset_index(drop=True,inplace=True)

    all_dataframes = [firstDataFrame, final_traffic_lightsdf,final_traffic_signsdf,
                      final_car_df,
                     final_bike_df,final_bus_df,final_motor_df,final_person_df,final_rider_df,
                     final_train_df,final_truck_df,final_left_lanesdf,final_right_lanesdf]

    overall_df = pd.concat(axis=1,objs=all_dataframes)

    overall_df.columns = ['image','scene','time','weather',
                         'traff_lights_dist','traff_lights_angle','traff_lights_area','traff_lights_color',
                        'traff_sign_dist','traff_sign_angle','traff_sign_area',
                        'car_dist','car_angle','car_area',
                         'bike_dist','bike_angle','bike_area',
                         'bus_dist','bus_angle','bus_area',
                         'motor_dist','motor_angle','motor_area',
                         'person_dist','person_angle','person_area',
                         'rider_dist','rider_angle','rider_area',
                         'train_dist','train_angle','train_area',
                         'truck_dist','truck_angle','truck_area',
                         'left_lane_direction','left_lane_style','left_lane_type',
                         'right_lane_direction','right_lane_style','right_lane_type']

    overall_df['direct_drive'] = direct
    overall_df['alternate_drive'] = alternative
    
    # concatenate with the dataframe of zeros
    
    zerodataset = pd.concat(axis=0,objs=[zerodataset,overall_df])
    


# ## Check what you did

# In[ ]:


zerodataset


# In[ ]:


zerodataset = zerodataset.drop(axis=0,labels=[1,2,3])
editdataset = zerodataset
zerodataset.head(4)


# ## Deal with NaN 

# In[ ]:


zerodataset[['traff_lights_area','traff_sign_area','car_area',
             'bike_area','bus_area','motor_area',
            'person_area','rider_area','train_area','truck_area']] = zerodataset[['traff_lights_area',
                                                                                  'traff_sign_area','car_area','bike_area','bus_area','motor_area',
            'person_area','rider_area','train_area','truck_area']].fillna(0)


# ## Categorical encoding
# 
# * Use discrete values to match columns having strings to float or int

# In[ ]:


zerodataset[['traff_lights_angle','traff_sign_angle','car_angle','bike_angle',
            'bus_angle','motor_angle','person_angle','rider_angle',
             'train_angle','truck_angle']]=zerodataset[['traff_lights_angle','traff_sign_angle','car_angle','bike_angle',
            'bus_angle','motor_angle','person_angle','rider_angle','train_angle','truck_angle']].fillna(180)

zerodataset[['traff_lights_dist','traff_sign_dist','car_dist',
             'bike_dist',
            'bus_dist','motor_dist','person_dist','rider_dist',
             'train_dist','truck_dist']]=zerodataset[['traff_lights_dist','traff_sign_dist','car_dist','bike_dist',
            'bus_dist','motor_dist','person_dist','rider_dist','train_dist','truck_dist']].fillna(2000)

zerodataset['alternate_drive']=zerodataset['alternate_drive'].fillna(0)
zerodataset['direct_drive']=zerodataset['direct_drive'].fillna(0)

dir_mapper = {'parallel':0,'vertical':1}
zerodataset['left_lane_direction']=zerodataset['left_lane_direction'].map(dir_mapper)
zerodataset['right_lane_direction']=zerodataset['right_lane_direction'].map(dir_mapper)

style_mapper = {'solid':0,'dashed':1}
zerodataset['left_lane_style']=zerodataset['left_lane_style'].map(style_mapper)
zerodataset['right_lane_style']=zerodataset['right_lane_style'].map(style_mapper)

type_mapper = {'road curb':0,'single white':1,'double white':2,'single yellow':3,'double yellow':4,
              'crosswalk':5,'single other':6,'double other':7}
zerodataset['left_lane_type']=zerodataset['left_lane_type'].map(type_mapper)
zerodataset['right_lane_type']=zerodataset['right_lane_type'].map(type_mapper)

zerodataset[['left_lane_direction','left_lane_style','left_lane_type',
            'right_lane_direction','right_lane_style','right_lane_type']]=zerodataset[['left_lane_direction','left_lane_style','left_lane_type',
            'right_lane_direction','right_lane_style','right_lane_type']].fillna(100)


# In[ ]:


time_mapper = {'daytime':0,'night':1,'dawn/dusk':2}
scene_mapper = {'highway':0,'city street':1,'residential':2,'parking lot':3,
               'gas stations':4,'tunnel':5}
weather_mapper = {'clear':0,'snowy':1,'rainy':2,'overcast':3,'cloudy':4,'partly cloudy':5
                 ,'foggy':6,'undefined':7}
traf_light_color_mapper = {'red':0,'green':1,'yellow':2}

zerodataset['time'] = zerodataset['time'].map(time_mapper)
zerodataset['scene'] = zerodataset['scene'].map(scene_mapper)
zerodataset['weather'] = zerodataset['weather'].map(weather_mapper)


zerodataset['time'] = zerodataset['time'].fillna(100)
zerodataset['scene'] = zerodataset['scene'].fillna(100)
zerodataset['weather'] = zerodataset['weather'].fillna(100)


zerodataset['traff_lights_color'] = zerodataset['traff_lights_color'].map(traf_light_color_mapper)
zerodataset['traff_lights_color'] = zerodataset['traff_lights_color'].fillna(100)


# ## Final dataset

# In[ ]:


zerodataset


# ## Save the dataset

# In[ ]:


zerodataset.to_csv("dataset.csv",index=False)

