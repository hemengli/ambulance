#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools

import copy
import datetime

from scipy.optimize import fsolve
from scipy.special import gamma

import ray
ray.shutdown()
ray.init()


# In[2]:


class City(object):
        
    def __init__(self, width, height, hospital_count, x_count, y_count, base_count, callRateList, muList):
        
        self.width = width
        self.height = height
        
        hospital_x = np.linspace(0, self.width, num = hospital_count + 2)
        self.hospital_location = [(hospital_x[i+1], self.height/2) for i in range(hospital_count)]
        
        self.subregion_width = width / x_count
        self.subregion_height = height / y_count
        self.x_count = x_count
        self.y_count = y_count
        self.subregion_count = x_count * y_count
        
        x = np.linspace(0, self.width, num=x_count+1)
        y = np.linspace(0, self.height, num=y_count+1)
        x_center = (x + np.diff(x)[0]/2)[:-1]
        y_center = (y + np.diff(y)[0]/2)[:-1] 
        self.center_location = list(itertools.product(*[x_center, y_center]))          
        
        self.base_count = base_count
        base_x = np.linspace(0, self.width, num = int(self.base_count/2 + 2))
        base_y = np.linspace(0, self.height, num = 4)
        base_location = [[(base_x[i+1], base_y[j+1]) for j in range(2)] for i in range(int(self.base_count/2))]
        self.base_location = list(itertools.chain(*base_location))
        
        self.callRateList = callRateList
        self.muList = muList
    
    def sample_subregion(self):
        return np.random.choice(np.arange(self.subregion_count), p=self.callRateList)
        


# In[3]:


def weibull_param_relationship(scale, mean, stddev):
    return stddev**2/mean**2 - gamma(1 + 2/scale)/(gamma(1+1/scale)**2) + 1

def get_weibull_parameters(mean = 30, stddev = 13):
    scale = fsolve(weibull_param_relationship, 1,args=(mean, stddev))
    shape = mean / gamma(1+1/scale)
    
    return scale, shape


# In[4]:


class Call(object):
    
    def __init__(self, call_index, city, time_arrival, 
                 call_arrival_mean, call_hospital_mean = 30, call_hospital_stddev = 13, call_service_mean = 12, 
                 priority_rate = 0):
        
        self.call_index = call_index
        self.city = city
        
        self.call_arrival_mean = call_arrival_mean
        self.call_service_mean = call_service_mean
        self.call_hospital_mean = call_hospital_mean
        self.call_hospital_stddev = call_hospital_stddev
        
        scale, shape = get_weibull_parameters(self.call_hospital_mean, self.call_hospital_stddev)
        self.scale = scale
        self.shape = shape
        
        self.priority_rate = priority_rate
        self.priority = self.gen_call_priority()
        
        self.arrival_time = time_arrival
        self.location, self.subregion = self.gen_call_arrival_loc()
        self.next_arrival_time = self.gen_call_arrival_time()
        self.atlocation_servicetime = self.gen_atlocation_service_time()
        self.athospital_servicetime = self.gen_athospital_service_time()
        
        
        #initialize the call to be unassigned
        #-1-unassigned, 0 to (N-1) assigned ambulance index 
        self.status = -1
        
        
    def gen_call_arrival_loc(self):
        subx_loc = np.random.uniform(0, self.city.subregion_width)
        suby_loc = np.random.uniform(0, self.city.subregion_height)

        region_index = city.sample_subregion()

        i,j = divmod(region_index, self.city.y_count)
        x_loc = subx_loc + i * self.city.subregion_width
        y_loc = suby_loc + j * self.city.subregion_height

        location = (x_loc, y_loc)
        subregion = region_index

        return location, subregion


    def gen_call_arrival_time(self):
        return self.arrival_time + np.random.exponential(self.call_arrival_mean)

    def gen_call_priority(self):
        return int(np.random.uniform(0,1) < self.priority_rate)

    def gen_atlocation_service_time(self):
        return np.random.exponential(self.call_service_mean)

    def gen_athospital_service_time(self):
        return (self.scale * np.random.weibull(self.shape))[0]


# In[5]:


def get_distance(location1, location2):
    
    distance = np.abs(location1[0] - location2[0]) + np.abs(location1[1] - location2[1])
    return distance


# In[6]:


def get_ambulance_travel_time(distance, speed = 30):
    return distance/speed


# In[7]:


def get_nearest_hospital(location, city):
    
    hospital_list = city.hospital_location
        
    min_distance = np.inf
    nearest_hospital = hospital_list[0]
    for hospital in hospital_list:
        distance = get_distance(location, hospital)
        if distance < min_distance:
            min_distance = distance
            nearest_hospital = hospital
    
    return nearest_hospital, min_distance


# In[8]:


class Ambulance(object):
    
    def __init__(self, base, speed = 30):
        
        #fix the ambulance's home base
        self.base = base
        
        #set the ambulance's travel speed
        self.speed = speed
        
        #status code: 0-idle at base, 1-going to scene of call, 2-serving at scene of call, 3-going to hospital
        #4-transferring patient at hospital, 5-returning to (home) base
        self.status = 0
        
        #initialize the ambulance idle at home base
        self.origin = base
        self.destination = base # if destination = origin, ambulance is stationary
        
        self.time = 0
        self.endtime = np.inf
        
        self.call = -1 # -1 if not assigned to any call

        
    def update_status(self, status, origin, destination, time, endtime):
        self.status = status
        
        self.origin = origin
        self.destination = destination
        self.time = time
        self.endtime = endtime
    
    def get_ambulance_travel_time(self, distance):
        return distance/self.speed
        
    def redployment(self, base = None):
        
        if base is None:
            base = self.base
        
        #current heuristic: return to home base
        distance = get_distance(self.origin, base)
        travel_time = self.get_ambulance_travel_time(distance)
        self.update_status(5, self.origin, self.base, 
                           copy.deepcopy(self.endtime), copy.deepcopy(self.endtime) + travel_time)
        self.call = -1
        
    def return_to_base(self):
        self.update_status(0, self.base, self.base, copy.deepcopy(self.endtime), np.inf)
        
    def assign_to_call(self, call, time, index):
        
        distance = get_distance(self.origin, call.location)
        travel_time = self.get_ambulance_travel_time(distance)
        self.update_status(1, self.origin, call.location, time, time + travel_time)
        self.call = call # updated assigned call
        
        call.status = index
        
        
    def reach_call_location(self, call, callList):
    
        atlocation_servicetime = call.atlocation_servicetime
        self.update_status(2, call.location, call.location, 
                           copy.deepcopy(self.endtime), copy.deepcopy(self.endtime) + atlocation_servicetime)

        callList.pop(call.call_index)
    
    def transport_to_hospital(self, city):
        
        #transport to nearest hospital
        nearest_hospital, min_distance = get_nearest_hospital(self.origin, city)
        travel_time = self.get_ambulance_travel_time(min_distance)
        self.update_status(3, self.origin, nearest_hospital, 
                           copy.deepcopy(self.endtime), copy.deepcopy(self.endtime) + travel_time)
        
    def reach_hospital(self):
        hospital_servicetime = self.call.athospital_servicetime
        self.update_status(4, copy.deepcopy(self.destination), copy.deepcopy(self.destination), 
                           copy.deepcopy(self.endtime), copy.deepcopy(self.endtime) + hospital_servicetime)


# In[9]:


def get_nearest_available_ambulance(location, ambulanceList):
    
    assign = -1
    index = 0

    nearest_distance = np.inf
    for ambulance in ambulanceList:

        if ambulance.status == 0:
            distance = get_distance(ambulance.origin, location)
            if distance < nearest_distance:
                nearest_distance = distance
                assign = index

        index = index + 1
    
    return assign


# In[10]:


def arrival(call_index, city, ambulanceList, callList, time_arrival, M, timeline, 
            call_arrival_mean = 4, call_service_mean = 12, hospital_mean = 30, hospital_stddev = 13, 
            priority_rate = 0.25):
    
    call = Call(call_index, city, time_arrival, 
                call_arrival_mean, hospital_mean, hospital_stddev, call_service_mean, 
                priority_rate)
    
    i = timeline.shape[0]
    timeline.loc[i] = [call_index, call.priority, call.subregion, '', 0, time_arrival]
    if len(callList) >= M:
        # print("New call arrived. No more capacity. Reject call.")
        i = timeline.shape[0]
        timeline.loc[i] = [call_index, call.priority, call.subregion, '', 7, time_arrival]
    
    else:
    
        # print("New call arrived. Add to call list.")
        callList[call_index] = call

        ambulance_index = get_nearest_available_ambulance(call.location, ambulanceList)

        if ambulance_index > -1:
            # when the call arrives, there are ambulances idle at base, so assign the call to the nearest ambulance
            # print("Idle ambulance at base available. Assign. Now travelling to call location.")
            i = timeline.shape[0]
            timeline.loc[i] = [call_index, call.priority, call.subregion, ambulance_index, 1, time_arrival]
            ambulanceList[ambulance_index].assign_to_call(call, time_arrival, ambulance_index)

        # else:
            # print("No available ambulance at the moment.")
            
    time_arrival = call.next_arrival_time
    
    return ambulanceList, callList, time_arrival, timeline


# # Heuristic Policies

# In[11]:


def get_first(ambulance, callList, M, k):
    
    call_index = -1
    min_arrival_time = np.inf
    for call_id, call in callList.items():
        if call.status == -1:
            call_time = call.arrival_time
            if call_time < min_arrival_time:
                call_index = call_id
                min_arrival_time = call_time
        
    return call_index


# In[12]:


def get_first_highpriority(ambulance, callList, M, k):
    
    # Assign based on FCFS within priority
    # assume only high and low, two levels or priority
    
    high_call_index = -1
    low_call_index = -1
    
    high_min_arrival_time = np.inf
    low_min_arrival_time = np.inf
    for call_id, call in callList.items():
        if call.status == -1:
            call_time = call.arrival_time
            
            if call.priority == 1:
                if call_time < high_min_arrival_time:
                    high_call_index = call_id
                    high_min_arrival_time = call_time
            
            else:
                if call_time < low_min_arrival_time:
                    low_call_index = call_id
                    low_min_arrival_time = call_time
            

    if high_call_index > -1:
        return high_call_index
    else:
        return low_call_index
    


# In[13]:


def get_nearest(ambulance, callList, M, k):
    
    ambulance_loc = ambulance.origin
    
    min_distance = np.inf
    call_index = -1
    
    for call_id, call in callList.items():
        if call.status == -1:
            distance = get_distance(call.location, ambulance_loc)
            if distance < min_distance:
                call_index = call_id
                min_distance = distance
    
    if call_index > -1:
        # there is some unassigned call in queue
        return call_index
    else:
        # all calls in queue have been assigned
        return -1


# In[14]:


def get_nearest_threshold(ambulance, callList, M, k):
    # only serve unassigned calls within k distance from the ambulance
    
    ambulance_loc = ambulance.origin
    
    min_distance = k
    call_index = -1
    
    for call_id, call in callList.items():
        if call.status == -1:
            distance = get_distance(call.location, ambulance_loc)
            if distance <= min_distance:
                call_index = call_id
                min_distance = distance
    
    if call_index > -1:
        # there is some unassigned call in queue
        return call_index
    else:
        # all calls in queue have been assigned
        return -1


# In[15]:


def get_nearest_threshold_else_fcfs(ambulance, callList, M, k):
    # serve unassigned calls within k distance from the ambulance
    # if no unassigned calls within the threshold, perform according to fcfs
    
    ambulance_loc = ambulance.origin
    
    min_distance = k
    call_index = -1
    min_arrival_time = np.inf
    
    for call_id, call in callList.items():
        if call.status == -1:
            distance = get_distance(call.location, ambulance_loc)
            if distance < min_distance:
                call_index = call_id
                min_distance = distance
            elif distance > k:
                if call.arrival_time < min_arrival_time:
                    call_index = call_id
                    min_arrival_time = call.arrival_time
    
    if call_index > -1:
        # there is some unassigned call in queue
        return call_index
    else:
        # all calls in queue have been assigned
        return -1


# In[16]:


def get_fcfs_threshold_else_nearest(ambulance, callList, M, k):
    # serve fcfs within a threshold, else serve nearest
    
    ambulance_loc = ambulance.origin
    
    min_distance = np.inf
    nearest_call_index = -1
    min_arrival_time = np.inf
    fcfs_call_index = -1
    
    for call_id, call in callList.items():
        
        if call.status == -1:
            distance = get_distance(call.location, ambulance_loc)
            
            if distance < k:
                if call.arrival_time < min_arrival_time:
                    fcfs_call_index = call_id
                    min_arrival_time = call.arrival_time
                    
            else:
                
                if distance < min_distance:
                    nearest_call_index = call_id
                    min_distance = distance
    
    if fcfs_call_index > -1:
        return fcfs_call_index
    else:
        return nearest_call_index


# In[17]:


def get_nearest_highpriority(ambulance, callList, M, k):
    
    ambulance_loc = ambulance.origin
    
    min_high_distance = np.inf
    min_low_distance = np.inf
    high_call_index = -1
    low_call_index = -1
    
    for call_id, call in callList.items():
        if call.status == -1:
            distance = get_distance(call.location, ambulance_loc)
            
            if call.priority == 1:
                if distance < min_high_distance:
                    high_call_index = call_id
                    min_high_distance = distance
            else:
                if distance < min_low_distance:
                    low_call_index = call_id
                    min_low_distance = distance
    
    if high_call_index > -1:
        return high_call_index
    elif low_call_index > -1:
        return low_call_index
    else:
        return -1


# In[18]:


def get_next_event(policy, time_arrival, ambulanceList, callList, timeline, 
                   city, M, time_threshold, distance_threshold = 2, 
                   call_arrival_mean = 4, call_service_mean = 12, hospital_mean = 30, hospital_stddev = 13, 
                   priority_rate = 0.25):
    
    ambulanceEndTime_min = np.inf
    index_min = -1
    index = 0
    for ambulance in ambulanceList:
        if ambulance.endtime < ambulanceEndTime_min:
            ambulanceEndTime_min = copy.deepcopy(ambulance.endtime)
            index_min = index
        
        index = index + 1
    
    next_event_time = min(time_arrival, ambulanceEndTime_min)
    
    if next_event_time == time_arrival:
        # print("New call arrived.")
        all_call = set(timeline['Call'])
        call_index = len(all_call) if -1 not in all_call else len(all_call)-1
        ambulanceList, callList, time_arrival, timeline =         arrival(call_index, city, ambulanceList, callList, time_arrival, M, timeline, 
                call_arrival_mean, call_service_mean, hospital_mean, hospital_stddev, 
                priority_rate)
        
    else:
        if ambulanceList[index_min].status == 1:
            # print("Now reach call location. Start at-location treatment. Remove call from call list.")
#             call_index, priority = ambulanceList[index_min].call
            call = ambulanceList[index_min].call
            i = timeline.shape[0]
            timeline.loc[i] = [call.call_index, call.priority, call.subregion, index_min, 2, next_event_time]
            ambulanceList[index_min].reach_call_location(call, callList)
            
        elif ambulanceList[index_min].status == 2:
            # print("Now finish at-location treatment. Start going to hospital.")
            call = ambulanceList[index_min].call
            i = timeline.shape[0]
            timeline.loc[i] = [call.call_index, call.priority, call.subregion, index_min, 3, next_event_time]
            ambulanceList[index_min].transport_to_hospital(city)
            
        elif ambulanceList[index_min].status == 3:
            # print("Now reach hospital. Start transferring patient to hospital.")
            call = ambulanceList[index_min].call
            i = timeline.shape[0]
            timeline.loc[i] = [call.call_index, call.priority, call.subregion, index_min, 4, next_event_time]
            ambulanceList[index_min].reach_hospital()
            
        elif ambulanceList[index_min].status == 4:
            
            # print("Now finish transfering patient to hospital. Decide next step (assign to call or return to base).")
            call = ambulanceList[index_min].call
            i = timeline.shape[0]
            timeline.loc[i] = [call.call_index, call.priority, call.subregion, index_min, 5, next_event_time]
            
            if len(callList) == 0:
                # print("Return to base.")
                ambulanceList[index_min].redployment()
            else:
                # print("Call waiting. Assign to call in queue according to policy.")
                call_index = policy(ambulanceList[index_min], callList, M, 
                                    time_threshold * ambulanceList[index_min].speed/distance_threshold)
                
                if call_index == -1:
                    # calls in callList have all been assigned with an ambulance, or exceed distance threshold
                    ambulanceList[index_min].redployment()
                    
                else:
                    i = timeline.shape[0]
                    call = callList[call_index]
                    timeline.loc[i] = [call_index, call.priority, call.subregion, index_min, 1, next_event_time]
                    ambulanceList[index_min].assign_to_call(call, next_event_time, index_min)
                    
        elif ambulanceList[index_min].status == 5:
            i = timeline.shape[0]
            timeline.loc[i] = [-1, -1, -1, index_min, 6, next_event_time]
            # print("Now reployed ambulance reach base. Start idling.")
            ambulanceList[index_min].return_to_base()
    
    return time_arrival, ambulanceList, callList, timeline, next_event_time


# # Performance Evaluation

# In[19]:


def get_jobcount(timeline):
    timediff = np.append(np.diff(timeline['Timestamp']), 0)
    timeline['timediff'] = timediff
    
    n = timeline.shape[0]
    numCalls = np.zeros(n)
    
    count = 0
    for i in range(n):
        event = timeline.iloc[i]['Event']
        if event == 0:
            count += 1
        elif event == 5 or event == 7: 
            count -= 1
            
        if count <0:
            print("hi")
            
        numCalls[i] = count
        
    numCalls[-1] = numCalls[n-2]
    timeline['numCalls'] = numCalls
    return timeline


# In[20]:


def get_jobs(timeline):
    total = int(max(timeline['Call']))
    
    arrival = np.zeros(total+1)*np.nan
    assign = np.zeros(total+1)*np.nan
    reach = np.zeros(total+1)*np.nan
    onsite = np.zeros(total+1)*np.nan
    transfer = np.zeros(total+1)*np.nan
    finish = np.zeros(total+1)*np.nan
    priority = np.zeros(total+1)
    subregion = np.zeros(total+1)
    
    for i in range(total + 1):
        c = timeline[timeline['Call'] == i]
        
        p = list(set(c['Priority']))[0]
        priority[i] = p

        r = list(set(c['Subregion']))[0]
        subregion[i] = r
        
        n = c.shape[0]
        for index, row in c.iterrows():
            t = row['Timestamp']
            event = row['Event']
            if event == 0:
                arrival[i] = t
            elif event == 1:
                assign[i] = t if n > 1 else np.nan
            elif event == 2:
                reach[i] = t if n > 2 else np.nan
            elif event == 3:
                onsite[i] = t if n > 3 else np.nan
            elif event == 4:
                transfer[i] = t if n > 4 else np.nan
            elif event == 5:
                finish[i] = t if n > 5 else np.nan
            elif event == 7:
                finish[i] = t
#         print(n, arrival[i], assign[i], reach[i], onsite[i], transfer[i], finish[i])
        
    columns = ['priority', 'subregion', 'arrival_time', 'assigned_time', 'reach_patient', 'finish_onsite', 'reach_hospital', 'finish']
    data = list(zip(priority, subregion, arrival, assign, reach, onsite, transfer, finish))
    df = pd.DataFrame(data, columns=columns)
    
    df['waiting_time'] = df['assigned_time'] - df['arrival_time']
    df['total_time'] = df['finish'] - df['arrival_time']
    return df


# In[21]:


def performance_metric(timeline, df, target, c = 4, verbose=True):
    
    result_dict = {}
    
    t = timeline.iloc[-1]['Timestamp']
    P = timeline.groupby('numCalls')['timediff'].sum() / t
    
    expectn = sum(P * P.index)
    try:
        expectw = sum(P[c+1:] * (P.index[c+1:]-c))
    except:
        expectw = sum(P[c+1:] * (P.index[c:]-c))
        
    utilization = (expectn - expectw) / c
    
    result_dict['totalCalls'] = df.shape[0]
    result_dict['utilization'] = utilization
    result_dict['expectNJobs'] = expectn
    result_dict['expectNQueue'] = expectw
    
    if verbose:
        print('Utilization:', utilization)
        print('Expected number of jobs in system:', expectn)
        print('Expected number of jobs in queue:', expectw)
    
    df_complete = df.dropna(axis=0)
    result_dict['expectedWaiting'] = np.mean(df_complete['waiting_time'])
    result_dict['expectedTotal'] = np.mean(df_complete['total_time'])
    result_dict['totalComplete'] = len(df_complete)
    
    if verbose:
        print('Expected time in queue:', np.mean(df_complete['waiting_time']))
        print('Expected time in system:', np.mean(df_complete['total_time']))
        print("Total completed patients: ",  len(df_complete))
    
    assigned = df[df['assigned_time'] > 0]
    count = 0
    for index, row in assigned.iterrows():
        if np.isnan(row['reach_patient']) or row['reach_patient']-row['arrival_time'] > target:
            count += 1
    
    result_dict['totalAssigned'] = len(assigned)
    result_dict['totalUnreachable'] = count
    result_dict['rateUnreachable'] = count / df.shape[0]
    
    reached = df[df['reach_patient'] > 0]
    result_dict['expectReach'] = np.mean(reached['reach_patient'] - reached['arrival_time'])
    
    if verbose:
        print("Total assigned patients: ", len(assigned))
        print("Total unreachable calls:", count)
        print("Portion of patients that is unreachable:", count/df.shape[0])
        print("Expected time to reach patients:", np.mean(reached['reach_patient'] - reached['arrival_time']))
    
    
    
    # Higher Priority
    highp = df[df['priority'] == 1]
    highp_complete = highp.dropna(axis=0)
    highp_assigned = highp[highp['assigned_time'] > 0]
    
    result_dict['totalHigh'] = len(highp)
    result_dict['totalCompleteHigh'] = len(highp_complete)
    result_dict['totalAssignedHigh'] = len(highp_assigned)
    
    if verbose:
        print("Total high priority patients: ",  len(highp))
        print("Total high priority patients completed: ",  len(highp_complete))
        print("Total high priority patients assigned: ",  len(highp_assigned))
    
    count = 0
    for index, row in highp_assigned.iterrows():
        if np.isnan(row['reach_patient']) or row['reach_patient']-row['arrival_time'] > target:
            count += 1
    
    highp_reached = highp[highp['reach_patient'] > 0]
    
    result_dict['expectReachHigh'] = np.mean(highp_reached['reach_patient'] - highp_reached['arrival_time'])
    result_dict['totalUnreachableHigh'] = count
    result_dict['expectWaitingHigh'] = np.mean(highp_complete['waiting_time'])
    result_dict['expectTotalHigh'] = np.mean(highp_complete['total_time'])
    result_dict['rateUnreachableHigh'] = count/len(highp)
    
    if verbose:
        print("Total high priority unreachable calls:", count)
        print("Portion of high priority calls that is unreachable:", count/len(highp))
        print('Expected time in queue (high priority patients):', np.mean(highp_complete['waiting_time']))
        print('Expected time in system (high priority patients):', np.mean(highp_complete['total_time']))
        print("Expected time to reach high priority patients:", result_dict['expectReachHigh'])
    
    # Lower Priority
    lowp = df[df['priority'] == 0]
    lowp_complete = lowp.dropna(axis=0)
    lowp_assigned = lowp[lowp['assigned_time'] > 0]

    result_dict['totalLow'] = len(lowp)
    result_dict['totalCompleteLow'] = len(lowp_complete)
    result_dict['totalAssignedLow'] = len(lowp_assigned)
    
    if verbose:
        print("Total low priority patients: ",  len(lowp))
        print("Total low priority patients completed: ",  len(lowp_complete))
        print("Total low priority patients assigned: ",  len(lowp_assigned))
    
    count = 0
    for index, row in lowp_assigned.iterrows():
        if np.isnan(row['reach_patient']) or row['reach_patient']-row['arrival_time'] > target:
            count += 1
            
    lowp_reached = lowp[lowp['reach_patient'] > 0]
    
    result_dict['expectReachLow'] = np.mean(lowp_reached['reach_patient'] - lowp_reached['arrival_time'])
    result_dict['totalUnreachableLow'] = count
    result_dict['expectWaitingLow'] = np.mean(lowp_complete['waiting_time'])
    result_dict['expectTotalLow'] = np.mean(lowp_complete['total_time'])
    result_dict['rateUnreachableLow'] = count/len(lowp)
    
    if verbose:
        print("Total low priority unreachable calls:", count)
        print("Portion of low priority calls that is unreachable:", count/len(lowp))
        print('Expected time in queue (low priority patients):', np.mean(lowp_complete['waiting_time']))
        print('Expected time in system (low priority patients):', np.mean(lowp_complete['total_time']))
        print("Expected time to reach high priority patients:", result_dict['expectReachLow'])
    
    
    return result_dict


# In[22]:


def subregion_performance(subregion, timeline, df, target, c = 4, verbose=True):
    df_region = df[df['subregion'] == subregion]
    return performance_metric(timeline, df_region, target, c, verbose=verbose)


# # Simulation

# ## City Set-up

# In[23]:


# Key parameters
city_dimension = 200 # assume square
num_hospital = 2

num_x = 4
num_y = 4
num_region = num_x * num_y
num_base = num_region

call_arrival_mean = 4
call_service_mean = 12
hospital_mean = 30 
hospital_stddev = 13
priority_rate = 0.25

ambulance_speed = 18

time_threshold = 9 # delta
horizon = 60*24*14

M = 20 # callList max capacity

callRateList = [1/(num_region)] * num_region
muList = [1/20] * num_region


# ## Heuristic Policies Implementation

# ### Nearest

# In[24]:


# city = City(city_dimension, city_dimension, num_hospital, num_x, num_y, num_base, callRateList, muList)
# ambulanceList = [Ambulance(city.base_location[i], speed = ambulance_speed) for i in range(len(city.base_location))]
# callList = {}
# time_arrival = 0
# timeline = pd.DataFrame(columns = ['Call', 'Priority', 'Subregion', 'Ambulance', 'Event', 'Timestamp'])


# In[25]:


# start = datetime.datetime.now()
# time = 0

# while time < horizon:
    
#     time_arrival, ambulanceList, callList, timeline, time = \
#     get_next_event(get_first, time_arrival, ambulanceList, callList, timeline, 
#                    city, M, time_threshold, 
#                    call_arrival_mean = call_arrival_mean, call_service_mean = call_service_mean, 
#                    hospital_mean = hospital_mean, hospital_stddev = hospital_stddev, 
#                    priority_rate = priority_rate)
    
# end = datetime.datetime.now()
   
# policies are:
# get_first, get_first_highpriority, 
# get_nearest, get_nearest_highpriority, 
# get_fcfs_threshold_else_nearest, get_nearest_threshold_else_fcfs


# In[26]:


# timeline = get_jobcount(timeline)
# df = get_jobs(timeline)
# fcfs_perform = performance_metric(timeline, df, time_threshold, c= num_base)


# # Parallel Implementation

# In[27]:


@ray.remote
def get_simulation(city, horizon, policy):
    
    ambulanceList = [Ambulance(city.base_location[i], speed = ambulance_speed) for i in range(len(city.base_location))]
    callList = {}
    time_arrival = 0
    timeline = pd.DataFrame(columns = ['Call', 'Priority', 'Subregion', 'Ambulance', 'Event', 'Timestamp'])
    
    time = 0

    while time < horizon:

        time_arrival, ambulanceList, callList, timeline, time =         get_next_event(policy, time_arrival, ambulanceList, callList, timeline, 
                       city, M, time_threshold, 
                       call_arrival_mean = call_arrival_mean, call_service_mean = call_service_mean, 
                       hospital_mean = hospital_mean, hospital_stddev = hospital_stddev, 
                       priority_rate = priority_rate)


    timeline = get_jobcount(timeline)
    df = get_jobs(timeline)
    policy_perform = performance_metric(timeline, df, time_threshold, c= city.base_count, verbose = False)
    
    return timeline, df, policy_perform


# In[28]:


MAX_ACTORS = 50


# In[29]:


def run_simulation(city, horizon, policy, simulation):
    
    actors_per_run = simulation // MAX_ACTORS
    remainder = simulation - actors_per_run * MAX_ACTORS
    
    timeline_output = []
    df_output = []
    policy_perform_output = []
    
    simulation_output_vec = []
    for i in range(actors_per_run):
        simulation_output_vec.extend([get_simulation.remote(city, horizon, policy) for _ in range(MAX_ACTORS)])
    
    if remainder > 0:
        simulation_output_vec.extend([get_simulation.remote(city, horizon, policy) for _ in range(remainder)])
        
    for i in range(len(simulation_output_vec)):
        
        simulation_output = ray.get(simulation_output_vec[i])
        timeline_output.append(simulation_output[0])
        df_output.append(simulation_output[1])
        policy_perform_output.append(simulation_output[2])
    
    return timeline_output, df_output, policy_perform_output

#     return simulation_output


# ### FCFS

# In[30]:


start = datetime.datetime.now()

city = City(city_dimension, city_dimension, num_hospital, num_x, num_y, num_base, callRateList, muList)
timeline_output, df_output, policy_perform_output = run_simulation(city, horizon, get_first, 50) 

end = datetime.datetime.now()
time_taken = end - start
print("FCFS Simulation took : ", time_taken.total_seconds(), " seconds")

result_print = pd.DataFrame.from_dict(policy_perform_output)
result_print.to_csv('fcfs_result.csv')

# policies are:
# get_first, get_first_highpriority, 
# get_nearest, get_nearest_highpriority, 
# get_fcfs_threshold_else_nearest, get_nearest_threshold_else_fcfs


# ### Nearest

# In[34]:


start = datetime.datetime.now()

city = City(city_dimension, city_dimension, num_hospital, num_x, num_y, num_base, callRateList, muList)
timeline_output, df_output, policy_perform_output = run_simulation(city, horizon, get_nearest, 50) 

end = datetime.datetime.now()
time_taken = end - start
print("Nearest Simulation took : ", time_taken.total_seconds(), " seconds")

result_print = pd.DataFrame.from_dict(policy_perform_output)
result_print.to_csv('nearest_result.csv')

# policies are:
# get_first, get_first_highpriority, 
# get_nearest, get_nearest_highpriority, 
# get_fcfs_threshold_else_nearest, get_nearest_threshold_else_fcfs


# ### First with Priority

# In[ ]:


start = datetime.datetime.now()

city = City(city_dimension, city_dimension, num_hospital, num_x, num_y, num_base, callRateList, muList)
timeline_output, df_output, policy_perform_output = run_simulation(city, horizon, get_first_highpriority, 50) 

end = datetime.datetime.now()
time_taken = end - start
print("FCFS (wPriority) Simulation took : ", time_taken.total_seconds(), " seconds")

result_print = pd.DataFrame.from_dict(policy_perform_output)
result_print.to_csv('fcfs_priority_result.csv')

# policies are:
# get_first, get_first_highpriority, 
# get_nearest, get_nearest_highpriority, 
# get_fcfs_threshold_else_nearest, get_nearest_threshold_else_fcfs


# ### Nearest with Priority

# In[ ]:


start = datetime.datetime.now()

city = City(city_dimension, city_dimension, num_hospital, num_x, num_y, num_base, callRateList, muList)
timeline_output, df_output, policy_perform_output = run_simulation(city, horizon, get_nearest_highpriority, 50) 

end = datetime.datetime.now()
time_taken = end - start
print("Nearest (wPriority) Simulation took : ", time_taken.total_seconds(), " seconds")

result_print = pd.DataFrame.from_dict(policy_perform_output)
result_print.to_csv('nearest_priority_result.csv')

# policies are:
# get_first, get_first_highpriority, 
# get_nearest, get_nearest_highpriority, 
# get_fcfs_threshold_else_nearest, get_nearest_threshold_else_fcfs

