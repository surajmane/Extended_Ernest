import pandas as pd
import numpy as np
from scipy.stats import hmean
def skewness():
  columns = ['column name', 'value']
  aircrafts_data = pd.read_csv("F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/aircrafts_data.csv")
  airports_data = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/airports_data.csv')
  boarding_passes = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/boarding_passes.csv')
  bookings = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/bookings.csv')
  flights = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/flights.csv')
  seats = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/seats.csv')
  ticket_flights = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/ticket_flights.csv')
  tickets = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/tickets.csv')
  aircrafts_data_skew = aircrafts_data.skew().abs().sum()
  
  #aircrafts_data_s = aircrafts_data_skew.sum()
  boarding_passes_skew = boarding_passes.skew().abs().sum()
  
  bookings_skew = bookings.skew().abs().sum()
  flights_skew = flights.skew().abs().sum()
  seats_skew = seats.skew().abs().sum()
  ticket_flights_skew = ticket_flights.skew().abs().sum()
  tickets_skew = tickets.skew().abs().sum()
  airports_data_skew = airports_data.skew().abs().sum()
  data = np.array([aircrafts_data_skew, airports_data_skew, boarding_passes_skew, bookings_skew, flights_skew, seats_skew, ticket_flights_skew, tickets_skew])
  #print(data)
  mean = data.mean()
  h_mean = hmean(data)
  print('The average skew in the dataset is: ', np.round(mean, 2))
  return mean
  
  #return harmonic_mean
skewness()
