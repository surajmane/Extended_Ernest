import pandas as pd
def skewness():
	aircrafts_data = pd.read_csv("F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/aircrafts_data.csv")
	airports_data = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/airports_data.csv')
	boarding_passes = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/boarding_passes.csv')
	bookings = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/bookings.csv')
	flights = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/flights.csv')
	seats = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/seats.csv')
	ticket_flights = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/ticket_flights.csv')
	tickets = pd.read_csv('F:/Academics/Sem 1/Big Data/Homewok/HM2/cs5614-hw-master/data/tickets.csv')
	
	aircrafts_data_skew = aircrafts_data.skew()
	print(aircrafts_data_skew)
skewness()
