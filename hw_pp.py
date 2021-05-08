#!/bin/python

import sys
import time
import csv
import math
from pyspark import SparkContext
from pyspark.sql import SQLContext
from sklearn import preprocessing
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

class NoIterPipe(object):
    
    def __init__(self, data_file=None):
        
        self.core_fracs=[]
        self.collect_data=[]
        self.start_t = 0
        self.end_t= 0
        
        if data_file:
          with open(data_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                if row[0][0] != '#':
                    parts = row[0].split(',')
                    self.cores = int(parts[0])
                    self.ip_frac = float(parts[1])
                    self.core_fracs.append([self.cores, self.ip_frac])
    
    def run_pp(self):
        with open('examples_1op.csv', 'w', newline='') as file:
            writer = csv.writer(file)    
            writer.writerow(["#Cores", "Input Fraction", "Time(s)"])
            for i in self.core_fracs:
                self.noniterpp(i[0],i[1])
                writer.writerow([int(i[0]), round(float(i[1]),6), round(float(self.end_t - self.start_t),9)])

    def variance(data, ddof=0):
        n = len(data)
        mean = sum(data) / n
        return sum((x - (sum(data) / len(data))) ** 2 for x in data) / (len(data))
    
    def stdev(data):
        var = variance(data)
        std_dev = math.sqrt(var)
        return std_dev


    def noniterpp(self, cor, ip):
        sample_frac = float(ip)
        num_parts = int(cor)

        spark = SparkSession.\
        builder.\
        appName("HW3_dataskew").\
        master("spark://spark-master:7077").\
        config("spark.executor.memory", "512m").\
        getOrCreate()
        #print(len(sys.argv))
        #if len(sys.argv) > 2:

        #print("Input frac is : ",sample_frac)

        #print("Cores are : ",num_parts)

        #print(num_parts)

        #sc = SparkContext(appName="LogisticRegressionWithElasticNet")
        sc = spark.sparkContext
        sc.setLogLevel("WARN")
        sqlContext = SQLContext(sc)


        # Load training data
        #training = sqlContext.read.format("libsvm").load("/data/rcv1_train")
        #training = training.sample(False, sample_frac).coalesce(num_parts)
        #training.cache().count()
        tickets_flights = sc.textFile("/data/ticket_flights.csv").map(lambda x: x.split(","))
        flights = sc.textFile("/data/flights.csv").map(lambda x: x.split(","))
        aircrafts = sc.textFile("/data/aircrafts_data.csv").map(lambda x: x.split(","))
        
        start = time.time()
        self.start_t = start
        tickets_aircrafts = flights.map(lambda x : (x[7] , x)).sample(False, sample_frac).coalesce(num_parts)
        joined = tickets_aircrafts.join(aircrafts.map(lambda x: (x[0], x)).sample(False, sample_frac).coalesce(num_parts))
        flights_aircrafts = joined.map(lambda x: (x[1][0][0] , x[1]))
        flights_aircrafts_tickets = flights_aircrafts.join(tickets_flights.map(lambda x: (x[1], x)).sample(False,          sample_frac).coalesce(num_parts)).\
        map(lambda x : (x[1][0][0] , x[1][0][1], x[1][1]))
        
        result1 = flights_aircrafts_tickets.map(lambda x: (x[1][1], [int(x[2][3])]))
        final = result1.reduceByKey(lambda x,y: x+y)
        final1 = final.map(lambda x: (x[0], math.sqrt(sum((r - (sum(x[1]) / len(x[1]))) ** 2 for r in x[1]) / (len(x[1])))))
        final1.collect()

        end = time.time()
        self.end_t = end
        print("Cores ",num_parts, "HW3_pipeline sample: ", sample_frac, " took ", (end-start))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage <predictor.py> <csv_file_train>")
        sys.exit(0)
    
    m1 = NoIterPipe(data_file=sys.argv[1])
    m1.run_pp()
