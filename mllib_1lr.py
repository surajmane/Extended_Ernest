#!/bin/python

import sys
import time
import csv
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext
from sklearn import preprocessing
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

class Mlibry(object):
    
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
    
    def run_lr(self):
        with open('examples_1op.csv', 'w', newline='') as file:
            writer = csv.writer(file)    
            writer.writerow(["#Cores", "Input Fraction", "Time(s)"])
            for i in self.core_fracs:
                self.logregelastic(i[0],i[1])
                writer.writerow([int(i[0]), round(float(i[1]),6), round(float(self.end_t - self.start_t),9)])

    def logregelastic(self, cor, ip):
        spark = SparkSession.\
        builder.\
        appName("LogisticRegressionWithElasticNet").\
        master("spark://spark-master:7077").\
        config("spark.executor.memory", "512m").\
        getOrCreate()
        #print(len(sys.argv))
        #if len(sys.argv) > 2:
        sample_frac = float(ip)
        #print("Input frac is : ",sample_frac)
        num_parts = int(cor)
        #print("Cores are : ",num_parts)
        
        #print(num_parts)

        #sc = SparkContext(appName="LogisticRegressionWithElasticNet")
        sc = spark.sparkContext
        sc.setLogLevel("WARN")
        sqlContext = SQLContext(sc)

        # Load training data
        training = sqlContext.read.format("libsvm").load("/data/rcv1_train.binary/rcv1_train")
        
        training = training.sample(False, sample_frac).coalesce(num_parts)
        lr = LogisticRegression(maxIter=10, elasticNetParam=0.8)
        #training = np.array(training)
        #training = np.fromstring(training, dtype=int, sep=',')
        start = time.time()
        self.start_t = start
        # Fit the model
        #lb = preprocessing.LabelBinarizer()
        #op = lb.fit_transform(training)
        lrModel = lr.fit(training)
        end = time.time()
        self.end_t = end
        print("Cores ",num_parts, "LR sample: ", sample_frac, " took ", (end-start))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage <predictor.py> <csv_file_train>")
        sys.exit(0)
    
    m1 = Mlibry(data_file=sys.argv[1])
    m1.run_lr()
