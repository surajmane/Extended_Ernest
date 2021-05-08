import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import nnls
import csv
import sys

class Predictor(object):

  def __init__(self, training_data_in=[], data_file=None):
    ''' 
        Initiliaze the Predictor with some training data
        The training data should be a list of [mcs, input_fraction, time]
    '''
    self.training_data = []
    self.pr_data = []
    self.a_times = []
    self.m_c = []
    self.p_times = []
    self.training_data.extend(training_data_in)
    if data_file:
      with open(data_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
          if row[0][0] != '#':
            parts = row[0].split(',')
            mc = int(parts[0])
            scale = float(parts[1])
            time = float(parts[2])
            self.training_data.append([mc, scale, time])
            self.m_c.append(mc)
            self.a_times.append(time)

  def add(self, mcs, input_fraction, time):
    self.training_data.append([mcs, input_fraction, time])

  def predict(self, input_fraction, mcs):
    ''' 
        Predict running time for given input fraction, number of machines.
    '''    
    test_features = np.array(self._get_features([input_fraction, mcs]))
    return test_features.dot(self.model[0])
  
  def pred_times(self):
    return self.p_times

  def machines(self):
    return self.m_c
    
  def act_times(self):
    return self.a_times
        

  def predict_all(self, test_data):
    ''' 
        Predict running time for a batch of input sizes, machines.
        Input test_data should be a list where every element is (input_fraction, machines)
    '''    
    test_features = np.array([self._get_features([row[0], row[1]]) for row in test_data])
    return test_features.dot(self.model[0])

  def fit(self):
    print("Fitting a model with ", len(self.training_data), " points")
    labels = np.array([row[2] for row in self.training_data])
    #print(labels)
    data_points = np.array([self._get_features(row) for row in self.training_data])
    #print(data_points)
    self.model = nnls(data_points, labels)
    # TODO: Add a debug logging mode ?
    # print "Residual norm ", self.model[1]
    # print "Model ", self.model[0]
    # Calculate training error
    training_errors = []
    for p in self.training_data:
      predicted = self.predict(p[0], p[1])
      self.p_times.append(predicted)
      training_errors.append(predicted / p[2])
      print(predicted," ",p[0])
    training_errors = [str(np.around(i*100, 2)) + "%" for i in training_errors]
    
    with open('expt_1op.csv','r') as file:
        with open('compare_pr.csv','w',newline='') as write_file:
            reader = csv.reader(file)
            writer = csv.writer(write_file)
            #row = r.next()
            i=0
            for row in reader:
                if row[0][0] == '#':
                    row.append("#Prediction_Ratios_1")
                    writer.writerow(row)
                else:
                    row.append(training_errors[i])
                    writer.writerow(row)
                    i=i+1    
                    
    print("Prediction ratios are", ", ".join(training_errors))
    return self.model[0]

  def num_examples(self):
    return len(self.training_data)

  def _get_features(self, training_point):
    mc = training_point[0]
    scale = training_point[1]
    return [1.0, float(scale) / float(mc), float(mc), np.log(mc)]

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage <predictor.py> <csv_file_train>")
    sys.exit(0)

  pred = Predictor(data_file=sys.argv[1])

  model = pred.fit()
  
  test_data = [[i, 1.0] for i in range(4, 64, 4)]

  predicted_times = pred.predict_all(test_data)
  actual_times = pred.act_times()
  mach = pred.machines()
  p_times = pred.pred_times()
  diff = []
  mse = []
  print("Machines, Predicted Time, Actual Time, MSE for data obtained from Optimal Experiment")
  for i in range(0,len(p_times)):
        mse.append(np.square(np.subtract(round(float(p_times[i]),6),round(float(actual_times[i]),6))))
        print(mach[i]," ",p_times[i]," ",actual_times[i]," ",mse[i])
  print("Machines, Predicted Time for 15 different models")
  for i in range(0, len(test_data)):
    print(test_data[i][0], predicted_times[i])
  plt.hist(mse, bins = 10)
  plt.xlabel('Mean Squared Error')
  plt.ylabel('Frequency')
  plt.show()
  mach.sort()
  mse.sort()
  plt.plot(mach,mse)
  plt.xlabel('Number of machines')
  plt.ylabel('Mean Squared Error')
  plt.show()
  #plt.plot(diff,marker='o',linestyle='')
  plt.plot(test_data, predicted_times)
  plt.xlabel('Number of machines')
  plt.ylabel('predicted times(secs)')
  
