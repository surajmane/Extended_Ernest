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
    data_points = np.array([self._get_features(row) for row in self.training_data])
    self.model = nnls(data_points, labels)
    # TODO: Add a debug logging mode ?
    # print "Residual norm ", self.model[1]
    # print "Model ", self.model[0]
    # Calculate training error
    training_errors = []
    for p in self.training_data:
        predicted = self.predict(p[0], p[1])
        self.p_times.append(predicted)
        self.pr_data.append([p[1],str(np.around((predicted/p[2])*100, 2))] )
        training_errors.append(predicted / p[2])
        print(predicted," ",p[0])
    training_errors = [str(np.around(i*100, 2)) + "%" for i in training_errors]
    
    with open('compare_1pr.csv','r') as file:
        with open('compare2_pr.csv','w',newline='') as write_file:
            reader = csv.reader(file)
            writer = csv.writer(write_file)
            #row = r.next()
            i=0
            temp = self.pr_data
            for row in reader:
                parts = row[3].split(",")
                #print("parts: ",parts)
                if row[0][0] == '#':
                    row.append("Projection_ratios_skew")
                    writer.writerow(row)
                
                else:
                    for p in temp:
                        #print(p[0],"   ",p[1])
                        if(float(parts[0]) == float(p[0])):
                            #print(p[0],"   ",p[1])
                            row.append(p[1])
                            writer.writerow(row)
                            temp.remove(p)
                            break   
                    
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
        ame = np.square(np.subtract(int(p_times[i]),int(actual_times[i])))
        mse.append(np.square(np.subtract(round(float(p_times[i]),6),round(float(actual_times[i]),6))))
        diff.append(round(float(p_times[i]),6) - round(float(actual_times[i]),6))
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
  plt.show()
  pred_ratio_1 = []
  pred_ratio_2 = []
  with open('compare2_pr.csv','r',newline='') as read_file:
    reader = csv.reader(read_file, delimiter=' ')
    for row in reader:
      if row[0][0] != '#':
        parts = row[0].split(',')
        mc = int(parts[0])
        pred_ratio_1.append(float(parts[2].split('%')[0]))
        pred_ratio_2.append(float(parts[4].split('%')[0]))
  #print(pred_ratio_1)
  bins = np.linspace(90, 110, 50)
  plt.hist(pred_ratio_1, bins, label='Without considering skew')
  plt.hist(pred_ratio_2, bins, label='Considering skew')
  plt.legend(loc='upper right')
  plt.xlabel('Prediction Ratios')
  plt.show()

    
