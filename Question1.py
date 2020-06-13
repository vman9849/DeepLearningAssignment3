import numpy as np
import random

# Input values
x = np.array([[0,0],[0,1],[1,0],[1,1]])

# target values
t = np.array([1,1,1,0])

w1 = np.random.uniform(-2.0,2,[2,3])
w2 = np.random.uniform(-2.0,2,[3,4])
w3 = np.random.uniform(-2.0,2,4)

b1 = np.random.uniform(-2.0,2,3)
b2 = np.random.uniform(-2.0,2,4)
b3 = np.random.uniform(-2.0,2)

#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivative of Sigmoid function 
def der_sigmoid(x):
    return x*(1-x)


count = 0
learning_rate = 0.8
output = []

while(count<500000):
  count = count + 1
  print("Count: ",count)
  for i in range(4):
    # print("For loop count: ",i)
    # print(x[i])
    y1 = (np.dot(x[i],w1)) + b1
    # print("Y1: ",y1)
    sigmoid_list1 = sigmoid(y1)
    # print("Sigmoid List1: ",sigmoid_list1)

    y2 = (np.dot(y1,w2)) + b2
    # print("Y2: ",y2)
    sigmoid_list2 = sigmoid(y2)
    # print("Sigmoid List2: ",sigmoid_list2)

    y3 = (np.dot(y2,w3)) + b3
    # print("Y3: ",y3)
    final_sigmoid = sigmoid(y3)
    # print("Final Sigmoid: ",final_sigmoid)

    error = (t[i] - final_sigmoid)**2
    der_error = (t[i] - final_sigmoid)

    dw3 = np.multiply((learning_rate*der_error*der_sigmoid(final_sigmoid)),sigmoid_list2)
    # print(dw1)
    db3 = learning_rate*der_error*der_sigmoid(final_sigmoid)
    # print("Dw3: ",dw3)
    # print("DB3: ",db3)

    w3 = w3 + dw3
    b3 = b3 + db3

    test_sum1 = np.sum(error * der_sigmoid(final_sigmoid))

    dw2 = np.transpose([learning_rate*der_error*test_sum1*der_sigmoid(sigmoid_list2)*layer2 for layer2 in sigmoid_list1])
    # print("Dw2: ",dw2)
    db2 = test_sum1 * der_sigmoid(sigmoid_list2)
    # print("Db2: ",db2)

    # print("Before Transpose:",w2)
    w2 = w2.T
    # print("After Transpose:",w2)
    w2 = w2 + dw2
    w2 = w2.T
    # print("After Transpose of Transpose:",w2)
    b2 = b2 + db2

    test_sum2 = np.sum(test_sum1*der_sigmoid(sigmoid_list2))

    dw1 = np.transpose([learning_rate*der_error*test_sum2*der_sigmoid(sigmoid_list1)*layer1 for layer1 in x[i]])
    # print("Dw1: ",dw1)
    db1 = learning_rate*test_sum2*der_error*der_sigmoid(sigmoid_list1)

    # print("Before Transpose:",w1)
    w1 = w1.T
    # print("After Transpose:",w1)
    w1 = w1 + dw1
    w1 = w1.T
    # print("After Transpose of Transpose:",w1)
    b1 = b1 + db1
    # print("Error: ",error)

    output.insert(i,final_sigmoid)
    # print("\n")
  
  print("Output: ",output)

  error1 = 0
  for m in range(len(output)):
    error1 = error1 + (output[m]-t[m])**2
  print("Error1 = ",error1)

  if(error1 > 0.1):
    output = []
  else:
    print("\nInput: \n",x)
    print("\nWeights1: {} \n\n Weights2: {} \n\n Weights3: {} \n\n Bias1: {} \n\n Bias2: {} \n\n Bias 3: {} \n\n".format(w1,w2, w3,b1,b2,b3))
    print("\nOutput: \n",output)
    break;
  print("\n\n\n")
