import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1 + np.exp(-z));

def train(x,y,learningRate,maxIter):

	lenWeights = len(x[1,:]);
	weights = np.random.rand(lenWeights);
	bias = np.random.random();
	t = 1;
	converged = False;

	# Perceptron Algorithm

	while not converged and t < maxIter:
		targets = [];
		for i in range(len(x)):

				# Calculate logistic function (sigmoid function)
				# The decision function is given by the line w'x + b = 0;

				z = np.dot(x[i,:],weights) + bias;
				logistic = sigmoid(z);

				# Logistic regression probability estimate

				if (logistic > 0.5):
					target = 1;
				else:
					target = 0;

				# Calculate error and update weights
				# Shifts the decision boundary

				error = y[i] - target;
				weights = weights + (x[i,:] * (learningRate * error));
				bias = bias + (learningRate * error);

				targets.append(target);

				t = t + 1;

		if ( list(y) == list(targets) ) == True:
			# As soon as a solution is found break out of the loop
			converged = True;


	return weights,bias

def test(weights, bias, x):

	predictions = [];
	margins = [];
	probabilties = [];

	for i in range(len(x)):
		
		# Calculate w'x + b and sigmoid of output
		z = np.dot(x[i,:],weights) + bias;
		logistic = sigmoid(z);
		
		# Get decision from hardlim function
		if (logistic > 0.5):
			target = 1;
		else:
			target = 0;

		predictions.append(target);
		probabilties.append(logit)

	return predictions,probabilties

if __name__ == '__main__':

	# Simple AND Gate Test
	# There are infinite solutions to the AND gate test so gradient descent will converge to a different solution every time
	# You can replace x and y with your own data

	x = np.array( [  [0,0], [0,1], [1,0], [1,1] ] );
	y = np.array( [0,0,0,1] );

	weights,bias = train(x,y,0.02,1000);
	predictions,probabilties = test(weights,bias,x);

	# Plot decision boundary
	# Only can plot if # of features = 2
	# For higher dimensional data use a contour plot
	# Get the two data points to connect to form a line through the equation: -b/w[i]

	decisionPlot = plt.subplot(1,1,1);
	decisionPlot.plot(x[0:-1,0],x[0:-1,1],'ro',markersize=10, label="Class 0");
	decisionPlot.plot(x[-1,0],x[-1,1],'bo',markersize=10, label="Class 1");
	decisionPlot.plot( np.array( [0,(-1*bias)/weights[0]] ) , np.array( [ (-1*bias)/weights[1], 0] ), '--r', label="Decision Boundary Line");

	legend = decisionPlot.legend(loc='upper right', shadow=True, fancybox=True)

	decisionPlot.set_xlim([-0.5, 1.5]);
	decisionPlot.set_ylim([-0.5, 1.5]);
	plt.show()

	print "Predicted Labels: " + str(predictions)
	print "Probability of (event|x[i,:]): " + str(probabilties);
	print "Actual Labels: " + str(list(y))
