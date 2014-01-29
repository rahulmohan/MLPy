import numpy as np
import matplotlib.pyplot as plt

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

				# Calculate output of the network
				# The decision function is given by the line w'x + b = 0;
				output = np.dot(x[i,:],weights) + bias;

				# Perceptron threshold decision: 
				# If w'x[i,:] + b > 0 then the output of x[i,:] is 1
				# If w'x[i,:] + b < 0 then the output of x[i,:] is 0
				if (output > 0):
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
	for i in range(len(x)):
		
		# Calculate w'x + b
		output = np.dot(x[i,:],weights) + bias;
		
		# Get decision from hardlim function
		if (output > 0):
			target = 1;
		else:
			target = 0;

		predictions.append(target);
	
	return predictions

if __name__ == '__main__':

	# Simple AND Gate Test
	# There are infinite solutions to the AND gate test so Perceptron will converge to a different solution every time

	x = np.array( [  [0,0], [0,1], [1,0], [1,1] ] );
	y = np.array( [0,0,0,1] );

	weights,bias = train(x,y,0.02,1000);
	predictions = test(weights,bias,x);

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
	print "Actual Labels: " + str(list(y))
