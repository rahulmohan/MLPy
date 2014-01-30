import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.distance import pdist, squareform

def rbfKernel(x,sigma):
	pairwise_dists = squareform(pdist(x, 'euclidean'));
	K = scipy.exp(pairwise_dists / sigma**2);
	return K;

def train(x,y,sigma,learningRate,maxIter):

	K = rbfKernel(x, sigma);
	lenWeights = len(K[1,:]);
	weights = np.random.rand(lenWeights);
	bias = np.random.random();
	t = 1;
	converged = False;

	# Perceptron Algorithm

	while not converged and t < maxIter:
		targets = [];
		for i in range(len(x)):

				# Calculate output of the network
				# Kernel method here allows for non-linear classifier
				output = np.dot(K[i,:],weights) + bias;

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
				weights = weights + ((K[i,:] * learningRate * error));
				bias = bias + (learningRate * error);

				targets.append(target);

				t = t + 1;

		if ( list(y) == list(targets) ) == True:
			# As soon as a solution is found break out of the loop
			converged = True;


	return K,weights,bias

def test(weights, bias, K):

	predictions = [];
	margins = [];

	for i in range(len(x)):
		
		# Calculate w'x + b
		output = np.dot(K[i,:],weights) + bias;
		margins.append(output);
		
		# Get decision from hardlim function
		if (output > 0):
			target = 1;
		else:
			target = 0;

		predictions.append(target);

	return predictions,margins

if __name__ == '__main__':

	# XOR test

	x = np.array( [  [0,0], [0,1], [1,0], [1,1] ] );
	y = np.array( [0,1,1,0] );

	K,weights,bias = train(x,y,1,0.02,1000);
	predictions,margins = test(weights,bias,K);

	# Plot non-linear decision boundary - only works if # of features = 2

	w1Point = (-1*bias)/weights[0];
	w2Point = (-1*bias)/weights[1];
	w3Point = (-1*bias)/weights[2];
	w4Point = (-1*bias)/weights[3];

	addDataPoints = np.array([-0.1,0.1,0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]);

	decisionLine1_Slope = ( 0 - (w1Point) ) / ( (w2Point) - 0 );
	decisionLine2_Slope = ( 0 - (w3Point) ) / ( (w4Point) - 0 );

	equationLine1 = (decisionLine1_Slope * addDataPoints) - (decisionLine1_Slope * w2Point);
	equationLine2 = (decisionLine2_Slope * addDataPoints) - (decisionLine2_Slope * w4Point);

	decisionPlot = plt.subplot(1,1,1);
	decisionPlot.plot(x[:,0],x[:,1],'ro',markersize=10, label="Class 0");
	decisionPlot.plot(addDataPoints, equationLine1, 'bo', label="Decision Boundary Line1");
	decisionPlot.plot(addDataPoints , equationLine2, 'bo', label="Decision Boundary Line2");

	decisionPlot.set_xlim([-2.5, 3.5]);
	decisionPlot.set_ylim([-2.5, 3.5]);
	plt.show()

	print "Predicted Labels: " + str(predictions)
	print "Actual Labels: " + str(list(y))
