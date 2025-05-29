import numpy as np

# Here we use stochastic gradient method to train the model
# We are trying to svm objective function: min (1/2||w||^2) s.t (y*(w.T+b)>=1)
# We use hinge loss to solve this problem
# If a point is correctly classified, the hinge loss is 0, then we do gradient on alpha as normal 
# In any case that a point is misclassified, we will add hingeloss function, then do gradient on alpha as normal
def svm_gradient_method_train(X, Y, radius ): #radius is like variance of gauss distribution
    # After solve first problem, we get formula of w,b, sj we just need to update alpha
    m,n = X.shape
    Y=2 * Y -1    #convert y: from {0,1} to {-1,1}
    X=1.0 *(X>0)  #convert X values from real number to binary {1 if presents, 0 if not}

    # We use Gauss Kernel function K(i,j)= exp[-1/2*(||x_i-x_j||^2)/(radius^2)]
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i*x_j
    squared= np.sum(X * X, axis=1).reshape((-1,1))
    mul= X@ (X.T)
    K= np.exp(-0.5 * (squared + squared -2 *mul)/ (radius**2))

    #Initialization
    alpha= np.zeros(m)
    alpha_avg= np.zeros(m)
    learning_rate= 1. /(64*m)
    outer_loops =10 

    #update alpha
    alpha_avg = 0
    ii = 0
    while ii < outer_loops * m:
        i = int(np.random.rand() * m)
        margin = Y[i] * np.dot(K[i, :], alpha)
        grad = m * learning_rate * K[:, i] * alpha[i]
        if margin < 1:
            grad -= Y[i] * K[:, i]
        alpha -= grad / np.sqrt(ii + 1)
        alpha_avg += alpha
        ii += 1

    alpha_avg /= (ii + 1) * m

    state={}
    state['alpha'] = alpha
    state['alpha_avg'] = alpha_avg
    state['Xtrain'] = X
    state['Sqtrain'] = squared
    return state

def svm_predict(state, matrix, radius):
    M, N = matrix.shape
    Xtrain = state['Xtrain']
    Sqtrain = state['Sqtrain']
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(Xtrain.T)
    K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (radius ** 2)))  # use Sqtrain instead of squared because of relation btw train point with new point(test or valid data)
    alpha_avg = state['alpha_avg']
    preds = K.dot(alpha_avg)
    output = (1 + np.sign(preds)) // 2

    return output


    

