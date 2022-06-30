import numpy as np

class GaussianDiscriminant:
    def __init__(self,k=2,d=8,priors=None,shared_cov=False):
        self.mean = np.zeros((k,d)) # mean
        self.shared_cov = shared_cov # using class-independent covariance or not
        if self.shared_cov:
            self.S = np.zeros((d,d)) # class-independent covariance
        else:
            self.S = np.zeros((k,d,d)) # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        x1 = np.copy(Xtrain[ytrain==1])
        x2 = np.copy(Xtrain[ytrain==2])

        self.mean[0] = np.average(x1, axis=0)
        self.mean[1] = np.average(x2, axis=0)

        if self.shared_cov:
            # compute the class-independent covariance
            self.S = np.cov(x1, rowvar=False, ddof=0)*self.p[0] + np.cov(x2, rowvar=False, ddof=0)*self.p[1]
        else:
            # compute the class-dependent covariance
            self.S[0] = np.cov(x1, rowvar=False, ddof=0)
            self.S[1] = np.cov(x2, rowvar=False, ddof=0)

    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        for i in np.arange(Xtest.shape[0]): # for each test set example
            c1 = np.zeros((Xtest.shape[0]))
            c2 = np.zeros((Xtest.shape[0]))
            x = Xtest[i]
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):

                mean = self.mean[c]
                s = self.S[c]
                p = self.p[c]

                if self.shared_cov:
                    val = -0.5*np.dot(np.dot((x-mean),np.linalg.inv(self.S)), x-mean) + np.log(p)
                else:
                    val = -0.5*np.log(np.linalg.det(s))-0.5*np.dot(np.dot((x-mean),np.linalg.inv(s)), x-mean) + np.log(p)

                if c == 0:
                    c1[i] = val
                else:
                    c2[i] = val

            # determine the predicted class based on the values of discriminant function
            if c1[i] > c2[i]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        if self.shared_cov:
            return self.mean[0], self.mean[1], self.S
        else:
            return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Diagonal:
    def __init__(self,k=2,d=8,priors=None):
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,)) # variance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        x1 = np.copy(Xtrain[ytrain==1])
        x2 = np.copy(Xtrain[ytrain==2])

        # compute the mean for each class
        self.mean[0] = np.average(x1, axis=0)
        self.mean[1] = np.average(x2, axis=0)

        # compute the variance of different features
        x = Xtrain.T

        for i in range(len(x)):
            self.S[i] = np.var(x[i], ddof=0)

    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        for i in np.arange(Xtest.shape[0]): # for each test set example
            c1 = np.zeros((Xtest.shape[0]))
            c2 = np.zeros((Xtest.shape[0]))
            x = Xtest[i]

            # calculate the value of discriminant function for each class
            d1 = ((x-self.mean[0])/self.S)
            d2 = ((x-self.mean[1])/self.S)
            dis = [np.dot(d1, d2), np.dot(d2, d2)]

            for c in np.arange(self.k):
                val = -1*dis[c]/2+np.log(self.p[c])

                if c == 0:
                    c1[i] = val
                else:
                    c2[i] = val

            # determine the predicted class based on the values of discriminant function
            if c1[i] > c2[i]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2
            pass
        print(predicted_class)
        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
