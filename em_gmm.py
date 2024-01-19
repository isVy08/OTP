from utils_io import load_pickle, write_pickle
import numpy as np
import sys
from scipy.stats import multivariate_normal
from scipy import special


no = sys.argv[1]
root = sys.argv[2]
data_path = f'data/gmm/data{no}.pkl'
X, true_pi, true_mu, labels = load_pickle(data_path)

labels = labels.cpu().numpy()
true_mu = true_mu.numpy().round(4)
true_pi = true_pi.numpy().round(4)
true_mu = np.sort(true_mu)
true_pi = np.sort(true_pi)
X = X.cpu().numpy()

class GMM:

    def __init__(self,X,number_of_sources,iterations):
        self.iterations = iterations
        self.number_of_sources = number_of_sources
        self.X = X
        self.mu = None
        self.pi = None
        self.cov = None
        self.XY = None

        self.mus = []
        self.pis = []
        
    

    """Define a function which runs for iterations, iterations"""
    def run(self):
        self.reg_cov = 1e-6*np.identity(len(self.X[0]))
        x,y = np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
        self.XY = np.array([x.flatten(),y.flatten()]).T
           
                    
        """ 1. Set the initial mu, covariance and pi values"""
        self.mu = np.random.randint(min(self.X[:,0]),max(self.X[:,0]),size=(self.number_of_sources,len(self.X[0]))) # This is a nxm matrix since we assume n sources (n Gaussians) where each has m dimensions
        self.cov = np.zeros((self.number_of_sources,len(X[0]),len(X[0]))) # We need a nxmxm covariance matrix for each source since we have m features --> We create symmetric covariance matrices with ones on the digonal
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim],1)

        self.pi = np.ones(self.number_of_sources)/self.number_of_sources # Are "Fractions"
        
        if root in ('pgmm', 'vpgmm'):
            # Misspecified settings
            probs = np.random.uniform(0,1) # true_pi[1] + 0.2 
            if probs > 0.5:
                self.pi = np.array([1-probs, probs])
            else:
                self.pi = np.array([probs, 1-probs])
        elif root in ('vgmm', 'vpgmm'):
            for dim in range(len(self.cov)):
                eps = np.random.uniform(0,2)
                self.cov[dim] = self.cov[dim] * eps
        
        self.log_likelihoods = [] # In this list we store the log likehoods per iteration and plot them in the end to check if
                             # if we have converged
            
        """Plot the initial state"""    
        
        for i in range(self.iterations):    
            print(f'Running {root} model ...')           

            """E Step"""
            r_ic = np.zeros((len(self.X),len(self.cov)))

            for m,co,p,r in zip(self.mu,self.cov,self.pi,range(len(r_ic[0]))):
                co+=self.reg_cov
                mn = multivariate_normal(mean=m,cov=co)
            
                r_ic[:,r] = p*mn.pdf(self.X)/np.sum([pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(X) for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.cov+self.reg_cov)],axis=0)
                    
            """
            The above calculation of r_ic is not that obvious why I want to quickly derive what we have done above.
            First of all the nominator:
            We calculate for each source c which is defined by m,co and p for every instance x_i, the multivariate_normal.pdf() value.
            For each loop this gives us a 100x1 matrix (This value divided by the denominator is then assigned to r_ic[:,r] which is in 
            the end a 100x3 matrix).
            Second the denominator:
            What we do here is, we calculate the multivariate_normal.pdf() for every instance x_i for every source c which is defined by
            pi_c, mu_c, and cov_c and write this into a list. This gives us a 3x100 matrix where we have 100 entrances per source c.
            Now the formula wants us to add up the pdf() values given by the 3 sources for each x_i. Hence we sum up this list over axis=0.
            This gives us then a list with 100 entries.
            What we have now is FOR EACH LOOP a list with 100 entries in the nominator and a list with 100 entries in the denominator
            where each element is the pdf per class c for each instance x_i (nominator) respectively the summed pdf's of classes c for each 
            instance x_i. Consequently we can now divide the nominator by the denominator and have as result a list with 100 elements which we
            can then assign to r_ic[:,r] --> One row r per source c. In the end after we have done this for all three sources (three loops)
            and run from r==0 to r==2 we get a matrix with dimensionallity 100x3 which is exactly what we want.
            If we check the entries of r_ic we see that there mostly one element which is much larger than the other two. This is because
            every instance x_i is much closer to one of the three gaussians (that is, much more likely to come from this gaussian) than
            it is to the other two. That is practically speaing, r_ic gives us the fraction of the probability that x_i belongs to class
            c over the probability that x_i belonges to any of the classes c (Probability that x_i occurs given the 3 Gaussians).
            """

            """M Step"""

            # Calculate the new mean vector and new covariance matrices, based on the probable membership of the single x_i to classes c --> r_ic
            self.mu = []
            # self.cov = []
            if root not in ('pgmm', 'vpgmm'):
                self.pi = []
            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:,c],axis=0)
                mu_c = (1/m_c)*np.sum(self.X*r_ic[:,c].reshape(len(self.X),1),axis=0)
                self.mu.append(mu_c)

                # Calculate the covariance matrix per source based on the new mean
                # self.cov.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.X),1)*(self.X-mu_c)).T,(self.X-mu_c)))+self.reg_cov)
                # Calculate pi_new which is the "fraction of points" respectively the fraction of the probability assigned to each source 
                
                if root not in ('pgmm', 'vpgmm'):
                    self.pi.append(m_c/np.sum(r_ic)) 

                # Here np.sum(r_ic) gives as result the number of instances. This is logical since we know 
                # that the columns of each row of r_ic adds up to 1. Since we add up all elements, we sum up all
                # columns per row which gives 1 and then all rows which gives then the number of instances (rows) 
                # in X --> Since pi_new contains the fractions of datapoints, assigned to the sources c,
                # The elements in pi_new must add up to 1

            
            
            """Log likelihood"""
            self.log_likelihoods.append(np.log(np.sum([k*multivariate_normal(self.mu[i],self.cov[j]).pdf(X) for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])))
            self.mus.append([np.sort(self.mu[0]), np.sort(self.mu[1])])
            self.pis.append(np.sort(self.pi))
            print('Iteration:', i)
            print('- True mu1:', true_mu[0, :])
            print('  Est. mu1:', self.mu[0])
            print('- True mu2:', true_mu[1, :])
            print('  Est. mu2:', self.mu[1])
            print('- True pi :', true_pi)
            print('  Est. pi :', self.pi)
            print('=====================================')
                    

            """
            This process of E step followed by a M step is now iterated a number of n times. In the second step for instance,
            we use the calculated pi_new, mu_new and cov_new to calculate the new r_ic which are then used in the second M step
            to calculat the mu_new2 and cov_new2 and so on....
            """
    
    """Predict the membership of an unseen, new datapoint"""
    def predict(self,Y):
        prediction = []        
        for m,c in zip(self.mu,self.cov):  
            #print(c)
            prediction.append(multivariate_normal(mean=m,cov=c).pdf(Y)/np.sum([multivariate_normal(mean=mean,cov=cov).pdf(Y) for mean,cov in zip(self.mu,self.cov)]))
        #plt.show()
        return prediction
         
    
    
GMM = GMM(X, 2, 1000)     
GMM.run()
output = (GMM.mus, GMM.pis)
write_pickle(output, f'{root}/em{no}.pkl')

# correct = 0
# for i in range(X.shape[0]):
#     pred = GMM.predict(X[i, :])
#     pred = np.argmax(pred)
#     if pred == int(labels[i]): 
#         correct += 1 
# print('Accuracy:', correct / X.shape[0])