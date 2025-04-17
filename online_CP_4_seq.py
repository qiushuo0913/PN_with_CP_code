#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.linalg import solve
import warnings
import time
import scipy.stats as stats
import probnum as pn
from probnum import randvars, linops
from scipy.stats import multivariate_normal
from scipy.stats import chi2
import math
import argparse

# warnings.filterwarnings('ignore')

from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
from probnum import linops, randvars

from probnum.linalg.solvers import (
ProbabilisticLinearSolver,
belief_updates,
beliefs,
information_ops,
policies,
stopping_criteria,
)

def parse_args():
    parser = argparse.ArgumentParser(description='PN_with_CP')
    parser.add_argument('--seed', type=int, default=7, help='the randomness of a simulation')
    parser.add_argument('--p', type=int, default=1, help='probability of requiring labels')
    parser.add_argument('--ifinter_CP', type=int, default=1, help='whether running intermittent CP')
    parser.add_argument('--ratio_list', type=float, nargs='+', 
                    default=[0.1, 0.1, 0.1], 
                    help='List of ratios, e.g., 0.1 0.1 0.1')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    print('Called with args:')
    print(args)
    size_range_left = 500
    size_range_right = 1000
    
    ratio_list = args.ratio_list


    p = args.p
    lr0 = 5e-2
    lr_decay = 0

    alpha = 0.1

    
        

    def volume_elliposid(n, lambda_t, det):
        cn_log = (n/2) * np.log(np.pi) - gammaln(n/2 + 1)
        cn = np.exp(cn_log)
        

        if lambda_t > 1:
            r_square = -2 * np.log(0.99)
        elif lambda_t < 0:
            r_square = -2 * np.log(1e-3)
        else:
            r_square = -2 * np.log(lambda_t)

        
        
        v_log = cn_log + n/2*np.log(r_square) + 1/2*np.log(det)
        
        
        return v_log



    

    # low-complexity (o(kn^2)) calculation of the matrix inverse
    # and the determinate of matrix
    def solution_SM(solver_state, n, iter):
        # Compute B_k in SM
        cov_prior = np.eye(n)
        cov_inverse_prior = np.eye(n)
        
        det = 1 # initial of final determinat
        
        
        det_multiply = 1
        
        action_A = np.vstack(solver_state.actions[0:iter]) @ solver_state.problem.A
        cov_xy = cov_prior @ action_A.T
        gram = action_A @ cov_xy
        for k in range(iter):
            gram_pinv = 1.0 / gram[k,k] if gram[k,k] > 0.0 else 0.0
            action_A_iter = solver_state.actions[k].T @ solver_state.problem.A
            cov_xy_iter = cov_prior @ action_A_iter.T
            gain = cov_xy_iter * gram_pinv
            cov_update = np.outer(gain, cov_xy_iter)
            
            
            Bk = cov_update
            
            # compute the inverse
            gk = 1/(1+np.trace(cov_inverse_prior@Bk))
            detk = 1+np.trace(cov_inverse_prior@Bk)
            det_multiply = det_multiply*detk
            cov_inverse_prior = cov_inverse_prior + gk*cov_inverse_prior@Bk@cov_inverse_prior
            
        return cov_inverse_prior, det*det_multiply


    def online_intermit_conformal_prediction(seed, T, left, right, ratio_list, p, lr0, lr_decay, threshold, ifinter_CP, beta):
        # seed - the seed of the indepedent experiment
        # T -- total time steps
        # left -- the minimal dimension of matrix A
        # right -- the maximal dimension of matrix A
        # ratio_list -- related with the maximal iteration in PN
        # p -- the probability of requiring label
        # lr0 -- initial learning rate
        # lr_decay -- learning rate decay
        # threshold -- the threshold for the set size
        # ifinter_CP -- judge whether the online CP or the online CP with probability p
        # beta -- hyperparamter

        lambda_0 = 0.99
        lambda_t = lambda_0
        coverage_sum = 0
        Coverage = []
        Size = []
        
        Size_list = []
        Dis_list = []
        
        alpha_Upper = []
        alpha_Lower = []
        t_obtain_label = 0
        t_obtain_label_list = []
        pmin = 1
        p_obs_min = 1
        
        
        P = []
        rng = np.random.default_rng(seed)
        for t in range(T):
            
            if t <= 1500:
                ratio = ratio_list[0]
            elif 1500 < t < 3500:
                ratio = ratio_list[1]
            else:
                ratio = ratio_list[2]
            
           
            
            n = rng.integers(low=left, high=right+1)
            A = random_spd_matrix(rng=rng, dim=n)
            b = rng.standard_normal(size=(n,))
            
            linsys = LinearSystem(A=A, b=b)
            maxiter = np.floor(n * ratio).astype(int)
           
            
            # define solution-based solver for obtaining Gaussian distribution of x
            
            pls = ProbabilisticLinearSolver(
            policy=policies.ConjugateGradientPolicy(),
            information_op=information_ops.ProjectedResidualInformationOp(),
            belief_update=belief_updates.solution_based.ProjectedResidualBeliefUpdate(),
            stopping_criterion=(
                stopping_criteria.MaxIterationsStoppingCriterion(maxiter=maxiter)
                # | stopping_criteria.ResidualNormStoppingCriterion(atol=1e-5, rtol=1e-5)
            ),

            )
            
            # define prior for the PN solver
            prior = beliefs.LinearSystemBelief(
            x=randvars.Normal(
                mean=np.zeros((n,)),
                cov=np.eye(n),
            ),
            A=randvars.Normal(
                mean=A, cov=linops.SymmetricKronecker(10 ** -6 * linops.Identity(A.shape[0]))
            ),
            # using diagnoal matrix as the prior of A^{-1} 
            Ainv=randvars.Normal(
                mean=np.eye(n), cov=linops.SymmetricKronecker(10 ** -6 * linops.Identity(A.shape[0]))
            ),
            )
            
            # using PN to obtain the Gaussian distribution of x
            belief, solver_state = pls.solve(prior=prior, problem=linsys)
            
            x_mean = belief.x.mean
            x_cov = belief.x.cov
            
            iter = len(solver_state.actions)-1
            
            # use SM theory to approximate the inverse of x_cov with 
            # complexity O(kn^2)
            x_cov_inv, x_cov_det = solution_SM(solver_state, n, iter)
            
            # calculate the score based on the complementary CDF
            # when has the label with probability p transmitted from cloud
            noise = rng.random(1)
        
            
            
            # judge if the label in the current set
            x_exact = np.linalg.solve(A, b)
            Q_inner = (x_exact-x_mean)@x_cov_inv@(x_exact-x_mean).T
            print('Q_inner', Q_inner)
            if ratio == 0.1:
                scale = 1e-15
            else:
                scale = 2
            Q_value = np.exp(-Q_inner/scale)
            
        
            
            indicator = (Q_value < lambda_t)
            if ifinter_CP: 
                
                v_set = volume_elliposid(n-iter, lambda_t, det=1)
                v_set_intense = (v_set)/(n-iter)
                # print('v_set_intense', v_set_intense)
                if v_set_intense < -9:
                    p = 0.1
                else:
                    p = 1/(1+np.exp(-(v_set_intense - threshold)/beta)) 
                    print('p', p)
                
            if (noise < p):
                t_obtain_label = t_obtain_label + 1
                
                ## find the min value of p
                if p < pmin:
                    pmin = p
                    
                    
                # # update the value of lambda_t
                lr = lr0*np.power(t+1, -lr_decay)
                
                # test the convergence performance with p
                lambda_t = lambda_t + lr/p*(alpha - indicator)
                
                
                lambda_max = 1 + lr/pmin*alpha
                lambda_min = 0+lr/pmin*(alpha-1)
                
                
                assert  lambda_min <= lambda_t <= lambda_max 
                    
            P.append(p)  
            
                
            coverage_sum = coverage_sum+1-indicator  
            
            coverage_t = coverage_sum/(t+1)
            Coverage.append(coverage_t)
            
            
            size = volume_elliposid(n-iter, lambda_t, det=1)
            
            size = size/(n-iter)

           
        
            
            alpha_upper = 1-alpha+(1+lr0/pmin)/(t+1)*1/lr0
            alpha_lower = 1-alpha-(1+lr0/pmin)/(t+1)*1/lr0
            
                
            alpha_Upper.append(alpha_upper)
            alpha_Lower.append(alpha_lower)
            
            Size_list.append(size)
            # Dis_list.append(dis)
            t_obtain_label_list.append(t_obtain_label)
            
            print(f'Finish {((t+1)/T*100):.1f}% , coverage: {coverage_t}, coverage upper: {alpha_upper}, coverage lower: {alpha_lower}, volum: {size}, lambda: {lambda_t}, require_label: {t_obtain_label}', flush=True) 
            
            
        
        coverage = coverage_sum/T
        print('min value of p', pmin)
        
        
        return Coverage, Size_list, Dis_list, P, t_obtain_label_list, alpha_Upper, alpha_Lower
                    
                
    T = 5000
    
    seed = args.seed
    beta = 1
    Coverage, Radius_list, _, p_list, require_label_list, alpha_Upper, alpha_Lower = online_intermit_conformal_prediction(seed = seed, T=T, left=size_range_left, right=size_range_right, ratio_list=ratio_list, p=p, lr0=lr0, lr_decay=lr_decay, threshold=-3.5, ifinter_CP=args.ifinter_CP, beta = beta)


    

