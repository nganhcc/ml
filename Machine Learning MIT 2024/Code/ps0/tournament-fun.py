'''A tennis tournament starts with sixteen players. Let’s call them hi
, i = 1, 2, . . . 16 (human i, to avoid
the potentially confusing notation pi). The first round has eight games, randomly drawn/paired;
i.e., every player has an equal chance of facing any other player. The eight winners enter the next
round.
As an enthusiastic tennis and data fan, you have an internal model of these 16 players based on
their past performance. In particular, you view each player hi as having a performance index score
si ∼ Gaussian(θi
, σ2
i
). The mean θi roughly captures the player’s ‘intrinsic ability’ and the variance
σ
2
i
roughly captures the player’s performance reliability (accounting for recent injuries etc.). In a
match between hi and hj , player hi wins if si > sj .
Based on your model, what’s the probability that your “top seed player” (the one with the
highest θ) enters the next round? Run 10,000 simulations to check if it agrees with your answer.

Solution: Suppose the top seed player is h1. For h1 to win against an opponent hj , we need
the event sj - s1 < 0. Since si and sj are independent normal distributions, their difference is
also a normal distribution, with mean θj - θ1 and variance σ1^2 + σ2^2 (prove it)
Because h1 has a 1/15 chance of facing any one (and only one) other player, we have 15
such disjoint events. So the total probability of h1 entering the next game is simply:
1/15* sigma(j=2 -> 15)p(sj - s1 < 0)
where (sj - s1) ~ Gaussian(θj - θ1, σ1^2 + σ2^2)
. Solution code below.'''

import numpy as np
import scipy.stats as stats

n=16
theta= np.linspace(1,2, n)  #mean of each gauss
sigma= np.linspace(1,3, n)
topP= np.argmax(theta)
theta_no_topP= np.delete(theta,topP)
sigma_no_topP= np.delete(sigma,topP)
theta_Pj_minus_topP=theta_no_topP-theta[topP]
sigma_Pj_minus_topP=(sigma_no_topP**2)+(sigma[topP]**2)
prob=0
for j in range(theta_no_topP.shape[0]):
    prob+=stats.norm.cdf(0,loc=theta_Pj_minus_topP[j],scale=np.sqrt(sigma_Pj_minus_topP[j]))
prob/=(n-1)
print(prob)
M=int(1e5)
count=0
def one_simulation(theta_no_topP, sigma_no_topP):
    # choose a random opponent index
    j = np.random.choice(range(15))
    sj = stats.norm.rvs(theta_no_topP[j], sigma_no_topP[j])
    topP_s = stats.norm.rvs(theta[topP],sigma[topP] )
    if topP_s < sj:
        return False
    return True
for i in range(M):
    if one_simulation(theta_no_topP, sigma_no_topP):
        count += 1
print(f"Top seed player wins {count/M} of the total simulated games.")
    

