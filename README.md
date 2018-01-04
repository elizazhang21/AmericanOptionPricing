# Valuing American Options by Simulation
Final Project - FRE 6831

## 1 Overview
Longstaff(2001)[1] proposed a pricing model for American options by simulation. The model is flexible and applicable in path-dependent and multi-factor situations where traditional finite difference techniques cannot be used. In this project, we implemented this pricing model in Python and briefly evaluated this mode. Also we developed FD and binomial method for American options and the results turn out to be really close and verify the feasibility of these methods.

## 2 Mathematical Details
### 2.1 Valuation Framework
We value American options by simulating enough paths of underlying asset price and compute option price of these paths at each time backwards. At a certain time $t_k$ and outcome $\omega$, we choose greater one of its early-exercise and continuation value as option price:
$$ Y(t_K, \omega) = \max(P(t_K, \omega), F(t_k, \omega))$$

where $Y$ is option price, $P$ and $F$ are early-exercise and continuation value. Continuation value $F(t_k, \omega)$ could be expressed as conditional expectation of future cash flow of option:
$$ F(t_k, \omega) = E[\delta \cdot C(t_k, \omega) | \mathscr{F}_t]$$

where $\delta$ is discount factor, $C$ is future cash flow conditional on $t_k$ and $\omega$. Option price at next time step of our simulated paths, $Y(t_{k+1}, \omega)$, is then one realized sample of all possible future cash flows conditioned on $t_k$.

### 2.2 Compute Continuation Value
In order to estimate continuation value only using out simulated asset price processes, we project continuation value on current asset value $X(t_k, \omega)$ and assume it can be expressed as linear combination of a set of base function dependent only on $X$. We then rewrite(2) as:

$$F(t_k, \omega) = F(X(t_k, \omega)) = \sum^M_{i=0} a_i L_i (X(t_k, \omega))$$ 
where $L_i$ is i-th base function and $a_i$ is corresponding weight. To solve weights $a$ in (3), note that $Y(t_{k+1})$ are realized samples of conditioned future cash flow $C(t_k)$, we could use least mean square algorithm to solve regression problem below:
$$ Y(t_{k+1}) \sim  \sum^M_{i=0} a_i L_i (X(t_k))$$

### 2.3 Algorithm
We now give Python-style pseudo code of valuing American options using our model:

Full Python codes used in experiment could be found in appendix A. Option Valuer.py contains implementation of this pseudo code.


## 3 Experiment Settings
### 3.1 Price Simulation
We first use geometric Brownian motion as in Black-Sheol model to simulate underlying asset price:

### 3.2 Base Functions
As mentioned in original paper, we use first 3 terms of Laguerre polynomials as well as a constant term as our base
functions:

### 3.3 Other Parameters
As in original paper, we set number of early exercise times as 50 per year, i.e. we take 50 points per year with equal
interval from simulated paths. Since early experiment results suggests higher simulation number doesn’t affect valued
price, we set it as 20,000 (10,000 antithetic in case of geometric Brownian motion) to instead of 100,000 as in original
paper, in order to speed up our program.

## 4 Experiment Result
We first applied our model on American put options and compared results with finite difference method.

### 4.1 Valuing American Put Option
Fixing strike price at 40 and risk free rate at 0.06, we computed American put option price when beginning asset price from 34 to 46, volatility is 0.2 or 0.4 and time to maturity is 1 or 2 years, using both our model and finite difference method.

Figure 1 shows valuing result of our simulation model and finite difference method. Difference of 52 option prices(13 prices for each combination of T and sigma) has a mean of 0.0255 and a standard deviation 0.0415.

### 4.2 Valuing American Call Option
We also computed price of American call option. We set strike as 40, risk free rate as 0.06. maturity time as year and year volatility as 0.2. Option prices of beginning asset price from 34 to 46 are then computed.

Figure 2 plots option prices for American call option.

### 4.3 Binomial Method
I also tried the binomial method to value the American options and the results turned out to be very close to what we got with the FD method and the method we see in the paper.

## 5 Conclusion
As we can see in section 4.1 and the figures, our simulation model is able to estimate American options correctly. Its biggest disadvantage, however, is computing speed. It takes minutes compute one price since many rounds of regression is done (50 times per year in our setting) . When we use original paper's setting, 10,000 simulation paths, time cost would be much higher.

However our method is flexible and could be used in many cases without deriving new analytical formulas. It's even applicable when early exercise value is path dependent (i.e. American-Asian option).

We also implemented the binomial method and we could see that the result is perfectly matched, too. So in my opinion these three methods are all practicable in American option pricing. While considering processing speed, we'd better use FD method and Binomial method to reduce running time.

## Reference
[1] Longstaff F A, Schwartz E S. Valuing American options by simulation: a simple least-squares approach[J]. Review
of Financial studies, 2001, 14(1): 113-147.
