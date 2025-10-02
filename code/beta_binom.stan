//Beta-Binomial Heirarchical Model
data {
  int<lower=0, upper=1> priors_only; //if true samples from the prior
  int<lower=0> num_obs; //number of observations
  int x[num_obs]; //number of successes
  int n[num_obs]; //number of trials
  real lambda_alpha; //lambda for alpha prior
  real lambda_beta; //lambda for beta prior
}

parameters {
  real<lower=0, upper=1> theta[num_obs];
  real<lower=0> alpha;
  real<lower=0> beta;
}

model {
  //hyper priors
  alpha ~ exponential(lambda_alpha);
  beta ~ exponential(lambda_beta);

  //likelihood and prior theta
  for (i in 1:num_obs){
    theta[i] ~ beta(alpha, beta);

    if (!priors_only) {
      x[i] ~ binomial(n[i], theta[i]);
    }
  }

}

//generating draws from the posterior predictive distribution
generated quantities {
  int gen_x[num_obs];

  for (i in 1:num_obs) {
    if (!priors_only) {
      gen_x[i] = binomial_rng(n[i], theta[i]);
    }
  }
}

