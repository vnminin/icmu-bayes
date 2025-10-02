// File for the SIR model in Stan
// Data are cases which are assumed to be noisy realizations of
// transitions from S to I in the time (t-1, t]
// Borrowing work from Jon Fintzi
// And also https://mc-stan.org/users/documentation/case-studies/boarding_school_case_study.html#coding_the_model:_stan_program

//first we define the SIR ODE system
functions {
  real [] sir(real t,
              real[] state, //compartments (4 in our case)
              real[] theta, //parameters governing dynamics
              real[] x_r, //fixed quantities which are real numbers (none)
              int[] x_i) { //fixed quantities which are integers (population size)
      real pop_size = x_i[1];
      real S = state[1];
      real I = state[2];
      real R = state[3];
      real C = state[4];

      // changes in S, I, R, C
      real dS_dt = -theta[1] * I * S/pop_size;
      real dI_dt = (theta[1] * I * S/pop_size) - theta[2] * I;
      real dR_dt =  theta[2] * I;
      real dC_dt = theta[1] * I * S/pop_size;

      return {dS_dt, dI_dt, dR_dt, dC_dt};

  }
}
//data and model parameters
data {
  int<lower=0, upper=1> priors_only; //if true, model samples from prior
  int<lower=0> num_obs; //number of observations
  int<lower=0> pop_size; //population size
  int  cases[num_obs]; //vector of observed cases
  real obs_times[num_obs]; //vector of times cases are observed (assumed weekly)
  real R0_mean; //prior mean for log R0
  real R0_sd; //prior sd for log R0
  real nu_mean; //prior mean for log nu
  real nu_sd; //prior sd for log nu
  real phi_mean; //prior mean for log phi
  real phi_sd; //prior sd for log phi
  real rho_mean; //prior mean for log rho
  real rho_sd; //prior sd for log rho
  real frac_S_mean; //prior mean for logit S_SIR
  real frac_S_sd; //prior sd for logit S_SIR
  real frac_I_mean; //prior mean for logit I_IR
  real frac_I_sd; //prior sd for logit I_IR

}
//transform data (needed for ODE solver)
transformed data {
  real x_r[0];
  int x_i[1] = { pop_size };
}
// The parameters of the model
parameters {
  real log_R0; //log R0 (non-centered)
  real logit_frac_S_init; // logit frac S init (non-centered)
  real logit_frac_I_init; // logit frac I init (non-centered)
  real log_nu; // log nu (non-centered)
  real log_rho; // log case detection parameter (non-centered)
  real log_phi; // log case over-dispersion parameter (non-centered)

}

//functions of our parameters
transformed parameters{
  real<lower=0> R0 = exp(log_R0 * R0_sd + R0_mean); //69 - 75 are centered parameters on correct scales
  real<lower=0> nu = exp(log_nu * nu_sd + nu_mean);
  real<lower=0> rho = exp(log_rho * rho_sd + rho_mean);
  real<lower=0> phi = exp(log_phi * phi_sd + phi_mean);
  real S_init = inv_logit(logit_frac_S_init * frac_S_sd + frac_S_mean) * pop_size;
  real I_init = inv_logit(logit_frac_I_init * frac_I_sd + frac_I_mean) * (pop_size - S_init);
  real R_init = pop_size - S_init - I_init;
  real C_init = 1; //C(0) is unimportant, we only care about differences, fix to 1
  real state[num_obs, 4]; //77-79 are needed to solve ODE
  real theta[2];
  real inits[4];
  real incidence[num_obs]; //function of ODE solutions


  inits[1] = S_init;
  inits[2] = I_init;
  inits[3] = R_init;
  inits[4] = C_init;

  theta[1] = R0/nu; // beta
  theta[2] = nu;
  //this is where we solve the ODE using one of stan's built-in ODE solver functions
  state = integrate_ode_rk45(sir,
                             inits,
                              0.0,
                              obs_times,
                              theta,
                              x_r,
                              x_i);
  //calculate incidence as a function of C(t)
  for (t in 1:num_obs) {
    if (t == 1) {
     incidence[t] = state[t,4] - 1; // for the first incidence its the cumalative incidence at time 1 minus the initial condition
    } else {
    incidence[t] = state[t,4] - state[t-1,4];
    }// for future incidence it is the difference between current cumulative incidence and previous
  }

}

// define priors and likelihood
model {
  // non-centered priors
  log_R0 ~ normal(0,1);
  log_nu ~ normal(0,1);
  log_rho ~ normal(0,1);
  log_phi ~ normal(0,1);
  logit_frac_S_init ~ normal(0,1);
  logit_frac_I_init ~ normal(0,1);
  //likelihood
  if (!priors_only) {
    for (t in 1:num_obs) {
        cases[t] ~ neg_binomial_2(incidence[t] * rho, phi);
    }
  }
}

//generating draws from the posterior predictive distribution
generated quantities {
  int gen_cases[num_obs];

  for (i in 1:num_obs) {
    gen_cases[i] = neg_binomial_2_rng(incidence[i] * rho, phi);
  }
}
