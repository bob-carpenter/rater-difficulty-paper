data {
  int<lower=0> L;  // # predictors
  int<lower=0> N;  // # observations
  array[N] int<lower=0, upper=1> y;  // observations
  matrix[N, L] x;  // predictors
}
model {
  real alpha;
  vector[L] beta;
}
model {
  // priors
  alpha ~ normal(0, 3);
  beta ~ normal(0, 3);
  // likelihood
  y ~ bernoulli_logit(alpha + x * beta);
}
