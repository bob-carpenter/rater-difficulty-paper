functions {
  vector sum_to_zero(vector u) {
    return append_row(u, -sum(u));
  }
}
data {
  // item-level data
  int<lower=2> K;  // # categories
  int<lower=0> L;  // # item-level predictors
  int<lower=0> I;  // # items
  array[I] vector[L] x;  // item-level data matrix

  // rating-level data
  int<lower=0> N;  // # ratings
  int<lower=0> J;  // # raters
  array[N] int<lower=1, upper=I> ii;  // item
  array[N] int<lower=1, upper=J> jj;  // rater
  array[N] int<lower=1, upper=K> y;  // rating
}
transformed data {
  vector[K] zerosK = rep_vector(0, K);
}
parameters {
  real alpha;  // item-category intercept
  matrix[K, L] beta;  // item-category slopes

  vector[K - 1] xi_pre;
  array[K] vector[K - 1] psi_pre;
  array[J, K] vector[K - 1] theta_pre;
  array[I] vector[K - 1] phi_pre;
  
  array[K] cholesky_factor_corr[K] L_Omega_theta;
  array[K] vector[K] sigma_theta;

  cholesky_factor_corr[K] L_Omega_phi;
  vector[K] sigma_phi;

  cholesky_factor_corr[K] L_Omega_psi;
  vector[K] sigma_psi;
}
model {
  // sum-to-zero parameters
  vector[K] xi;
  array[K] vector[K] psi;
  array[I] vector[K] phi;
  array[K, J] vector[K] theta;
  xi = sum_to_zero(xi_pre);
  for (k in 1:K)
    psi[k] = sum_to_zero(psi_pre[k]);
  for (i in 1:I)
    phi[i] = sum_to_zero(phi_pre[i]);
  for (k in 1:K)
    for (j in 1:J)
      theta[k, j] = sum_to_zero(theta_pre[k, j]);

  // cov matrix Cholesky factors
  matrix[K, K] L_Sigma_psi;
  matrix[K, K] L_Sigma_phi;
  array[K] matrix[K, K] L_Sigma_theta;
  L_Sigma_psi = diag_post_multiply(L_Omega_psi, sigma_psi);
  L_Sigma_phi = diag_post_multiply(L_Omega_phi, sigma_phi);
  for (k in 1:K)
    L_Sigma_theta[k] = diag_post_multiply(L_Omega_theta[k], sigma_theta[k]);

  // likelihood
  array[I] vector[K] lp;
  // item category likelihood
  for (i in 1:I)
    lp[i] = log_softmax(alpha + beta * x[i]);
  // rating likelihood
  for (n in 1:N) {
    for (k in 1:K) {
      lp[ii[n], k] += categorical_lpmf(y[n] | xi + psi[k] + phi[ii[n]] + theta[jj[n], k]);
    }
  }
  for (i in 1:I) {
    target += log_sum_exp(lp[i]);
  }

  // priors
  alpha ~ normal(0, 5);
  to_vector(beta) ~ normal(0, 3);
  xi ~ normal(0, 3);
  psi ~ multi_normal_cholesky(zerosK, L_Sigma_psi);
  phi ~ multi_normal_cholesky(zerosK, L_Sigma_phi);
  for (k in 1:K)
    theta[k] ~ multi_normal_cholesky(zerosK, L_Sigma_theta[k]);

  // hyperpriors
  sigma_psi ~ normal(0, 3);
  L_Omega_psi ~ lkj_corr_cholesky(5);
  sigma_phi ~ normal(0, 3);
  L_Omega_phi ~ lkj_corr_cholesky(5);
  for (k in 1:K) {
    sigma_theta[k] ~ normal(0, 3);
    L_Omega_theta[k] ~ lkj_corr_cholesky(5);
  }
}
