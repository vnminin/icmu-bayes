## ICMU Bayes Intro 2025
## Bayesian inference of population proportion
## Author: Volodymyr Minin
## Last updated: September 21, 2025


library(ggplot2)
library(patchwork)

# Prior parameters
prior_alpha = 2.0
prior_beta = 20.0

# Data for two hypothetical samples: 0 positive tests out of 100 and 10 positive tests out of 100
num_positives1 = 0
num_positives2 = 10
num_samples = 100

# Parameters of the Beta posterior distribution 1
posterior_alpha1 = prior_alpha + num_positives1
posterior_beta1 = prior_beta + num_samples - num_positives1 

# Parameters of the Beta posterior distribution 2
posterior_alpha2 = prior_alpha + num_positives2
posterior_beta2 = prior_beta + num_samples - num_positives2 

## create an x-axis grid for plotting
x <- seq(0, 0.3, length.out = 500)

# Data frame for first plot
df1 <- data.frame(
  x = rep(x, 2),
  density = c(dbeta(x, prior_alpha, prior_beta), dbeta(x, posterior_alpha1, posterior_beta1)),
  Distribution = rep(c("Prior", "Posterior"), each = length(x))
)

# Data frame for second plot
df2 <- data.frame(
  x = rep(x, 2),
  density = c(dbeta(x, prior_alpha, prior_beta), dbeta(x, posterior_alpha2, posterior_beta2)),
  Distribution = rep(c("Prior", "Posterior"), each = length(x))
)

# First plot
p1 <- ggplot(df1, aes(x = x, y = density, color = Distribution)) +
  geom_line(size = 1.2) +
  labs(title = "y=0, n=100", x = "Population Proportion", y = "Density") +
  theme_minimal()

# Second plot
p2 <- ggplot(df2, aes(x = x, y = density, color = Distribution)) +
  geom_line(size = 1.2) +
  labs(title = "y=10, n = 100", x = "Population Proportion", y = "Density") +
  theme_minimal()

# Combine plots side by side with one legend
(p1 + p2) + plot_layout(guides = "collect")