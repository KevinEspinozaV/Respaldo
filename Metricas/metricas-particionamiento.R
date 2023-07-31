acc1 <- c(0.667, 0.583, 0.750, 0.667, 0.833)
acc2 <- c(0.583, 0.417, 0.750, 0.583, 0.750)

f1 <- c(0.625, 0.528, 0.743, 0.657, 0.831)
f2 <- c(0.576, 0.446, 0.743, 0.581, 0.737)

wil_acc <- wilcox.test(x = acc1, y = acc2, alternative = "two.sided", mu = 0, paired = FALSE, conf.int = 0.95, exact = FALSE)
print(wil_acc)

wil_f1 <- wilcox.test(x = f1, y = f2, alternative = "two.sided", mu = 0, paired = FALSE, conf.int = 0.95, exact = FALSE)
print(wil_f1)

print("--------------------")

# Otra forma para calcular el poder
mean_group1 = mean(acc1)
sd_group1 = sd(acc1)
mean_group2 = mean(acc2)
sd_group2 = sd(acc2)

function_group_1 = function(n) {rnorm(n, mean = mean_group1, sd = sd_group1)}
function_group_2 = function(n) {rnorm(n, mean = mean_group2, sd = sd_group2)}

library(MKpower)
sim.power.wilcox.test(nx = 5,  rx = function_group_1, ny = 5, ry = function_group_2, rx.H0 = NULL, ry.H0 = NULL, 
                      alternative = c("two.sided"), 
                      sig.level = 0.05, conf.int = FALSE, approximate = FALSE,
                      ties = TRUE, iter = 10000, nresample = 10000,
                      parallel = "no", ncpus = 1L, cl = NULL)