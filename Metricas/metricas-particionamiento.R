acc1 <- c(0.667, 0.583, 0.750, 0.667, 0.833)
acc2 <- c(0.583, 0.417, 0.750, 0.583, 0.750)

f1 <- c(0.625, 0.528, 0.743, 0.657, 0.831)
f2 <- c(0.576, 0.446, 0.743, 0.581, 0.737)

wil_acc <- wilcox.test(x = acc1, y = acc2, alternative = "two.sided", mu = 0, paired = FALSE, conf.int = 0.95, exact = FALSE)
print(wil_acc)

wil_f1 <- wilcox.test(x = f1, y = f2, alternative = "two.sided", mu = 0, paired = FALSE, conf.int = 0.95, exact = FALSE)
print(wil_f1)

print("--------------------")

#remotes::install_github("speegled/wilcoxpower")
library(wilcoxpower)
powerAcc <- power_wilcox_test(sample_size = 5, p1 = acc1, p2 = acc2)
print(powerAcc)

print("--------------------")

powerf1 <- power_wilcox_test(sample_size = 5, p1 = f1, p2 = f2)
print(powerf1)