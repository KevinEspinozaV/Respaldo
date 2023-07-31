acc1 <- c(0.667, 0.583, 0.750, 0.667, 0.833, 0.500, 0.250, 0.500, 0.667, 0.750, 0.417) # accuracy sin filtrado
acc2 <- c(0.833, 0.583, 0.667, 0.667, 0.667, 0.583, 0.583, 0.667, 0.833, 0.750, 0.917) # accuracy con filtrado

f1 <- c(0.625, 0.528, 0.743, 0.657, 0.831, 0.493, 0.222, 0.510, 0.686, 0.742, 0.394) # f1 score sin filtrado
f2 <- c(0.837, 0.489, 0.642, 0.654, 0.692, 0.506, 0.586, 0.646, 0.837, 0.742, 0.914) # f1 score con filtrado

prec1 <- c(0.657, 0.833, 0.929, 0.875, 0.857, 0.686, 0.611, 0.833, 0.875, 0.833, 0.833) # precision sin filtrado
prec2 <- c(0.750, 0.875, 0.833, 0.833, 0.750, 0.829, 0.833, 0.833, 0.929, 0.833, 0.929) # precision con filtrado

wil_acc <- wilcox.test(x = acc1, y = acc2, alternative = "two.sided", mu = 0, paired = FALSE, conf.int = 0.95, exact = FALSE, correct = FALSE)
print(wil_acc)

wil_f1 <- wilcox.test(x = f1, y = f2, alternative = "two.sided", mu = 0, paired = FALSE, conf.int = 0.95, exact = FALSE)
print(wil_f1)

wil_prec <- wilcox.test(x = prec1, y = prec2, alternative = "two.sided", mu = 0, paired = FALSE, conf.int = 0.95, exact = FALSE)
print(wil_prec)

