acc1 <- c(0.667, 0.583, 0.750, 0.667, 0.833, 0.500, 0.833, 0.500, 0.667, 0.750, 0.667) # accuracy sin filtrado
acc2 <- c(0.833, 0.583, 0.667, 0.667, 0.667) # accuracy con filtrado

f1 <- c(0.625, 0.528, 0.743, 0.657, 0.831, 0.493, 0.837, 0.510, 0.686, 0.742, 0.686) # f1 score sin filtrado
f2 <- c(0.837, 0.489, 0.642, 0.654, 0.692) # f1 score con filtrado

wil_acc <- wilcox.test(x = acc1, y = acc2, alternative = "two.sided", mu = 0, paired = FALSE, conf.int = 0.95, exact = FALSE)
print(wil_acc)

wil_f1 <- wilcox.test(x = f1, y = f2, alternative = "two.sided", mu = 0, paired = FALSE, conf.int = 0.95, exact = FALSE)
print(wil_f1)