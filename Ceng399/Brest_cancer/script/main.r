diag1 <- read.csv("data/cbis-ddsm-r-dataset.csv")

colSums(is.na(diag1) | diag1 == "")[colSums(is.na(diag1) | diag1 == "") > 0]

