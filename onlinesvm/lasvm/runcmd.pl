system("./la_svm -g 0.005 -c 1 ../format/train.lib");
system("./la_test ../format/test.lib train.lib.model out");
