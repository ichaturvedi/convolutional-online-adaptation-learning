system("matlab -r format_lib");
system("python2 csv2lib.py train train.lib 0 False");
system("python2 csv2lib.py test test.lib 0 False");
