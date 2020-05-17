for($k=0;$k<1;$k++){
system("perl format1.pl outputs/layer0_vid_train$k\.csv > outputs/train_cnn$k");
system("perl format1.pl outputs/layer0_vid_val$k\.csv > outputs/val_cnn$k");
system("perl format1.pl outputs/layer0_vid_test$k\.csv > outputs/test_cnn$k");
}
