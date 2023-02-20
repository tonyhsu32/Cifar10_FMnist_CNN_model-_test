## Cifar10 and Fashion Mnist practice ##

1. **Cifar10 test:**  
   * cifar10_cnn_orig.py  
   * cifar10_convert_img.ipynb  
   * cifar10_classifier.ipynb  
   * best_cifar10_weights.h5  
   * test_model.ipynb  
   * tf_tuner.ipynb  
   * tf_tuner.py  
   * cifar-10-batches-py dir  
   * hyper_dir/hyper_record dir
   
   **CNN model test view**  
   *Use **Vgg16** ->* accuarcy is ≈ **0.89**  
   *Use **Colab GPU Telsa T4 16GB***
   ![Vgg16_acc_0.89](https://github.com/tonyhsu32/Cifar10_FMnist_CNN_model_test/blob/main/Vgg16_acc_0.89_fig.png)  
   
   *Use **ResNet9 128 batch** ->* accuracy ≈ **0.9392**  
   ![ResNet9_128_batch_acc_0.9392](https://github.com/tonyhsu32/Cifar10_FMnist_CNN_model_test/blob/main/ResNet9_128_batch_acc_0.9392_fig.png)
   
2. **Fashion Mnist test:**  
   *Use **modified Vgg16** ->* accuarcy is ≈ **0.9404**  
   *Use **Colab GPU Telsa T4 16GB***
   * fMnist_classifier.ipynb  
   * best_fmnist_weights.h5  
 
3. **Tools:**  
   Convert_h5_to_pb_script.ipynb
