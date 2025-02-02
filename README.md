# Prostate Cancer Detection using Deep Learning

# Abstract
Prostate cancer is the major cause of death in men, and its complicated biology and rising incidence
provide considerable diagnostic problems. The study used the PANDA dataset, which included
high-resolution prostate biopsy images, to assess the effectiveness of machine learning and deep
learning methods for prostate cancer grade classification. Baselines were developed using traditional
machine learning models such as Random Forest, K-Nearest Neighbors (KNN), Decision Tree, and
Naive Bayes. Random Forest outperformed the others, with an F1 score of 39% and an accuracy of
41%. However, the overall efficacy of these approaches was limited due to their inability to capture
the detailed patterns shown in histopathology images.

In contrast, deep learning models outperformed machine learning methods. Five pre-trained
models—DenseNet121, VGG16, EfficientNetB7, NASNetMobile, and MobileNet—were tested
with a new attention-based model, VarMIL.

Despite achieving 100% validation accuracy, VGG16's comparatively poor recall (77.48%) and
precision (81.59%) indicated overfitting. With a validation accuracy of 98.54%, precision of 86.48%,
recall of 78.52%, and F1 score of 82.19%, ENet7 demonstrated the most balanced performance.
With an F1 score of 81.73% and a validation accuracy of 97.94%, DenseNet121 performed similarly.
MobileNet's recall was lower (69.29%) than its precision (91.40%), suggesting that it missed some
positive cases. With a validation accuracy of only 25%, the VarMIL model performed noticeably
worse than expected, highlighting the need for architectural changes.

These results highlight the potential of deep learning in medical imaging and stress the value of
utilizing cutting-edge architectures to increase clinical reliability and diagnostic accuracy in the
identification of prostate cancer.

Keywords: Artificial Intelligence; Machine Learning, Deep learning; Convolutional Neural
Networks; Transfer Learning.


