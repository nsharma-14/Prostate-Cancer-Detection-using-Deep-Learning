# Prostate Cancer Detection using Deep Learning
* Created By
Natasha Sharma
University at Albany, NY, USA
E-mail: nsharma@albany.edu

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

1. Introduction
The second most significant cause of cancer-related death in men and the most frequently diagnosed
non-skin cancer is prostate cancer. (Van Booven et al., 2021)

The prostate, a walnut-sized male reproductive gland, secretes alkaline fluid. Positioned in the pelvis,
it is flanked by the rectum and the bladder and is surrounded by parts of the urethra. Research
suggests that a higher count of glandular tissue is linked to a higher chance of developing prostate
cancer. (Iqbal et al., 2021)

In addition, it is projected that 1 in 6 American men may experience this disease at some point in
their lives. Increased prostate biopsies and a lack of urological pathologists, which hinder prostate
cancer diagnosis, are two difficulties encountered throughout prostate cancer treatment. (Van
Booven et al., 2021)

In 2019, there were 191,930 new instances of prostate cancer in the US, with men making up 21%
of the cases. Prostate cancer accounted for the majority of the 893,660 cases of cancer. According to
estimates, 321,160 Americans died from cancer in 2020, with 33,310 of those deaths being
attributable to prostate cancer. (Iqbal et al., 2021)

It is estimated that by 2030, the number of new cases expected yearly could reach 25 million.
(Alshareef et al., 2022)

There are two methods to detect prostate cancer (PCa) early on: PSA testing and digital rectal
examination (DRE). The European Association of Urology's guidelines urge a transrectal ultrasound
(TRUS) systematic biopsy when these tests suggest PCa. However, the sensitivity and specificity of
PSA and DRE are very low. In addition, TRUS is intrusive, comparatively insensitive, and
underestimates aggression. Compared to traditional TRUS, multi-parametric magnetic resonance
imaging (mp-MRI) has demonstrated improved sensitivity, a potential 25% reduction in needless
biopsies, and a decrease in overdiagnosis. (Vente et al., 2021)

Recently, there has been much interest in the new fields of artificial intelligence (AI) and machine
learning in healthcare. Deep learning, a type of ML, uses a layered structure to create complex
modules that can understand complicated inputs. Deep Learning algorithms can now demonstrate
traditional machine learning approaches over various domains, including image classification,
computer vision, speech recognition, and more. (Alshareef et al., 2022)

AI often uses artificial neural networks (ANN), which use statistical models directly inspired by and
partially modeled on biological neural networks. These networks can parallelly model and process
nonlinear interactions between inputs and outputs. (Van Booven et al., 2021)

Computer-assisted diagnosis (CAD) analysis using digitized images has demonstrated the excellent
efficiency of deep convolutional neural networks (DCNNs), a modified version of AAN. DCNNs
made it possible for computerized images of PCa to extract imaging features automatically, and they
are currently utilized to identify PCa and benign tissues using magnetic resonance imaging (MRI).
(Tătaru et al., 2021)

2. Background/Literature Review
   
2.1 About Prostate Cancer
   
Prostate cancer (PCA) is the second most common type of cancer in males globally. It ranks as the
third most common cause of cancer-related mortality in both Europe and the US. Prostate cancer
and its treatment are growing to be major public health challenges as the disease's incidence rises
with patient age. (Michaely et al., 2022)

It was estimated that there were 1.3 million new cases of PCa and 359,000 associated deaths
worldwide in 2018. (Vente et al., 2021)

The rapid increase in the growth of prostate cancer in China was reported in a study conducted in
2015. There is an estimate of (1.7) million diagnosed cases up till 2030. (Abbasi et al., 2020)

(Patel et al., 2021) The scaling of prostate cancer can be called as the ordinal classification issue.

In an early stage, Prostate Specific Antigen (PSA) measurement and Digital Rectal Examination
(DRE) were used to detect PCa. When these tests indicate the possibility of PCa, a transrectal
ultrasound (TRUS) systematic biopsy is recommended by the European Association of Urology
guidelines. PSA and DRE, however, have a relatively low sensitivity and specificity. Furthermore,
TRUS is invasive, has a relatively low sensitivity, and underestimates aggressiveness. (Vente et al.,
2021)

Prostate cancer diagnosis is significantly aided by medical imaging. For a long time, systematic
biopsy was guided solely by transrectal ultrasonography (TRUS). Recent research has demonstrated
that prostate cancer detection can be significantly enhanced by magnetic resonance imaging (MRI).
Radiologists are using MRI-ultrasound fusion biopsies more frequently to target lesions identified on
MRIs. MRI ultrasound fusion biopsies, as opposed to ultrasound-guided systematic biopsies alone,
increase the diagnosis of clinically relevant prostate cancer. (Bhattacharya et al., 2022)

PIRADS (Prostate Imaging Reporting and Data System) was established by key global experts in
prostate imaging from America and Europe (European Society of Urogenital Radiology (ESUR),
American College of Radiology (ACR)) to facilitate and standardize prostate MRI and assess the risk
of clinically significant prostate cancer (PCA). (Michaely et al., 2022)

The combination of different anatomic and functional magnetic resonance imaging (MRI) sequences
has led to the development of multiparametric MRI, which has been proven to be capable of
detecting and visualizing prostate cancer lesions. However, like all imaging techniques, its diagnostic
performance is highly dependent on radiologist training and expertise. (Mehralivand et al., 2022)

Expert readers assess problems more accurately and with less ambiguity. Their high level of
experience also makes it feasible to adopt MRI approaches that avoid contrast medium injections
(parametric MRI), helping to reduce costs and increase patient throughput. Furthermore, reader
experience improves the consistency and reliability of MRI data for clinical decision-making by
reducing variability in clinically significant cancer yields within the PI-RADS suspicion categories.
(Turbey and Padhani, 2019)

2.2 Role of AI in prostate cancer detection:

A new age of customized healthcare is upon us. It is no longer thought that there is a "one treatment
fits all" solution. This is primarily because of the advancements in genomics. Genetic predispositions
to pathologies like diabetes or cancer can be predicted through genomics. Although characterizing a
single genome is becoming more accessible, it also produces a vast amount of data. As a result,
machine learning is an ideal resource for genomics applications. (Thenault et al., 2020)

Machine learning uses a mathematical and statistical framework to enable computers to recognize
complex patterns and "learn" from data. The machine learning method typically consists of two
stages: the "learning" stage and the "verification" phase. The learning step is to build a model
capable of handling a given task. ML techniques are categorized as "supervised," "unsupervised,"
and "reinforcement" learning based on the kind of data that is supplied during the learning phase.
Supervised machine learning models include Random Forests (RF), Support Vector Machines
(SVM), and Naïve Bayes categorization. (Thenault et al., 2020)

In the context of prostate cancer, ML algorithms can automatically extract qualitative information
regarding tumor features such as size, shape, texture, and intensity from medical imaging data. This
quantitative information can be given back to a urologist to forecast results and offer objective, datadriven
insights on the traits and behavior of prostate cancers, as well as information on tracking
treatment response. This makes it possible for clinicians to modify treatment programs more quickly
as needed. (Chu et al., 2023)

ANNs are among the most significant supervised machine-learning techniques. ANNs are a system
whose conceptual design was influenced by how biological neurons function. ANNs, like biological
neural networks, are composed of layers of basic information units called nodes that work together
to form an integrated processing network. Training, validation, and verification stages are typically
required with non-overlapping data sets to have working ANNs. The multilayer perceptron has
three neuronal layers and is the most used artificial neural network (ANN) in medical research.
Forward-feeding neurons form an interconnected layer (hidden nodes) between the first and last
layers. (Thenault et al., 2020)

There is a subtype of ANN that shows its usefulness for dividing, categorizing, and identifying
intricate patterns in digital photos (such as those of animals, objects, or cells) called the
Convolutional Neural Network (CNN). CNN's capabilities include the ability to identify the prostate
on MRI pictures and classify skin cancer from digital photos. Prostate cancer (PCa) has been the
subject of an increasing amount of AI research during the last ten years. It may be challenging to
have a comprehensive understanding of the actual condition; however, AI has been suggested as a
tool to aid doctors in the management of PCa. (Thenault et al., 2020)

Artificial intelligence (AI) has showed amazing skills in a variety of medical applications, ranging
from physician-level diagnosis to workflow efficiency, and has the potential to help cancer therapy.
As clinical acceptance of digital histopathology grows, AI can be used more generally in cancer care.
Unlike traditional risk-stratification methods, which are rigid and based on a small number of
variables, AI can learn from enormous amounts of little processed data across multiple modalities.
In contrast to genomic biomarkers, AI systems that use digitized images are less expensive and more
scalable. (Esteva et al., 2022)

2.3 Techniques of prostate cancer detection:

In 2004, the World Health Organization (WHO) approved the Gleason grading system. It has also
been included in the AJCC/UICC staging system and NCCN guidelines. The classical Gleason
grading system consists of five histological growth patterns, and the biopsies or whole slide images
are graded accordingly. Gleason 1 describes the best differentiated, whereas Gleason 5 describes the
least differentiated. Gleason 1 is more correlated and favorable for the prognosis. However, Gleason
5 is correlated with poor prognosis. (Linkon et al., 2021)

The method of Gleason grading involves identifying and categorizing cancer tissue into Gleason
patterns according to the architectural growth patterns of the tumor. A biopsy report is first marked
according to the matching Gleason score, and then it is transformed into an ISUP grade on a scale
from 1 to 5. The Gleason grading method is unique since it relies exclusively on tumor architectural
traits. Instead of emphasizing the worst pattern, the approach concentrates on cytological features.
Two frequent patterns are considered by the Gleason score (GS). This approach is among the best
prognostic indicators available for the identification of prostate cancer. The ISUP grade system plays
an integral part in cancer detection for patients worldwide. Correctness in Gleason grading is vital to
help pathologists make the correct decision. (Linkon et al., 2021)

Grayscale transrectal ultrasound-guided biopsy is another method used to diagnose prostate cancer.
Although the prostate gland can be consistently identified by grayscale ultrasonography, low signalto-
noise ratio and artifacts (such as shadowing and speckle) make it difficult for doctors to
distinguish between malignant and non-cancerous parts. According to reports, as little as 40% of
grayscale ultrasound pictures can identify prostate cancer. Cancers reflect far fewer sound echoes
than normal tissue, which makes them seem hypoechoic when seen on ultrasound. In addition to
grayscale ultrasound, other novel ultrasound-based imaging modalities have been proposed,
including color doppler ultrasonography, contrast-enhanced ultrasonography, microultrasonography,
and their combination. These alternative ultrasound-based imaging modalities
provide enhanced image resolution and better visualization of the prostate compared to grayscale
ultrasound and enable prostate cancer detection with better sensitivity compared to grayscale
ultrasound. (Bhattacharya et al., 2022)

(Hosseinzadeh et al., 2022) Described a DL-CAD model that can detect and localize csPCa on
bpMRI. Their study demonstrates that the performance of a DL-CAD system for the detection and
localization of csPCa in biopsy-naive men is improved by using prior knowledge of DL-based zonal
segmentation. A more extensive data set leads to an improved performance, which can potentially
reach expert-level performance when substantially more than 2000 training cases are used.

In order to diagnose prostate cancer using Matlab version 2017b, (Hosseinzadeh et al., 2022) used a
Deep Learning Convolution Neural Network (CNN) model whose performance was compared with
machine learning techniques. GoogleNet from CNN was utilized to train the cancer images using a
transfer learning technique. Robust machine-learning approaches were employed to extract
multimodal features, including texture, morphology, entropy, SIFT, and EFDs, from a cancer
imaging library. The performance of these features was assessed for both individual and combined
features. SVM Gaussian with texture + morphological and EFDs + morphological features, with
sensitivity and total accuracy of 99.71%, demonstrated the best performance when using machine
learning techniques. This was followed by SVM Gaussian with texture + SIFT features, with a
sensitivity of 98.83% and AUC of 0.999. The GoogleNet technique, which employed Deep Learning
CNN, yielded an AUC of 1.00 along with 100% TA, PPV, specificity, and sensitivity. (Abbasi et al.,
2020)

(Yang et al., 2023) presented an automated method for diagnosing prostate cancer utilizing MRI
images and the EOADL-PCDC method. The primary objective of the EOADL-PCDC
methodology is the identification and classification of prostate cancer. The model's four main phases
are SBiLSTM-based classification, EOA-based hyperparameter tuning, CapsNet feature extraction,
and image preprocessing. The EOADL-PCDC method primarily uses image preprocessing to
enhance the quality of the images. Furthermore, CapsNet is used by the EOADL-PCDC approach
for feature extraction. CapsNet's performance is improved by the EOA-based hyperparameter
tuning method. For the categorization of prostate cancer, the EOADL-PCDC method uses the
SBiLSTM model. The benchmark MRI dataset served as the testing ground for an extensive
collection of simulations of the EOADL-PCDC method. According to a number of criteria, the
experimental results demonstrated the EOADL-PCDC method's superior performance over the
other approaches.

(Khosravi et al., 2021) trained a multimodal model to integrate the MR image and pathology score as
predictors and further encode the interdependency between these diagnosis sets. There are two
primary levels of data integration: early fusion (data are integrated before feeding to the model) and
late fusing (different trained models will be integrated using various ML techniques). They used the
early fusion technique and proposed a CNN-based system, AI-biopsy, to fully utilize MR images and
biopsy results to detect prostate cancer. They trained and validated AI-biopsy using MR images of
400 patients that were labeled with histopathology information. In addition, compared to the PIRADS
score, their model indicated higher agreement with biopsy results.

In order to be implemented into clinical routine, an algorithm must be externally evaluated and
trained using ground truth labeling that represents the histological data of the entire prostate gland.
These algorithms will need ongoing quality control and monitoring to ensure their functionality even
after they are used in clinical settings.
Ground truth refers to labels that are annotated by an expert to categorize MRI volumes into
defined classes, for example, PCa versus normal-appearing prostate tissue. This classification is used
for training, testing, and evaluating AI algorithms. The development of AI algorithms depends on
the annotation quality of the histopathological truth in the mpMRI scans. Moreover, the ground
truth must reflect the histopathological information of the entire prostate as accurately as possible to
be clinically relevant and applicable. (Suarez-Ibarrola et al., 2022)


Several groups in the world are exploring routes to improve prostate MRI quality, and recently,
prostate imaging quality (PI-QUAL) was released by European prostate MRI experts. This system
aims to evaluate the diagnostic quality of prostate MRIs using a set of criteria for each pulse
sequence. PI-QUAL has been recently investigated in an interreader agreement study with 103
patients and two dedicated radiologists. The agreement for each single PI-QUAL score was strong
(κ = 0.85 and percent agreement = 84%). The agreement for diagnostic quality of each pulse
sequence was 89% (92/103 scans), 88% (91/103 scans), and 78% (80/103 scans) for T2 weighted
imaging, dynamic contrast-enhanced MRI, and diffusion-weighted imaging, respectively. The actual
impact of PI-QUAL evaluation system on improving prostate MRI quality will soon be reported.
(Turkbey & Haider, 2022)

3. Artificial Intelligence for Prostate Cancer Detection (Materials and Methods)
   
Data Description:

The PANDA dataset (Prostate Cancer GraDe Assessment) was employed to detect Prostate Cancer,
which is made up of high-resolution pathology images from prostate biopsies. The training dataset
(train_df) contains image IDs and their associated labels (isup_grade), which describe the severity of
prostate cancer on an ordinal scale. The parameters of the training and test datasets are represented
by train_df.shape and test_df.shape, respectively. The test dataset (test_df) contains only image IDs
without labels and is meant for evaluation purposes. These biopsy photos are in.tiff format and must
be processed using specialist software.

Image preprocessing:

The preprocessing stage resizes and normalizes the biopsy images before they are sent into the
machine-learning model. The preprocess_image function loads.tiff images from the open slide
library and generate a thumbnail of the provided size (desired_size). These images are then scaled to
224x224 pixels using the PIL package, and then converted to NumPy arrays for use in machine
learning processes. This feature facilitates subsequent model training procedures by guaranteeing
that image formats and dimensions are uniform across the dataset.

Dataset Loading:

The code stores all training images in memory for quick access during model training. It iterates over
every image ID, processes the corresponding.tiff image with the preprocess_image function, and
places the processed images in the x_train array. The images are kept in a 4-dimensional NumPy
array with the following dimensions: (N, 224, 224, 3), where N represents the quantity of training
samples, 224x224 is the image resolution, and 3 is the number of RGB color channels. At the same
time, the cancer grade labels are one-hot encoded and saved in y_train.

Label Transformation (Multi-label Encoding):

To better illustrate the ordinal character of cancer grades, the single-label classification problem is
changed to a multi-label problem. The transformation covers all lower grades for each label up to
the specified grade level. For example, the original label for grade 4 [0, 0, 0, 0, 1] is changed into [1,
1, 1, 1, 1], which includes all grades lower than grade 4. In order to replicate the inherent structure of
cancer severity grading, this technique, which is employed in y_train_multi, guarantees that the
model learns not only the assigned grade but also the accumulated information of all lower grades.

Train-Validation Split:

The dataset is divided into subgroups for training and validation in order to assess model
performance during training. The split ratio is controlled by the TRAIN_VAL_RATIO option,
which is set to 0.27 (27% of the data is allocated for validation). This maintains enough samples for
training and validation while allowing for a balanced distribution of data. The split is performed
using scikit-learn's train_test_split function, which generates x_train, x_val, y_train, and y_val
subsets for the images and labels.

Data Augmentation

To improve the model's generalization capabilities, the algorithm includes a data augmentation
pipeline called ImageDataGenerator. This augmentation approach uses random transformations on
training images, such as zooming (up to 15%), random horizontal and vertical flipping, and filling
empty regions with constant values when alterations exceed image bounds. By incorporating
variations in the training photos, the model learns to recognize patterns under a variety of settings,
increasing its robustness and decreasing overfitting on the training data.

3.1 Machine Learning Techniques

To create a baseline for prostate cancer detection, conventional machine learning models were used
as a first step. The four traditional machine learning models used are—Random Forest, K-Nearest
Neighbors (KNN), Decision Tree, and Naive Bayes—for the task of prostate cancer grade
classification.

The Random Forest ((Rustam & Angie, 2021)) model scored the best accuracy of any machine
learning method, at 41%. However, its precision (43%) and recall (41%) were quite low, indicating
that while it recognized some positive instances, it also missed a large number of real positives. The
model appears to have trouble balancing false positives and false negatives, as evidenced by its F1
score of 39%, which shows an imbalance between precision and recall. Although Random Forest
did not perform at its best for this categorization task, it did show promise overall.

KNN ((Stanley et al., 2024)) performed similarly to Random Forest, achieving 33% accuracy, 31%
precision, and 33% recall. The 31% F1 score suggests a small imbalance in the model's ability to
accurately categorize both positive and negative cases. KNN's poor performance is most likely due
to its reliance on proximity-based predictions, which may not be appropriate for the complexity and
high-dimensionality of the prostate cancer dataset.

The Decision Tree model ((Alzboon et al., 2023)) performed even worse, with an accuracy of 26%
and matching precision, recall, and F1 scores of 26%. These findings indicate that the model was
unable to detect important patterns or trends in the dataset, most likely overfitting or underfitting
due to its unsophisticated decision-making process.

Naive Bayes ((Srivenkatesh Muktevi, 2020)) had the lowest performance, with an accuracy of 24%,
precision of 30%, and recall of 24%, resulting in an F1 score of 23%. These findings show that
Naive Bayes was inefficient at differentiating between different prostate cancer grades, which is
expected given that it is based on strong assumptions (such as feature independence) that are
frequently violated in medical imaging data.

Overall, while Random Forest showed the best performance among the machine learning models,
but the effectiveness of typical machine learning models was limited, highlighting the difficulties
faced by the complicated patterns found in histopathology pictures. These findings underlined the
need for improved techniques, such as deep learning, to automatically learn and extract relevant
features from raw image data.

<img width="572" alt="image" src="https://github.com/user-attachments/assets/01fac13e-3649-4a58-bb60-6aa3fb54250b" />

3.2 Deep Learning Methods

The study evaluated the performance of five established pre-trained deep learning models
(DenseNet121, VGG16, EfficientNetB7, NASNetMobile, and MobileNet) with a new attentionbased
model, VarMIL, for grading prostate cancer. The models were evaluated on both training and
validation datasets using a variety of measures like as accuracy, precision, recall, and F1-score. These
criteria offer a full assessment of each model's ability to generalize and accurately predict cancer
grades.

DenseNet121((Rao & Sankar, 2022)) had a training accuracy of 85.69% and a validation accuracy of
97.94% indicating consistent performance. The model has a validation precision of 85.54%, a recall
of 79.05%, and an F1-score of 81.00%, demonstrating its balanced ability to properly identify
positive cases while reducing false negatives. DenseNet121's densely connected layers allowed for
rapid feature extraction, resulting in good generalization across datasets.

The VGG16 model ((Pm et al., 2024)) demonstrated its capacity to match training data with a high
training accuracy of 97.83%. The model surprisingly achieved a 100% validation accuracy, indicating
successful task mastery. However, the difference between validation accuracy and other metrics,
such as recall (76.72%) and F1-score (77.53%), suggests that there may be overfitting or recall
constraints. Despite this, VGG16 displayed high classification abilities overall.

EfficientNetB7 ((Alici-Karaca & Akay, 2024)) attained a training accuracy of 87.47% and a
validation accuracy of 98.54%, demonstrating strong learning and outstanding generalization. While
its precision on validation data was good (86.48%), a lower recall of 80.94% indicates that the model
may have struggled to detect all positive occurrences, resulting in false negatives. The model's F1-
score of 82.45% demonstrates an overall balance of precision and recall.

NASNet7 ((Balaha et al., 2024)) had a training accuracy of 88.66% and a validation accuracy of
90.37%, indicating good learning ability and generalization. The model maintained a validation
precision of 79.94%, recall of 80.75%, and F1-score of 80.17%, indicating the capacity to effectively
manage both positive and negative situations. 

The lightweight MobileNet model ((Rao & Sankar, 2022)) had a training accuracy of 85.11% and a
validation accuracy of 90.62%, making it a good choice for resource-constrained situations. It has
high validation precision (91.40%) but lower recall (69.29%), resulting in an F1-score of 78.65%.
While its recall was less competitive, it still had a huge advantage in computational efficiency.

The newly introduced VarMIL ((Darbandsari et al., 2024)) technique underperformed severely when
compared to the other models, with a training accuracy of 26.00% and a validation accuracy of 25%.
With a validation precision of 21.30%, recall of 25.38%, and F1-score of 19.98%, the model
struggled to detect meaningful patterns in the data. This emphasizes the need for additional refining
and optimization of the VarMIL methodology.

<img width="870" alt="image" src="https://github.com/user-attachments/assets/198eb107-4b1f-49ad-9aef-be69507adbf2" />

4. Discussion:
   
This study examined how well many machine learning and deep learning models performed in
assessing the grade of prostate cancer. It has been shown that deep learning models can handle
complex histopathological imaging data, consistently outperforming typical machine learning
techniques.

The two most promising models among those analyzed were ENet7 and DenseNet121. While
attaining high validation accuracies of 98.54% and 97.94%, respectively, both maintained balanced
accuracy and recall ratings. These results show that they are robust and have good generalization to
fresh data. VGG16 raised concerns about overfitting because it demonstrated decreased recall
(77.48%) and precision (81.59%), although achieving perfect validation accuracy (100%). This
implies that even though it performs well on training data, it might not be able to generalize well to
larger datasets.

MobileNet performed remarkably well considering its simplicity, with a validation precision of
91.40%. Despite this, its very low recall (69.29%) suggests that it might not be able to identify every
positive case, potentially leading to missed detections in critical outcomes.
The performance and complexity of NASNet7 were well-balanced, and the validation metrics
showed consistency. With only 26% training accuracy and 25% validation accuracy, the New
Method (VarMIL) significantly underperformed pre-trained models despite being creative. This
could be due to its architectural originality, which necessitates additional refinement and
optimization in order to properly extract and learn patterns from high-dimensional medical imaging
data. In contrast, machine learning algorithms such as Random Forest, KNN, Decision Tree, and
Naive Bayes struggled to generalize, resulting in low accuracies and unbalanced precision-recall
results. This disparity highlights the limitations of conventional techniques for evaluating intricate
and extensive datasets, like histopathology images.

Overall, the results highlight the benefits of applying pre-trained architectures and transfer learning
to medical imaging applications. Finding a balance between classification performance and
computational efficiency is still a major objective for future research, even if lightweight models like
MobileNet have demonstrated promise in resource-constrained scenarios.

5. Conclusion
   
This research demonstrates the advantage of deep learning approaches over typical machine learning
methods for determining prostate cancer grade. Pre-trained models, particularly EfficientNetB7 and
DenseNet121, performed well in terms of accuracy, precision, and recall, demonstrating their
suitability for clinical diagnostic applications. Despite its potential, VGG16 and MobileNet's
performance is constrained by problems like overfitting and low recall, respectively.
In spite of its innovative attention-based design, the new VarMIL model did not perform as
expected in this preliminary test. It provides a good starting point for future studies into more
specialized designs for processing histopathology images, though. In order to increase sensitivity and
accuracy, future work should focus on improving these models, adding better attention mechanisms,
and applying hybrid approaches.

Future studies should focus on improving lightweight models like MobileNet for higher recall, fixing
overfitting in models like VGG16, and streamlining model architectures. Moreover, the poor
performance of the New Method highlights the need for improvements in feature extraction and
loss function design. Model performance and reliability in prostate cancer detection tasks may be
further improved by utilizing cutting-edge strategies like ensemble modeling, transfer learning, or
extra data augmentation.


6. References:
   
Abbasi, A. A., Hussain, L., Awan, I. A., Abbasi, I., Majid, A., Nadeem, M. S. A., & Chaudhary, Q.-A.
(2020). Detecting prostate cancer using deep learning convolution neural network with transfer
learning approach. Cognitive Neurodynamics, 14(4), 523–533. https://doi.org/10.1007/s11571-020-
09587-5

Alici-Karaca, D., & Akay, B. (2024). An Efficient Deep Learning Model for Prostate Cancer
Diagnosis. IEEE Access, 12, 150776–150792. https://doi.org/10.1109/ACCESS.2024.3472209

Alshareef, A. M., Alsini, R., Alsieni, M., Alrowais, F., Marzouk, R., Abunadi, I., & Nemri, N. (2022).
Optimal Deep Learning Enabled Prostate Cancer Detection Using Microarray Gene Expression.
Journal of Healthcare Engineering, 2022, 1–12. https://doi.org/10.1155/2022/7364704

Alzboon et al. (2023). Prostate Cancer Detection and Analysis using Advanced Machine Learning.
Vol. 14, No. 8, 2023.

Balaha, H. M., Shaban, A. O., El-Gendy, E. M., & Saafan, M. M. (2024). Prostate cancer grading
framework based on deep transfer learning and Aquila optimizer. Neural Computing and
Applications, 36(14), 7877–7902. https://doi.org/10.1007/s00521-024-09499-z

Bhattacharya, I., Khandwala, Y. S., Vesal, S., Shao, W., Yang, Q., Soerensen, S. J. C., Fan, R. E.,
Ghanouni, P., Kunder, C. A., Brooks, J. D., Hu, Y., Rusu, M., & Sonn, G. A. (2022). A review of
artificial intelligence in prostate cancer detection on imaging. Therapeutic Advances in Urology, 14,
17562872221128791. https://doi.org/10.1177/17562872221128791

Chu, T. N., Wong, E. Y., Ma, R., Yang, C. H., Dalieh, I. S., & Hung, A. J. (2023). Exploring the Use
of Artificial Intelligence in the Management of Prostate Cancer. Current Urology Reports, 24(5),
231–240. https://doi.org/10.1007/s11934-023-01149-6

Darbandsari, A., Farahani, H., Asadi, M., Wiens, M., Cochrane, D., Khajegili Mirabadi, A., Jamieson,
A., Farnell, D., Ahmadvand, P., Douglas, M., Leung, S., Abolmaesumi, P., Jones, S. J. M., Talhouk,
A., Kommoss, S., Gilks, C. B., Huntsman, D. G., Singh, N., McAlpine, J. N., & Bashashati, A.
(2024). AI-based histopathology image analysis reveals a distinct subset of endometrial cancers.
Nature Communications, 15(1), 4973. https://doi.org/10.1038/s41467-024-49017-2

Esteva, A., Feng, J., Van Der Wal, D., Huang, S.-C., Simko, J. P., DeVries, S., Chen, E., Schaeffer,
E. M., Morgan, T. M., Sun, Y., Ghorbani, A., Naik, N., Nathawani, D., Socher, R., Michalski, J. M.,
Roach, M., Pisansky, T. M., Monson, J. M., Naz, F., … Mohamad, O. (2022). Prostate cancer
therapy personalization via multi-modal deep learning on randomized phase III clinical trials. Npj
Digital Medicine, 5(1), 71. https://doi.org/10.1038/s41746-022-00613-w

Hosseinzadeh, M., Saha, A., Brand, P., Slootweg, I., De Rooij, M., & Huisman, H. (2022). Deep
learning–assisted prostate cancer detection on bi-parametric MRI: Minimum training data size
requirements and effect of prior knowledge. European Radiology, 32(4), 2224–2234.
https://doi.org/10.1007/s00330-021-08320-y

Iqbal, S., Siddiqui, G. F., Rehman, A., Hussain, L., Saba, T., Tariq, U., & Abbasi, A. A. (2021).
Prostate Cancer Detection Using Deep Learning and Traditional Techniques. IEEE Access, 9,
27085–27100. https://doi.org/10.1109/ACCESS.2021.3057654

Khosravi, P., Lysandrou, M., Eljalby, M., Li, Q., Kazemi, E., Zisimopoulos, P., Sigaras, A., Brendel,
M., Barnes, J., Ricketts, C., Meleshko, D., Yat, A., McClure, T. D., Robinson, B. D., Sboner, A.,
Elemento, O., Chughtai, B., & Hajirasouliha, I. (2021). A Deep Learning Approach to Diagnostic
Classification of Prostate Cancer Using Pathology–Radiology Fusion. Journal of Magnetic
Resonance Imaging, 54(2), 462–471. https://doi.org/10.1002/jmri.27599

Linkon, A. H. Md., Labib, Md. M., Hasan, T., Hossain, M., & Jannat, M.-E.-. (2021). Deep learning
in prostate cancer diagnosis and Gleason grading in histopathology images: An extensive study.
Informatics in Medicine Unlocked, 24, 100582. https://doi.org/10.1016/j.imu.2021.100582

Mehralivand, S., Yang, D., Harmon, S. A., Xu, D., Xu, Z., Roth, H., Masoudi, S., Kesani, D., Lay,
N., Merino, M. J., Wood, B. J., Pinto, P. A., Choyke, P. L., & Turkbey, B. (2022). Deep learningbased
artificial intelligence for prostate cancer detection at biparametric MRI. Abdominal Radiology,
47(4), 1425–1434. https://doi.org/10.1007/s00261-022-03419-2

Michaely, H. J., Aringhieri, G., Cioni, D., & Neri, E. (2022). Current Value of Biparametric Prostate
MRI with Machine-Learning or Deep-Learning in the Detection, Grading, and Characterization of
Prostate Cancer: A Systematic Review. Diagnostics, 12(4), 799.
https://doi.org/10.3390/diagnostics12040799

Padhani, A. R., & Turkbey, B. (2019). Detecting Prostate Cancer with Deep Learning for MRI: A
Small Step Forward. Radiology, 293(3), 618–619. https://doi.org/10.1148/radiol.2019192012
Patel, A., Singh, S. K., & Khamparia, A. (2021). Detection of Prostate Cancer Using Deep Learning
Framework. IOP Conference Series: Materials Science and Engineering, 1022(1), 012073.
https://doi.org/10.1088/1757-899X/1022/1/012073

Pm, E., Ananthanagu, U., Mara, G. C., B, I., Thomas, R., & Mathkunti, N. M. (2024). A
Breakthrough Approach for Prostate Cancer Identification Utilizing VGG-16 CNN Model with
Migration Learning. 2024 IEEE International Conference on Interdisciplinary Approaches in
Technology and Management for Social Innovation (IATMSI), 1–6.
https://doi.org/10.1109/IATMSI60426.2024.10503011

Rao, H. V. R., & Sankar, Dr. V. R. (2022). Precision in Prostate Cancer Diagnosis: A
Comprehensive Study on Neural Networks. Journal of Wireless Mobile Networks, Ubiquitous
Computing, and Dependable Applications, 15(2), 109–122.
https://doi.org/10.58346/JOWUA.2024.I2.008

Rustam, Z., & Angie, N. (2021). Prostate Cancer Classification Using Random Forest and Support
Vector Machines. Journal of Physics: Conference Series, 1752(1), 012043.
https://doi.org/10.1088/1742-6596/1752/1/012043

Srivenkatesh Muktevi, Dr. M. (2020). Prediction of Prostate Cancer using Machine Learning
Algorithms. International Journal of Recent Technology and Engineering (IJRTE), 8(5), 5353–5362.
https://doi.org/10.35940/ijrte.E6754.018520

Stanley et al. (n.d.). A Comparative and Predictive Analysis of Prostate Cancer Diagnosis and
Treatment using Decision Tree, Neural Network, Support Vector Machine, Random Forest and KNearest
Neighbor KNN Classification Algorithms. ICONIC RESEARCH AND ENGINEERING
JOURNALS.

Suarez-Ibarrola, R., Sigle, A., Eklund, M., Eberli, D., Miernik, A., Benndorf, M., Bamberg, F., &
Gratzke, C. (2022). Artificial Intelligence in Magnetic Resonance Imaging–based Prostate Cancer
Diagnosis: Where Do We Stand in 2021? European Urology Focus, 8(2), 409–417.
https://doi.org/10.1016/j.euf.2021.03.020

Tătaru, O. S., Vartolomei, M. D., Rassweiler, J. J., Virgil, O., Lucarelli, G., Porpiglia, F., Amparore,
D., Manfredi, M., Carrieri, G., Falagario, U., Terracciano, D., De Cobelli, O., Busetto, G. M.,
Giudice, F. D., & Ferro, M. (2021). Artificial Intelligence and Machine Learning in Prostate Cancer
Patient Management—Current Trends and Future Perspectives. Diagnostics, 11(2), 354.
https://doi.org/10.3390/diagnostics11020354

Thenault, R., Kaulanjan, K., Darde, T., Rioux-Leclercq, N., Bensalah, K., Mermier, M., Khene, Z.,
Peyronnet, B., Shariat, S., Pradère, B., & Mathieu, R. (2020). The Application of Artificial
Intelligence in Prostate Cancer Management—What Improvements Can Be Expected? A Systematic
Review. Applied Sciences, 10(18), 6428. https://doi.org/10.3390/app10186428

Turkbey, B., & Haider, M. A. (2022). Deep learning-based artificial intelligence applications in
prostate MRI: Brief summary. The British Journal of Radiology, 95(1131), 20210563.
https://doi.org/10.1259/bjr.20210563

Van Booven, D. J., Kuchakulla, M., Pai, R., Frech, F. S., Ramasahayam, R., Reddy, P., Parmar, M.,
Ramasamy, R., & Arora, H. (2021). A Systematic Review of Artificial Intelligence in Prostate Cancer.
Research and Reports in Urology, Volume 13, 31–39. https://doi.org/10.2147/RRU.S268596

Vente, C. D., Vos, P., Hosseinzadeh, M., Pluim, J., & Veta, M. (2021). Deep Learning Regression for
Prostate Cancer Detection and Grading in Bi-Parametric MRI. IEEE Transactions on Biomedical
Engineering, 68(2), 374–383. https://doi.org/10.1109/TBME.2020.2993528

Yang, E., Shankar, K., Kumar, S., Seo, C., & Moon, I. (2023). Equilibrium Optimization Algorithm
with Deep Learning Enabled Prostate Cancer Detection on MRI Images. Biomedicines, 11(12),
3200. https://doi.org/10.3390/biomedicines11123200
