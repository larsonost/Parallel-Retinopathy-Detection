# SPRINT: Speedy Parallel Recognition Interface for
Severe Diabetic Retinopathy


Given the wide-scale availability of high dimensional and high sample count datasets primed for
classification tasks, the need to move away from training
neural-network-based classifiers off devices (on CPUs) and
onto devices (on GPUs) is ever-growing. One such example
of a domain where high dimensional data proliferates
is in medical imaging applications. This paper evaluates
the speedup present in training a Convolutional Neural
Network (CNN) both on and off devices over a Diabetic
Retinopathy Disease Severity dataset containing FUNDUS
images of the retina. This paper explores the performance
of a single-CNN, and the speedup over a single epoch is
compared across on-device and off-device training methods. Additionally, the overall accuracy of the CNN when
trained over the respective Diabetic Retinopathy dataset is
compared and shown to exceed a prediction based solely
on the dataset’s label distributions.