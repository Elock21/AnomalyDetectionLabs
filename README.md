L1
E1

Red ones are anomalies and blue ones are noramal data


E2

The ROC curve does not change when the value of the contamination differs i also added randomstate = 42 to keep the data constant so i can observe if the curve changes

E3

The Z-score is a good algorithm to identify the anomalies that are further from the normal data but it kinda gives false positives at the edge of the normal data "cluster"(not so much a cluster because of the representation in a straight line) so it misses the closer anomalies and it being unidimenisonal it is repesented in a straight line ( only on one axis)

E4

The multidimensional Z-score has the same limitation as the unidimeniosnal one, it detects well the anomalies that are far from the center of the normal data, it also misses anomalies inside it and produces false positives at the edge of normal data cluster, the balanced accuracy = 0.7333333333333334
 