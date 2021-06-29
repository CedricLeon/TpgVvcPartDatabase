# TpgVvcPartDatabase

Using the *Gegelati* library to test **Tangled Program Graphs** implementation on **VVC (Versatile Video Coding)** Partitioning process.



## VVC Partitioning

VVC owns 6 different splits :

- Non-Partitioning (**NP**), ID: 0 ;
- Quad-Tree Partitioning (**QT**), ID: 1 ;
- Binary-Tree Horizontal Partitioning (**BTH**), ID: 2 ;
- Binary-Tree Vertical Partitioning (**BTV**), ID: 3 ;
- Ternary-Tree Horizontal Partitioning (**TTH**), ID: 4 ;
- Ternary-Tree Vertical Partitioning (**TTV**), ID: 5.

The idea is to create a single TPG or an association of several TPGs choosing the split from the pixel values of the Coding Unit (CU), the input image.

This problem is very close to a classification type problem so we used 2 different *Gegelati* LearningEnvironment: `LearningEnvironment` (default) and `ClassificationLearningEnvironment` (classification).



## Every different solutions

Different solutions and implementations were tested and each code differs from others. Whatever differences between solutions were too complex to create a parent class or interface.

Hence, each solution code really looks like the first which is not optimal in coding but sufficient for test and research work.

Moreover, in order to accelerate the training we avoid opening and reading CU files everytime it's needed, we preload these targets into 2 different vectors : `trainingTargets` and `validationTargets`.

`validationTargets` is loaded once at the very beginning of the training and contains `nbValidationTarget` (usually 1.000).

But `trainingTargets` is changed every `nbGenerationTargetChange`(usually 30) and contains `nbValidationTarget` (usually 10.000).

Each solution has its own executable.

### Classic classification TPG

- LearningEnvironment class: **ClassEnv** (*include/classification/ClassEnv.h* + *src/classification/ClassEnv.cpp*)
- Main: *src/classification/classTPG.cpp* 

The first implementation realized implements a simple TPG using as input a `PrimitiveTypeArray2D` of 32x32 `uint8_t` (CU pixel values).

This environment actually uses the `ClassificationLearningEnvironment` of *Gegelati* but can easily use the default `LearningEnvironment.

### Default Binary Environment

- LearningEnvironment class: **BinaryDefaultEnv** (*include/binary/DefaultBinaryEnv.h* + *src/binary/DefaultBinaryEnv.cpp*)
- Main: *src/binary/binaryTPGs.cpp* 

This solution allows to train binary TPGs, these TPGs only have 2 different actions :

- SPLIT: A specific split in which they are specialized
- OTHERS: every other split than their own

Then they are called one by one, "in a waterfall",  until one of them says "It's my split".

The test in inference can be realized in *src/binary/inferenceBinaryTPGs*. We distinguish 2 "waterfalls" :

- In VVC order: NP -> QT -> BTH -> BTV -> TTH -> TTV
- In success order (during my internship): TTV -> NP -> QT -> BTH -> BTV -> TTH

Note: The last TPG of the waterfall isn't necessary since the **OTHERS** action of the second last is only the remaining split.



Specific to the default `LearningEnvironment` the score is computed in the following way :

- +1 if the taken action corresponds to the optimal split
- +0 else

### Classification Binary Environment

- LearningEnvironment class: **BinaryClassifEnv** (*include/binary/ClassBinaryEnv.h* + *src/binary/ClassBinaryEnv.cpp*)
- Main: *src/binary/binaryTPGs.cpp*

This solution works exactly as the Default Binary Environment but inherits from the `ClassificationLearningEnvironment` of *Gegelati* (and therefore uses F1-score).



### Features Environment

- LearningEnvironment class: **FeaturesEnv** (*include/features/FeaturesEnv.h* + *src/features/FeaturesEnv.cpp*)
- Main: *src/features/featuresTPG.cpp*

In this solution we changed the input of the agent. We don't use pixel values (`uint8_t`) anymore but an array of features (spacial probabilities of split computed by a CNN) which are `double`.

The agent is not binary and can choose between the 6 VVC splits. 

The class **FeaturesEnv** can load the features of a CU from 2 type of file:

- the original file format, a .csv containing 11 lines, the first with column titles and the 10 next each corresponding to a  different CU (*getRandomCUFeaturesFromOriginalCSVFile* function) ;
- the simplified format (used to balance the original database), a .csv with only 1 line for 1 CU (*getRandomCUFeaturesFromSimpleCSVFile* function). 



### Binary Features Environment

- LearningEnvironment class: **BinaryFeaturesEnv** (*include/features/BinaryFeaturesEnv.h* + *src/features/BinaryFeaturesEnv.cpp*)
- Main: *src/features/binaryFeaturesTPG.cpp*

This environment is used to train a binary TPG using CU features as input. It's a mix between the **Classification Binary Environment** and the **Features Environment**.

This environment is the most generalized, its main uses `argv[]` to generalized parameters as the `seed`, the `CUsize`, the number of training elements, etc...

This environment also owns a seccond main for inference: *inferenceBinaryFeaturesTPG.cpp*.

