#ifndef INC_2048_GAME2048_H
#define INC_2048_GAME2048_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include<fstream>

#include <gegelati.h>
// #include <array2DWrapper.h>

class PartCU : public Learn::ClassificationLearningEnvironment {
private:

    // ----- Constant -----
    // Number of different actions for the Agent
    static const uint8_t NB_ACTIONS = 6;
    // On 1.4M elements in the database, we use 80% : 
    static const uint32_t NB_TRAINING_ELEMENTS = 329999; // Pour la database balanced, sinon : 1136424;

    // Randomness control
    Mutator::RNG rng;

    /**
    * \brief Available actions for the LearningAgent.
    * 6 different splits :
    *   - NP  (0) : Non-Partitionning
    *   - QT  (1) : Quad-Tree    Partitionning
    *   - BTH (2) : Binary-Tree  Horizontal
    *   - BTV (3) : Binary-Tree  Vertical
    *   - TTH (4) : Ternary-Tree Horizontal
    *   - TTV (5) : Ternary-Tree Vertical
    */
    //const std::vector<uint64_t> availableActions;

    /**
    * \brief Learning Agent score for the current job
    * +1 each time he chose the best split, else +0
    */
    //uint64_t score; // Managed by ClassificationLearningEnvironment

    /**
    * \brief Current LearningMode of the LearningEnvironment.
    * Either TRAINING, either VALIDATION, switch set of preloaded targets
    * (TRAINING : trainingTargetsCU, VALIDATION : validationTargetsCU)
    */
    Learn::LearningMode currentMode;

    /**
    * \brief Current State of the environment
    * Vector containing all pixels of the current CU
    * CU are 32x32 => 1024 values
    */
    Data::PrimitiveTypeArray<uint8_t> currentCU;

    // ---------- Intern Variables ----------
    // Optimal split for the current CU extract from the .bin file
    //uint8_t optimal_split;   // Now : this->currentClass

    void LoadNextCU();

public:
    // ---------- Intern Variables ----------

    // Number of actions per Evaluation, initialized by params.json
    const uint64_t NB_TRAINING_TARGETS;
    // In order to accelerate the Learning we preload targets (CUs) which will be used for X generations
    const uint64_t  NB_GENERATION_BEFORE_TARGETS_CHANGE;

    /**
    * \brief List of CU datas and their corresponding optimal split
    * Each of those vectors contains ${NB_TRAINING_TARGETS} elements and is updated every ${NB_GENERATION_BEFORE_TARGETS_CHANGE}
    */
    static std::vector<Data::PrimitiveTypeArray<uint8_t> *> *trainingTargetsCU; //  PrimitiveTypeArray / Array2DWrapper
    static std::vector<uint8_t> *trainingTargetsOptimalSplits;
    // Index of the actual loaded CU 
    uint64_t actualTrainingCU;
    // ****** VALIDATION Arguments ******
    const uint64_t NB_VALIDATION_TARGETS;       // default 1 000
    static std::vector<Data::PrimitiveTypeArray<uint8_t>*> *validationTargetsCU; //  PrimitiveTypeArray / Array2DWrapper
    static std::vector<uint8_t> *validationTargetsOptimalSplits;
    uint64_t actualValidationCU;

    // Constructor
    PartCU(std::vector<uint64_t> actions, const uint64_t nbActionsPerEval, const uint64_t nbGeneTargetChange, const uint64_t nbValidationTarget, size_t seed)
            : ClassificationLearningEnvironment(NB_ACTIONS),
              rng(seed),
              //availableActions(actions),
              //score(0),
              currentMode(Learn::LearningMode::TRAINING),
              currentCU(32 * 32),
              //optimal_split(6),   // Unexisting split
              NB_TRAINING_TARGETS(nbActionsPerEval),
              NB_GENERATION_BEFORE_TARGETS_CHANGE(nbGeneTargetChange),
              actualTrainingCU(0),
              NB_VALIDATION_TARGETS(nbValidationTarget),
              actualValidationCU(0) {}

    Data::PrimitiveTypeArray<uint8_t>* getRandomCU(uint64_t index, Learn::LearningMode mode);
    void printClassifStatsTable(const Environment& env, const TPG::TPGVertex* bestRoot, const int numGen, std::string const outputFile);

    // -------- LearningEnvironment --------
    LearningEnvironment *clone() const;
    bool isCopyable() const;
    void doAction(uint64_t actionID);
    void reset(size_t seed = 0, Learn::LearningMode mode = Learn::TRAINING);
    std::vector<std::reference_wrapper<const Data::DataHandler>> getDataSources();
    double getScore() const;
    bool isTerminal() const;
};

#endif //INC_2048_GAME2048_H
