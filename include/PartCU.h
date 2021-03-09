#ifndef INC_2048_GAME2048_H
#define INC_2048_GAME2048_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include<fstream>

#include <gegelati.h>

class PartCU : public Learn::LearningEnvironment
{
private:

    // ----- Constant -----
    // Number of different actions for the Agent
    static const uint8_t  NB_ACTIONS = 6;
    // On 1.4M elements in the database, we use 80% : 
    static const uint32_t NB_TRAINING_ELEMENTS = 1136424;

    // Randomness control
    Mutator::RNG rng;

    /**
    * \brief Available actions for the LearningAgent.
    * 6 different splits :
    *   - NP  (0) : Non-Partitionning
    *   - QT  (1) : Quad-Tree Partitionning
    *   - BTH (2) : Binary-Tree  Horizontal
    *   - BTV (3) : Binary-Tree  Vertical
    *   - TTH (4) : Ternary-Tree Horizontal
    *   - TTV (5) : Ternary-Tree Vertical
    */
    const std::vector<uint64_t> availableActions;

    /**
    * \brief Learning Agent score for the current job
    * +1 each time he chose the best split, else +0
    */
    uint64_t score;

    /**
    * \brief Current State of the environment
    * Vector containing all pixels of the current CU
    * CU are 32x32 => 1024 values
    */
    Data::PrimitiveTypeArray<uint8_t> currentCU;

    // ---------- Intern Variables ----------
    // Optimal split for the current CU extract from the .bin file
    uint8_t optimal_split;

public:
    // ---------- Intern Variables ----------

    // Number of actions per Evaluation, initialized by params.json
    const uint64_t MAX_NB_ACTIONS_PER_EVAL;
    // In order to accelerate the Learning we preload targets (CUs) which will be used for X generations
    const uint8_t  NB_GENERATION_BEFORE_TARGETS_CHANGE;

    /**
    * \brief List of CU datas and their corresponding optimal split
    * Each of those vectors contains ${MAX_NB_ACTIONS_PER_EVAL} elements and is updated every ${NB_GENERATION_BEFORE_TARGETS_CHANGE}
    */
    static std::vector<Data::PrimitiveTypeArray<uint8_t>*> trainingTargetsCU;
    static std::vector<uint8_t> trainingTargetsOptimalSplits;
    // Index of the actual loaded CU 
    static uint64_t actualCU;

    // Constructor
    PartCU(std::vector<uint64_t> actions, const uint64_t nbActionsPerEval, const uint8_t nbGeneration) : LearningEnvironment(NB_ACTIONS),
                                            availableActions(actions),
                                            MAX_NB_ACTIONS_PER_EVAL(nbActionsPerEval),
                                            NB_GENERATION_BEFORE_TARGETS_CHANGE(nbGeneration),
                                            score(0),
                                            currentCU(32*32),
                                            optimal_split(6)   // Unexisting split
                                            {}

    uint8_t getNbGenerationsBeforeTargetChange();
    void LoadNextCU();
    Data::PrimitiveTypeArray<uint8_t>* getRandomCU(int index);

    // -------- LearningEnvironment --------
    LearningEnvironment* clone() const;
    bool isCopyable() const;
    void doAction(uint64_t actionID);
    void reset(size_t seed = 0, Learn::LearningMode mode = Learn::TRAINING);
    std::vector<std::reference_wrapper<const Data::DataHandler>> getDataSources();
    double getScore() const;
    bool isTerminal() const;
};

#endif //INC_2048_GAME2048_H
