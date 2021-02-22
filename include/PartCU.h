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
    //static const size_t REWARD_HISTORY_SIZE = 300;
    static const uint8_t  NB_ACTIONS = 6;
    static const uint32_t NB_TRAINING_ELEMENTS = 1136424;
    static const uint32_t MAX_NB_ACTIONS_PER_EVAL = 1000; // Regarder comment le recup depuis le params.json

    // Reward history for score computation
    //double rewardHistory[REWARD_HISTORY_SIZE];

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

    // ----- Intern Variables -----
    //uint32_t nbSplitsTotal;
    uint32_t nbSplitsJob; // Used in isTerminal() but dirty, it has to be another way ...

    // Optimal split for the current CU extract from the .bin file
    uint8_t optimal_split;

    // Vector containing CU describing files numbers, from 0 to NB_TRAINING_ELEMENTS in a stochastic order
    std::vector<uint32_t> CU_list;

public:
    // Constructor
    PartCU(std::vector<uint64_t> actions) : LearningEnvironment(NB_ACTIONS), availableActions(actions), score(0), nbSplitsJob(0), currentCU(32*32) { /*this->InitRandomList();*/ }

    void InitRandomList();  // Unused for the moment
    bool LoadNextCU();

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
