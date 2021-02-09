#ifndef INC_2048_GAME2048_H
#define INC_2048_GAME2048_H

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <gegelati.h>

class PartCU : public Learn::LearningEnvironment
{
private:

    // ----- Constant -----
    static const size_t REWARD_HISTORY_SIZE = 300;

    // Reward history for score computation
    double rewardHistory[REWARD_HISTORY_SIZE];

    // Randomness control
    Mutator::RNG rng;

    /**
    * \brief Available actions for the LearningAgent.
    *
    * Each number $a$ in this list, corresponds to ....
    */
    const std::vector<uint64_t> availableActions;

    /**
    * \brief Current State of the environment
    *
    * ....
    */
    Data::PrimitiveTypeArray<int> board;

    // ----- Intern Variables -----
    // ...

protected:
    // Setters
    // Getters

public:
    // Constructor
    PartCU(std::vector<uint64_t> actions) : /**board(16),*/ availableActions(actions) {} // Init the environment

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
