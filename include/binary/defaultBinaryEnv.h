#ifndef TPGVVCPARTDATABASE_DEFAULTBINARYENV_H
#define TPGVVCPARTDATABASE_DEFAULTBINARYENV_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include <gegelati.h>
/**
* \brief Heritage of the LearningEnvironment Interface
* This class defines the environment for a binary TPG interacting with a database
*/
class BinaryDefaultEnv : public Learn::LearningEnvironment {

private:
    /**
    * \brief Number of different actions for the Agent
    **/
    static const uint8_t NB_ACTIONS = 2;

    /**
    * \brief Randomness control
    **/
    Mutator::RNG rng;

    /**
    * \brief Available actions for the LearningAgent.
    * 2 different actions :
    *   - (0) : Every other split than the specialized one
    *   - (1) : The specialized action
     *
    * Example : The trained agent will be specialized on the BTV (default : N°3) split then :
    *   - If the optimal split to be chosen is not BTV the agent should pick (0)
    *   - If the optimal split is actually BTV the agent should pick (1)
    */
    const std::vector<uint64_t> availableActions;

    /**
    * \brief Index of the action which the TPG is specialized in
    */
    const int specializedAction;

    /**
    * \brief Score for the current job (+1 when best split is chosen, else +0)
    */
    double score;

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
    Data::PrimitiveTypeArray2D<uint8_t> currentCU;

    /**
    * \brief Optimal split for the current CU extract from the .bin file
    */
    uint8_t optimal_split;

public:
    // ********************************************* Intern Variables *********************************************
    /**
    * \brief Number of training element
    **/
    const uint32_t NB_TRAINING_ELEMENTS;

    /**
     * \brief Number of VALIDATION target
     */
    const uint64_t NB_TRAINING_TARGETS;
    /**
     * \brief Number of generation between training set reloads
     * (In order to accelerate the Learning we preload targets (CUs) which are used for X generations)
     */
    const uint64_t  NB_GENERATION_BEFORE_TARGETS_CHANGE;
    /**
    * \brief Number of VALIDATION target
    */
    const uint64_t NB_VALIDATION_TARGETS;

    // ********************************************* TRAINING Arguments *********************************************
    /**
    * \brief Vector of CU datas
    * This vector contains ${NB_TRAINING_TARGETS} elements and is updated every ${NB_GENERATION_BEFORE_TARGETS_CHANGE}
    */
    static std::vector<Data::PrimitiveTypeArray2D<uint8_t> *> *trainingTargetsCU;
    /**
    * \brief Vector of CU optimal split
    * This vector contains ${NB_TRAINING_TARGETS} elements and is updated every ${NB_GENERATION_BEFORE_TARGETS_CHANGE}
    */
    static std::vector<uint8_t> *trainingTargetsOptimalSplits;
    /**
    * \brief Index of the actual loaded CU for training
    */
    uint64_t actualTrainingCU;

    // ********************************************* VALIDATION Arguments *********************************************
    /**
    * \brief Vector of validation CU datas
    * This vector contains ${NB_VALIDATION_TARGETS} elements and is loaded once at training beginning
    */
    static std::vector<Data::PrimitiveTypeArray2D<uint8_t>*> *validationTargetsCU;
    /**
    * \brief Vector of validation CU optimal split
    * This vector contains ${NB_VALIDATION_TARGETS} elements and is loaded once at training beginning
    */
    static std::vector<uint8_t> *validationTargetsOptimalSplits;
    /**
    * \brief Index of the actual loaded CU for validation
    */
    uint64_t actualValidationCU;

    // ********************************************* CONSTRUCTORS *********************************************
    /**
    * \brief Default constructor for the Binary Environment.
    * \param[in] actions Every different actions for the Agent. In this binary TPG, will always be {0,1}.
    * 0 stands for every actions the agent is not specialized in and 1 for his special action.
    * \param[in] speAct The number of the specialized action (0: NP, 1: QT, 2: BTH, 3:BTV, 4: TTH, 5: TTV)
    * \param[in] nbTrainingTargets number of training targets
    * \param[in] nbGeneTargetChange number of generation before reload the preloaded training target set
    * \param[in] nbValidationTarget number of validation targets
    * \param[in] seed for randomness control
    */
    BinaryDefaultEnv(std::vector<uint64_t> actions, int speAct, const uint64_t nbTrainingElements, const uint64_t nbTrainingTargets, const uint64_t nbGeneTargetChange, const uint64_t nbValidationTarget, size_t seed)
            : LearningEnvironment(NB_ACTIONS),
              rng(seed),
              availableActions(actions),
              specializedAction(speAct),
              score(0.0),
              currentMode(Learn::LearningMode::TRAINING),
              currentCU(32, 32),    // 2D Array
              optimal_split(6),           // Unexisting split
              NB_TRAINING_ELEMENTS(nbTrainingElements),
              NB_TRAINING_TARGETS(nbTrainingTargets),
              NB_GENERATION_BEFORE_TARGETS_CHANGE(nbGeneTargetChange),
              NB_VALIDATION_TARGETS(nbValidationTarget),
              actualTrainingCU(0),
              actualValidationCU(0) {}

    // ********************************************* SPECIAL FUNCTIONS *********************************************

    /**
     * \brief Opens, reads and stores a random CU file in the database
     * CU datas are returned and the corresponding split is stored in the mode vector)
     *
     * \param[in] index the index where the loaded CU will be stored
     * \param[in] mode the LearningMode : store either in validationTargetsOptimalSplits or trainingTargetsOptimalSplits
     * \return a PrimitiveTypeArray2D<uint8_t>* containing loaded CU datas
     */
    Data::PrimitiveTypeArray2D<uint8_t>* getRandomCU(uint64_t index, Learn::LearningMode mode, const char current_CU_path[100]);

    /**
     * \brief Load the next preloaded CU either for training or for validation (depending on the currentMode)
     */
    void LoadNextCU();

    /**
     * \brief Update Training and Validation Targets depending on the generation
     * For the first generation (numGen == 0), load NB_VALIDATION_TARGETS CUs for validation and NB_TRAINING_TARGETS CUs for training
     * Else every NB_GENERATION_BEFORE_TARGETS_CHANGE, delete old training targets and load NB_TRAINING_TARGETS new CUs.
     *
     * \param[in] currentGen The number of the current generation
     */
    void UpdatingTargets(uint64_t currentGen, const char current_CU_path[100]);

    /**
     * \brief Print the classification table of the best root in a .txt file
     *
     * \param[in] env the Environment
     * \param[in] bestRoot the root whose classification table will be printed
     * \param[in] numGen generation number
     * \param[in] outputFile the name of the destination file
     */
    void printClassifStatsTable(const Environment& env, const TPG::TPGVertex* bestRoot, const int numGen, std::string const& outputFile, bool readable);

    /**
     * \brief Return a string corresponding to the name of the action :
     * (0: NP, 1: QT, 2: BTH, 3:BTV, 4: TTH, 5: TTV)
     * \param[in] speAct the number of the action
     */
    std::string getActionName(uint64_t speAct);

    // ********************************************* GETTERS *********************************************
    /**
     * \brief Getter for specializedAction
     */
    int getSpecializedAction() const;
    /**
     * \brief Getter for optimalSplit
     */
    uint8_t getOptimalSplit() const;

    // ********************************************* SETTERS *********************************************
    /**
     * \brief Setter for currentMode
     */
    void setCurrentMode(Learn::LearningMode mode);

    // ********************************************* LearningEnvironment *********************************************
    /**
    * \brief Get a copy of the LearningEnvironment (Default implementation returns a null pointer)
    * \return a copy of the LearningEnvironment if it is copyable, otherwise this method returns a NULL pointer.
    */
    LearningEnvironment *clone() const;

    /**
    * \brief Can the LearningEnvironment be copy in order to evaluate several LearningAgent in parallel ?
    * \return true if the LearningEnvironment can be copied and run in parallel (Default implementation returns false)
    */
    bool isCopyable() const;

    /**
    * \brief Execute an action on the LearningEnvironment
    *
    * The purpose of this method is to execute an action, represented by
    * an actionId comprised between 0 and nbActions - 1.
    * The LearningEnvironment implementation only checks that the given
    * actionID is comprised between 0 and nbActions - 1.
    * It is the responsibility of this method to call the updateHash
    * method on dataSources whose content have been affected by the action.
    *
    * \param[in] actionID the integer number representing the action to execute.
    * \throw std::runtime_error if the actionID exceeds nbActions - 1.
    */
    void doAction(uint64_t actionID);

    /**
    * \brief Reset the LearningEnvironment.
    *
    * Resetting a learning environment is needed to train an agent.
    * Optionally seed can be given to this function to control the
    * randomness of a LearningEnvironment (if any). When available, this
    * feature will be used:
    * - for comparing the performance of several agents with the same
    * random starting conditions.
    * - for training each agent with diverse starting conditions.
    *
    * \param[in] seed Integer value for controlling the randomness of
    * the LearningEnvironment.
    * \param[in] mode LearningMode in which the Environment should be
    * reset for the next set of actions.
    */
    void reset(size_t seed = 0, Learn::LearningMode mode = Learn::TRAINING);

    /**
    * \brief Get the data sources, every pixel of the current CU, for this LearningEnvironment
    *
    * This method returns a vector of reference to the DataHandler that
    * will be given to the learningAgent, and to its Program to learn how
    * to interact with the LearningEnvironment. Throughout the existence
    * of the LearningEnvironment, data contained in the data will be
    * modified, but never the number, nature or size of the dataHandlers.
    * Since this methods return references to the DataHandler, the
    * learningAgent will assume that the referenced dataHandler are
    * automatically updated each time the doAction, or reset methods
    * are called on the LearningEnvironment.
    *
    * \return a vector of references to the DataHandler
    */
    std::vector<std::reference_wrapper<const Data::DataHandler>> getDataSources();

    /**
    * \brief Returns the current score of the Environment
    * The returned score will be used as a reward during the learning phase
    * \return the current score for the LearningEnvironment
    */
    double getScore() const;

    /**
    * \brief Checks if the LearningEnvironment has reached a terminal state
    * \return a boolean indicating termination
    */
    bool isTerminal() const;
};

#endif //TPGVVCPARTDATABASE_DEFAULTBINARYENV_H
