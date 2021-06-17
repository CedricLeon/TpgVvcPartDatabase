#ifndef TPGVVCPARTDATABASE_FEATURESENV_H
#define TPGVVCPARTDATABASE_FEATURESENV_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

#include <gegelati.h>

/**
* \brief Heritage of the LearningEnvironment Interface
* This class defines the environment for TPG interacting with a database
*/
class FeaturesEnv : public Learn::ClassificationLearningEnvironment {

private:
    // const uint8_t NB_ACTIONS; // Unused but 6

    /**
    * \brief Randomness control
    **/
    Mutator::RNG rng;

    /**
    * \brief Current LearningMode of the LearningEnvironment.
    * Either TRAINING, either VALIDATION, switch set of preloaded targets
    * (TRAINING : trainingTargetsCUFeatures, VALIDATION : validationTargetsCUFeatures)
    */
    Learn::LearningMode currentMode;

    /// Width of the data in the CSV file
    static const uint64_t CSV_FILE_WIDTH = 112;

    /**
    * \brief Current State of the environment
    * Vector containing all the probability computed from CU pixels by CNN
    * CU are CSV_FILE_WIDTH + 1 (QP) size => 113 values
    */
    Data::PrimitiveTypeArray<double> currentState;

public:
    // ********************************************* Intern Variables *********************************************
    /**
    * \brief Number of elements in the database
    **/
    const uint32_t NB_TRAINING_ELEMENTS;

    /**
     * \brief Number of TRAINING target
     */
    const uint64_t NB_TRAINING_TARGETS;
    /**
    * \brief Number of VALIDATION target
    */
    const uint64_t NB_VALIDATION_TARGETS;
    /**
     * \brief Number of generation between training set reloads
     * (In order to accelerate the Learning we preload targets (CUs) which are used for X generations)
     */
    const uint64_t NB_GENERATION_BEFORE_TARGETS_CHANGE;

    // ********************************************* TRAINING Arguments *********************************************
    /**
    * \brief Vector of CU datas
    * This vector contains ${NB_TRAINING_TARGETS} elements and is updated every ${NB_GENERATION_BEFORE_TARGETS_CHANGE}
    */
    static std::vector<Data::PrimitiveTypeArray<double> *> *trainingTargetsCUFeatures;
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
    static std::vector<Data::PrimitiveTypeArray<double>*> *validationTargetsCUFeatures;
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
    * \brief Default constructor for the Features Environment.
    * \param[in] actions Every different actions for the Agent. In this TPG, will always be {0,1,2,3,4,5}.
    * \param[in] nbTrainingTargets number of training targets
    * \param[in] nbGeneTargetChange number of generation before reload the preloaded training target set
    * \param[in] nbValidationTarget number of validation targets
    * \param[in] seed for randomness control
    */
    FeaturesEnv(std::vector<uint64_t> actions, const uint64_t nbTrainingElements, const uint64_t nbTrainingTargets, const uint64_t nbGeneTargetChange, const uint64_t nbValidationTarget, size_t seed)
            : ClassificationLearningEnvironment(actions.size()),
              rng(seed),
              currentMode(Learn::LearningMode::TRAINING),
              currentState(CSV_FILE_WIDTH + 1),
              NB_TRAINING_ELEMENTS(nbTrainingElements),
              NB_TRAINING_TARGETS(nbTrainingTargets),
              NB_VALIDATION_TARGETS(nbValidationTarget),
              NB_GENERATION_BEFORE_TARGETS_CHANGE(nbGeneTargetChange),
              actualTrainingCU(0),
              actualValidationCU(0) {}

    // ********************************************* SPECIAL FUNCTIONS *********************************************

    /**
     * \brief Opens, reads and stores a random CSV file in the database
     * Each CSV file contains 10 CU features (and their correspond optimal split)
     * CU features stored in trainingTargetsCUFeatures or validationTargetsCUFeatures and the corresponding split is stored in the mode vector
     *
     * \param[in] mode The LearningMode : store either in validationTargetsOptimalSplits or trainingTargetsOptimalSplits
     * \param[in] current_CU_path The path of the database
     * \return a PrimitiveTypeArray<double>* containing loaded CU features
     */
    void getRandomCUFeaturesFromOriginalCSVFile(Learn::LearningMode mode, const char current_CU_path[100]);

    /**
     * \brief Opens, reads and stores a random CSV file in the database
     * Each CSV file contains 10 CU features (and their correspond optimal split)
     * CU features stored in trainingTargetsCUFeatures or validationTargetsCUFeatures and the corresponding split is stored in the mode vector
     *
     * \param[in] mode The LearningMode : store either in validationTargetsOptimalSplits or trainingTargetsOptimalSplits
     * \param[in] current_CU_path The path of the database
     * \return a PrimitiveTypeArray<double>* containing loaded CU features
     */
    void getRandomCUFeaturesFromSimpleCSVFile(Learn::LearningMode mode, const char current_CU_path[100]);

    /**
     * \brief Load the next preloaded features either for training or for validation (depending on the currentMode)
     */
    void LoadNextCUFeatures();

    /**
     * \brief Update Training and Validation Targets depending on the generation
     * For the first generation (numGen == 0), load NB_VALIDATION_TARGETS CU features for validation and NB_TRAINING_TARGETS CU features for training
     * Else every NB_GENERATION_BEFORE_TARGETS_CHANGE, delete old training targets and load NB_TRAINING_TARGETS new CU features.
     *
     * \param[in] currentGen The number of the current generation
     * \param[in] current_CU_path The path of the database
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
     * \brief Return a uint8_t corresponding to number of the action :
     * (0: NP or NS, 1: QT, 2: BTH, 3:BTV, 4: TTH, 5: TTV) else return 6 (error)
     * \param[in] speAct the name of the action (std::string)
     */
    uint8_t getSplitNumber(const std::string& split);

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

#endif //TPGVVCPARTDATABASE_FEATURESENV_H
