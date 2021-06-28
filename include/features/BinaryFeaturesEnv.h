#ifndef TPGVVCPARTDATABASE_BINARYFEATURESENV_H
#define TPGVVCPARTDATABASE_BINARYFEATURESENV_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

#include <gegelati.h>

/**
* \brief Heritage of the ClassificationLearningEnvironment Interface
* This class defines the environment for a binary TPG interacting with a database of CU Features (custom size)
*/
class BinaryFeaturesEnv : public Learn::ClassificationLearningEnvironment {

private:

    // *********************************************** Actions arguments ***********************************************
    /// Number of different actions for the Agent
    static const uint8_t NB_ACTIONS = 2;
    /// List of split corresponding to the action 0 (generaly only one split, except for vertical / horizontal)
    std::vector<uint8_t> actions0;
    /// List of split corresponding to the action 1
    std::vector<uint8_t> actions1;

    // ***************************************** Data and database arguments *******************************************
    /// Height of the partitioned Coding Unit
    const uint64_t CU_HEIGHT;
    /// Width of the partitioned Coding Unit
    const uint64_t CU_WIDTH;
    /// Width of the data in the CSV file
    const uint64_t NB_FEATURES;

    /// Randomness control
    Mutator::RNG rng;
    /// Seed for randomness control
    size_t seed;

    /// Current LearningMode of the LearningEnvironment (either TRAINING, either VALIDATION). The set of preloaded targets depends on this mode
    Learn::LearningMode currentMode;

    /**
    * \brief Current State of the environment
    * Vector containing all the features computed from CU pixels by CNN
    * CU are NB_FEATURES + 1 (for the QP value) size (=> for 32x32 CUs: 113 values)
    */
    Data::PrimitiveTypeArray<double> currentState;

public:
    // ********************************************* Intern Variables *********************************************
    /// Total number of elements in the database. Elements from the database are picked from 0 to NB_DATABASE_ELEMENTS-1
    const uint64_t NB_DATABASE_ELEMENTS;

    /// Number of targets used for TRAINING
    const uint64_t NB_TRAINING_TARGETS;
    /// Number of targets used for VALIDATION
    const uint64_t NB_VALIDATION_TARGETS;
    /// Number of generations between reloads of training set (In order to accelerate the Learning we preload targets which are used for X generations)
    const uint64_t NB_GENERATION_BEFORE_TARGETS_CHANGE;

    // ********************************************* TRAINING Arguments *********************************************
    /**
    * \brief Vector containing TRAINING targets data
    * This vector contains NB_TRAINING_TARGETS elements and is updated every NB_GENERATION_BEFORE_TARGETS_CHANGE
    * Elements in this vector are accessed iteratively from 0 to NB_TRAINING_TARGETS (then loop to 0)
    * The index actualTrainingCU keeps track of the current one.
    */
    static std::vector<Data::PrimitiveTypeArray<double> *> *trainingTargetsData;
    /**
    * \brief Vector containing TRAINING targets optimal split (associated to the corresponding target in trainingTargetsData)
    * This vector contains NB_TRAINING_TARGETS elements and is updated every NB_GENERATION_BEFORE_TARGETS_CHANGE
    */
    static std::vector<uint8_t> *trainingTargetsSplits;
    /// Index of the actual loaded training target
    uint64_t actualTrainingCU;

    // ********************************************* VALIDATION Arguments *********************************************
    /**
    * \brief Vector containing VALIDATION targets data
    * This vector contains NB_VALIDATION_TARGETS elements and is loaded once at training beginning
    * Elements in this vector are accessed iteratively from 0 to NB_VALIDATION_TARGETS (then loop to 0)
    * The index actualValidationCU keeps track of the current one.
    */
    static std::vector<Data::PrimitiveTypeArray<double>*> *validationTargetsData;
    /**
    * \brief Vector containing VALIDATION targets optimal split (associated to the corresponding target in validationTargetsData)
    * This vector contains NB_VALIDATION_TARGETS elements and is loaded once at training beginning
    */
    static std::vector<uint8_t> *validationTargetsSplits;
    /// Index of the actual loaded validation target
    uint64_t actualValidationCU;

    // ********************************************* CONSTRUCTORS *********************************************
    /**
    * \brief Default constructor for the Features Environment.
    * \param[in] actions0 Vector of every split corresponding to the action 0 for this binary Agent. Generally unique (ex: {0}, binary TPG specialized in the NP split)
    * \param[in] actions1 Vector of every split corresponding to the action 1 for this binary Agent. The complementary set of actions0 for {0,1,2,3,4,5}
    * \param[in] seed for randomness control
    * \param[in] cuHeight height of the CUs used for training (allow to switch database)
    * \param[in] cuWidth width of the CUs used for training (allow to switch database)
    * \param[in] nbFeatures width of the CUs used for training (allow to switch database)
    * \param[in] nbTrainingElements number of usable elements in the database
    * \param[in] nbTrainingTargets number of training elements
    * \param[in] nbGeneTargetChange number of generation before reloading the preloaded training target set
    * \param[in] nbValidationTarget number of validation targets
    */
    BinaryFeaturesEnv(std::vector<uint8_t> actions0, std::vector<uint8_t> actions1, size_t seed,
                      const uint64_t cuHeight, const uint64_t cuWidth, const uint64_t nbFeatures,
                      const uint64_t nbTrainingElements, const uint64_t nbTrainingTargets,
                      const uint64_t nbGeneTargetChange, const uint64_t nbValidationTarget)
            : ClassificationLearningEnvironment(NB_ACTIONS),
              actions0(actions0),
              actions1(actions1),
              CU_HEIGHT(cuHeight),
              CU_WIDTH(cuWidth),
              NB_FEATURES(nbFeatures),
              rng(seed),
              seed(seed),
              currentMode(Learn::LearningMode::TRAINING),
              currentState(NB_FEATURES + 1),
              NB_DATABASE_ELEMENTS(nbTrainingElements),
              NB_TRAINING_TARGETS(nbTrainingTargets),
              NB_VALIDATION_TARGETS(nbValidationTarget),
              NB_GENERATION_BEFORE_TARGETS_CHANGE(nbGeneTargetChange),
              actualTrainingCU(0),
              actualValidationCU(0) {}


    // *************************************************** GETTERS *****************************************************
    uint64_t getCuHeight() const;
    uint64_t getCuWidth() const;
    const uint64_t getNbFeatures() const;
    const std::vector<uint8_t> &getActions0() const;
    const std::vector<uint8_t> &getActions1() const;
    // *************************************************** SETTERS *****************************************************
    void setCurrentState(const Data::PrimitiveTypeArray<double> &currentState);

    // *********************************************** SPECIAL FUNCTIONS ***********************************************

    /**
     * \brief Opens, reads and stores a random CSV file in the database
     * Each CSV file contains 10 CU features (and their correspond optimal split)
     * CU features stored in trainingTargetsCUFeatures or validationTargetsCUFeatures and the corresponding split is stored in the mode vector
     *
     * \param[in] mode The LearningMode : store either in validationTargetsOptimalSplits or trainingTargetsOptimalSplits
     * \param[in] databasePath The path of the database
     */
    void getRandomCUFeaturesFromCSVFile(Learn::LearningMode mode, const std::string& databasePath);

    /// Load the next preloaded features either for training or for validation (depending on the currentMode)
    void LoadNextCUFeatures();

    /**
     * \brief Update Training and Validation Targets depending on the generation
     * For the first generation (numGen == 0), load NB_VALIDATION_TARGETS CU features for validation and NB_TRAINING_TARGETS CU features for training
     * Else every NB_GENERATION_BEFORE_TARGETS_CHANGE, delete old training targets and load NB_TRAINING_TARGETS new CU features.
     *
     * \param[in] currentGen The number of the current generation
     * \param[in] current_CU_path The path of the database
     */
    void UpdateTargets(uint64_t currentGen, const std::string& databasePath);

    void updateCurrentClass(uint8_t optimalSplit);

    /**
     * \brief Print the classification table of the best root in a .txt file
     *
     * \param[in] env the Environment
     * \param[in] bestRoot the root whose classification table will be printed
     * \param[in] numGen generation number
     * \param[in] outputFile the name of the destination file
     */
    void printClassifStatsTable(const Environment& env, const TPG::TPGVertex* bestRoot, const int numGen, const std::string& outputFile, bool readable);

    /**
     * \brief Return a uint8_t corresponding to the number of the action :
     * (0: NP or NS, 1: QT, 2: BTH, 3:BTV, 4: TTH, 5: TTV) else return 6 (error)
     * \param[in] speAct the name of the action (std::string)
     */
    uint8_t getSplitNumber(const std::string& split);

    /**
     * \brief Return a std::string corresponding to the name of the action :
     * (0: NP or NS, 1: QT, 2: BTH, 3:BTV, 4: TTH, 5: TTV) else return 6 (error)
     * \param[in] speAct the number of the action (uint64_t)
     */
    std::string getActionName(uint64_t speAct);

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

#endif //TPGVVCPARTDATABASE_BINARYFEATURESENV_H
