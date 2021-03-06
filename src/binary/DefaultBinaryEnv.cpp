#include <vector>
#include <iomanip>

#include "../../include/binary/DefaultBinaryEnv.h"

// ********************************************************************* //
// ************************** GEGELATI FUNCTIONS *********************** //
// ********************************************************************* //

void BinaryDefaultEnv::doAction(uint64_t actionID)
{
    // Managing the reward (+1 if the best split is chosen, else +0)
    if(actionID == this->optimal_split)
        this->score++;

    /*if(actionID == this->optimal_split)
    {
        if(actionID == 0)
            this->score += 0.2;
        else if(actionID == 1)
            this->score += 1;
        else
            std::cout << "ComputingScore : Wrong use of binary TPG. ActionID not in action range." << std::endl;
    }*/

    // Loading next CU
    this->LoadNextCU();
}

std::vector<std::reference_wrapper<const Data::DataHandler>> BinaryDefaultEnv::getDataSources()
{
    // Return a vector containing every element constituting the State of the environment
    std::vector<std::reference_wrapper<const Data::DataHandler>> result{this->currentCU};
    return result;
}

void BinaryDefaultEnv::reset(size_t seed, Learn::LearningMode mode)
{
    // Update the LearningMode
    this->currentMode = mode;

    // Reset the score
    this->score = 0.0;

    // RNG Control : Create seed from seed and mode
    size_t hash_seed = Data::Hash<size_t>()(seed) ^Data::Hash<Learn::LearningMode>()(mode);
    // Reset the RNG
    this->rng.setSeed(hash_seed);

    // Preload the first CU (depending on the current mode)
    this->LoadNextCU();
}

Learn::LearningEnvironment *BinaryDefaultEnv::clone() const
{
    return new BinaryDefaultEnv(*this);
}

bool BinaryDefaultEnv::isCopyable() const
{
    return true; // false : to avoid ParallelLearning (Cf. LearningAgent)
}

double BinaryDefaultEnv::getScore() const
{
    return this->score;
}

bool BinaryDefaultEnv::isTerminal() const
{
    return false;
}


// ********************************************************************* //
// *************************** PartCU FUNCTIONS ************************ //
// ********************************************************************* //

// ****** TRAINING Arguments ******
std::vector<Data::PrimitiveTypeArray2D<uint8_t> *> *BinaryDefaultEnv::trainingTargetsCU = new std::vector<Data::PrimitiveTypeArray2D<uint8_t>*>;
std::vector<uint8_t> *BinaryDefaultEnv::trainingTargetsOptimalSplits = new std::vector<uint8_t>;
// ****** VALIDATION Arguments ******
std::vector<Data::PrimitiveTypeArray2D<uint8_t>*> *BinaryDefaultEnv::validationTargetsCU = new std::vector<Data::PrimitiveTypeArray2D<uint8_t>*>;
std::vector<uint8_t> *BinaryDefaultEnv::validationTargetsOptimalSplits = new std::vector<uint8_t>;

Data::PrimitiveTypeArray2D<uint8_t> *BinaryDefaultEnv::getRandomCU(Learn::LearningMode mode, const char current_CU_path[100])
{
    // ------------------ Opening and Reading a random CU file ------------------
    // Generate the path for a random CU
    uint32_t next_CU_number = this->rng.getInt32(0, (int) NB_TRAINING_ELEMENTS - 1);
    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number);
    char CU_path[100];
    std::strcpy(CU_path, current_CU_path);
    char bin_extension[10] = ".bin";
    std::strcat(CU_path, next_CU_number_string);
    std::strcat(CU_path, bin_extension);

    // Opening the file
    std::FILE *input = std::fopen(CU_path, "r");
    if (!input)
    {
        char error_file_path[300] = "File opening failed : ";
        std::strcat(error_file_path, CU_path);
        std::perror(error_file_path);
        return nullptr; // return EXIT_FAILURE;
    }

    // Stocking content in a uint8_t tab, first 32x32 uint8_t are CU pixels values and the 1025th value is the optimal split
    uint8_t contents[32*32+1];
    size_t nbCharRead = std::fread(&contents[0], 1, 32*32+1, input);
    if (nbCharRead != 32*32+1)
        std::perror("File Read failed");

    // Important ...
    std::fclose(input);

    // Creating a new PrimitiveTypeArray<uint8_t> and filling it
    auto *randomCU = new Data::PrimitiveTypeArray2D<uint8_t>(32, 32);   // 2D Array
    for (uint32_t pxlIndex = 0; pxlIndex < 32 * 32; pxlIndex++)
        randomCU->setDataAt(typeid(uint8_t), pxlIndex, contents[pxlIndex]);

    // Updating the corresponding optimal split depending of the current mode
    if (mode == Learn::LearningMode::TRAINING)
        BinaryDefaultEnv::trainingTargetsOptimalSplits->push_back(contents[1024]);
    else if (mode == Learn::LearningMode::VALIDATION)
        BinaryDefaultEnv::validationTargetsOptimalSplits->push_back(contents[1024]);

    return randomCU;
}

void BinaryDefaultEnv::UpdatingTargets(uint64_t currentGen, const char current_CU_path[100])
{
    // Each ${nbGeneTargetChange} generation, generate new random training targets so that different targets are used
    if (currentGen % NB_GENERATION_BEFORE_TARGETS_CHANGE == 0)
    {
        // ---  Deleting old targets ---
        if (currentGen != 0) // Don't clear trainingTargets before initializing them
        {
            this->reset(0, Learn::LearningMode::TRAINING);
            for (uint64_t idx_targ = 0; idx_targ < NB_TRAINING_TARGETS; idx_targ++)
                delete BinaryDefaultEnv::trainingTargetsCU->at(idx_targ);   // targets are allocated in getRandomCU()
            BinaryDefaultEnv::trainingTargetsCU->clear();
            BinaryDefaultEnv::trainingTargetsOptimalSplits->clear();
            this->actualTrainingCU = 0;
        }
        else        // Load VALIDATION Targets at the beginning of the training (i == 0)
        {
            for (uint64_t idx_targ = 0; idx_targ < NB_VALIDATION_TARGETS; idx_targ++)
            {
                Data::PrimitiveTypeArray2D<uint8_t>* target = this->getRandomCU(Learn::LearningMode::VALIDATION, current_CU_path);
                BinaryDefaultEnv::validationTargetsCU->push_back(target);
            }
        }

        // ---  Loading next targets ---
        for (uint64_t idx_targ = 0; idx_targ < NB_TRAINING_TARGETS; idx_targ++)
        {
            Data::PrimitiveTypeArray2D<uint8_t>* target = this->getRandomCU(Learn::LearningMode::TRAINING, current_CU_path);
            BinaryDefaultEnv::trainingTargetsCU->push_back(target);
            // Optimal split is saved in LE->trainingTargetsOptimalSplits inside getRandomCU()
        }
    }
}

void BinaryDefaultEnv::LoadNextCU()
{
    // Checking validity is no longer necessary
    if (this->currentMode == Learn::LearningMode::TRAINING)
    {
        this->currentCU = *BinaryDefaultEnv::trainingTargetsCU->at(this->actualTrainingCU);

        // Updating next split solution :
        //   -  If the optimal split is not the specialized action, the optimal split is set to 0
        //   -  Else the optimal split is set to 1
        this->optimal_split = BinaryDefaultEnv::trainingTargetsOptimalSplits->at(this->actualTrainingCU) != this->specializedAction ? 0 : 1;
        this->actualTrainingCU++;

        // Looping on the beginning of training targets
        if (this->actualTrainingCU >= NB_TRAINING_TARGETS)
            this->actualTrainingCU = 0;
    }
    else if (this->currentMode == Learn::LearningMode::VALIDATION)
    {
        this->currentCU = *BinaryDefaultEnv::validationTargetsCU->at(this->actualValidationCU);

        // Updating next split solution :
        //   -  If the optimal split is not the specialized action, the optimal split is set to 0
        //   -  Else the optimal split is set to 1
        this->optimal_split = BinaryDefaultEnv::validationTargetsOptimalSplits->at(this->actualValidationCU) != this->specializedAction ? 0 : 1;
        this->actualValidationCU++;

        // Looping on the beginning of validation targets
        if (this->actualValidationCU >= NB_VALIDATION_TARGETS)
            this->actualValidationCU = 0;
    }
}

void BinaryDefaultEnv::printClassifStatsTable(const Environment& env, const TPG::TPGVertex* bestRoot, const int numGen, std::string const& outputFile, bool readable)
{
    // Create a new TPGExecutionEngine from the environment
    TPG::TPGExecutionEngine tee(env, nullptr);

    // Change the LearningMode in VALIDATION
    this->reset(0, Learn::LearningMode::VALIDATION);

    // Fill the table
    const int nbClasses = 2;

    uint64_t classifTable[nbClasses][nbClasses] = {{0}};
    uint64_t nbPerClass[nbClasses] = {0};
    uint8_t actionID;

    for (uint64_t nbImage = 0; nbImage < this->NB_VALIDATION_TARGETS; nbImage++)
    {
        // Get answer
        uint64_t optimalActionID = this->optimal_split;
        nbPerClass[optimalActionID]++;

        // std::cout << optimalActionID << " : " << nbPerClass[optimalActionID] << std::endl;

        // Execute
        auto path = tee.executeFromRoot(*bestRoot);
        const auto *action = (const TPG::TPGAction *) path.at(path.size() - 1);
        actionID = (uint8_t) action->getActionID();

        // Increment table
        classifTable[optimalActionID][actionID]++;

        // Do action in order to trigger image update and load the next CU
        this->LoadNextCU();
    }

    // Reset the learning mode to TESTING
    this->reset(0, Learn::LearningMode::TESTING);

    // Computing Score
    double validationScore = 0.0;
    validationScore += ((double) classifTable[0][0]);
    validationScore += (double) classifTable[1][1];
    // Computing ScoreMax :
    double scoreMax = 0.0;
    scoreMax += ((double) nbPerClass[0]);
    scoreMax += (double) nbPerClass[1];

    // What is the specialized action ?
    std::string speActionName = this->getActionName(this->specializedAction);
    if (this->specializedAction <= 1)
        speActionName += " ";

    // If readable confusion matrix is needed
    if(readable)
    {
        // Print the table
        std::ofstream file(outputFile.c_str(), std::ios::app);
        if (file)
        {
            file << "-------------------------------------------------" << std::endl;
            file << "Gen: " << numGen << " | Score: " << std::setprecision(4) << validationScore / scoreMax * 100 << "  ("
                 << validationScore << "/" << scoreMax << ")" << std::endl;
            file << "  OTHER    " << speActionName << "  Nb  |     OTHER      " << speActionName << "    Nb" << std::endl;

            for (int x = 0; x < nbClasses; x++)
            {
                // ----------  Number of CUs ----------
                // Print real class number
                file << x << " ";
                // Print number of guessed instances for each class
                for (int y = 0; y < nbClasses; y++)
                {
                    uint64_t nb = classifTable[x][y];
                    file << "  " << std::setw(3) << nb << " ";
                }
                // Print total number of class instances
                uint64_t nb = nbPerClass[x];
                file << std::setw(4) << nb;

                // ----------  Normalized ----------
                file << "  |  " << x;
                for (int y = 0; y < nbClasses; y++)
                {
                    double nbNorm = (double) classifTable[x][y] / (double) nb * 100;
                    file << "  " << std::setw(5) << nbNorm << " ";
                }
                file << std::setw(6) << nb << std::endl;
            }
            file.close();
        } else {
            std::cout << "Unable to open the file " << outputFile << "." << std::endl;
        }
    }else  // If non-readable confusion matrix is asked (re-usable data)
    {
        // Print the table
        std::ofstream file(outputFile.c_str(), std::ios::app);
        if (file)
        {
            if(numGen == 0)
            {
                file << speActionName << " training, maxScore = " << scoreMax << std::endl;
                file << nbPerClass[1] << " " << speActionName << " CUs and " << nbPerClass[0] << " in OTHER class" << std::endl << std::endl;
                file << " Gen   OTHER    " << speActionName << "    TOT" << std::endl;
            }
            double otherNorm = (double) classifTable[0][0] / (double) nbPerClass[0] * 100;
            double actNorm   = (double) classifTable[1][1] / (double) nbPerClass[1] * 100;
            file << std::setw(4) << numGen << "   ";
            file << std::fixed << std::setprecision(2) << otherNorm << "  ";
            file << actNorm << "  ";
            file << validationScore / scoreMax * 100 << std::endl;
            file.close();
        } else
        {
            std::cout << "Unable to open the file " << outputFile << "." << std::endl;
        }
    }
}

std::string BinaryDefaultEnv::getActionName(uint64_t speAct)
{
    std::string speActionName("???");
    switch(speAct)
    {
        case 0 :
            speActionName = "NP";
            break;
        case 1 :
            speActionName = "QT";
            break;
        case 2 :
            speActionName = "BTH";
            break;
        case 3 :
            speActionName = "BTV";
            break;
        case 4 :
            speActionName = "TTH";
            break;
        case 5 :
            speActionName = "TTV";
            break;
        default:
            speActionName = "WTF";
            break;
    }
    return speActionName;
}

int BinaryDefaultEnv::getSpecializedAction() const { return specializedAction; }
uint8_t BinaryDefaultEnv::getOptimalSplit()  const { return optimal_split; }
Mutator::RNG BinaryDefaultEnv::getRng() const { return rng; }

void BinaryDefaultEnv::setCurrentMode(Learn::LearningMode mode) { BinaryDefaultEnv::currentMode = mode; }
void BinaryDefaultEnv::setCurrentCu(const Data::PrimitiveTypeArray2D<uint8_t> &currentCu) { currentCU = currentCu; }
