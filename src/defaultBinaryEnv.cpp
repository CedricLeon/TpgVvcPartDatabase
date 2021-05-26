#include "../include/defaultBinaryEnv.h"
#include <vector>

// ********************************************************************* //
// ************************** GEGELATI FUNCTIONS *********************** //
// ********************************************************************* //

void BinaryEnv::doAction(uint64_t actionID)
{
    // Managing the reward (+1 if the best split is chosen, else +0)
    if(actionID == this->optimal_split)
    {
        if(actionID == 0)
            this->score += 0.2;
        else if(actionID == 1)
            this->score += 1;
        else
            std::cout << "ComputingScore : Wrong use of binary TPG. ActionID not in action range." << std::endl;
    }

    // Loading next CU
    this->LoadNextCU();
}

std::vector<std::reference_wrapper<const Data::DataHandler>> BinaryEnv::getDataSources()
{
    // Return a vector containing every element constituting the State of the environment
    std::vector<std::reference_wrapper<const Data::DataHandler>> result{this->currentCU};
    return result;
}

void BinaryEnv::reset(size_t seed, Learn::LearningMode mode)
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

Learn::LearningEnvironment *BinaryEnv::clone() const
{
    return new BinaryEnv(*this);
}

bool BinaryEnv::isCopyable() const
{
    return true; // false : to avoid ParallelLearning (Cf. LearningAgent)
}

double BinaryEnv::getScore() const
{
    return this->score;
}

bool BinaryEnv::isTerminal() const
{
    return false;
}


// ********************************************************************* //
// *************************** PartCU FUNCTIONS ************************ //
// ********************************************************************* //

// ****** TRAINING Arguments ******
std::vector<Data::PrimitiveTypeArray2D<uint8_t> *> *BinaryEnv::trainingTargetsCU = new std::vector<Data::PrimitiveTypeArray2D<uint8_t>*>;
std::vector<uint8_t> *BinaryEnv::trainingTargetsOptimalSplits = new std::vector<uint8_t>;
// ****** VALIDATION Arguments ******
std::vector<Data::PrimitiveTypeArray2D<uint8_t>*> *BinaryEnv::validationTargetsCU = new std::vector<Data::PrimitiveTypeArray2D<uint8_t>*>;
std::vector<uint8_t> *BinaryEnv::validationTargetsOptimalSplits = new std::vector<uint8_t>;

Data::PrimitiveTypeArray2D<uint8_t> *BinaryEnv::getRandomCU(uint64_t index, Learn::LearningMode mode)
{
    // ------------------ Opening and Reading a random CU file ------------------
    // Generate the path for a random CU
    uint32_t next_CU_number = this->rng.getInt32(0, NB_TRAINING_ELEMENTS - 1);
    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number);
    char current_CU_path[100] = "/home/cleonard/Data/dataset_tpg_balanced/dataset_tpg_32x32_27_balanced2/";
    char bin_extension[10] = ".bin";
    std::strcat(current_CU_path, next_CU_number_string);
    std::strcat(current_CU_path, bin_extension);

    // Opening the file
    std::FILE *input = std::fopen(current_CU_path, "r");
    if (!input)
    {
        char error_file_path[300] = "File opening failed : ";
        std::strcat(error_file_path, current_CU_path);
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
        BinaryEnv::trainingTargetsOptimalSplits->push_back(contents[1024]);
    else if (mode == Learn::LearningMode::VALIDATION)
        BinaryEnv::validationTargetsOptimalSplits->push_back(contents[1024]);

    return randomCU;
}

void BinaryEnv::LoadNextCU()
{
    // Checking validity is no longer necessary
    if (this->currentMode == Learn::LearningMode::TRAINING)
    {
        this->currentCU = *BinaryEnv::trainingTargetsCU->at(this->actualTrainingCU);

        // Updating next split solution :
        //   -  If the optimal split is not the specialized action, the optimal split is set to 0
        //   -  Else the optimal split is set to 1
        this->optimal_split = BinaryEnv::trainingTargetsOptimalSplits->at(this->actualTrainingCU) != this->specializedAction ? 0 : 1;
        this->actualTrainingCU++;

        // Looping on the beginning of training targets
        if (this->actualTrainingCU >= NB_TRAINING_TARGETS)
            this->actualTrainingCU = 0;
    }
    else if (this->currentMode == Learn::LearningMode::VALIDATION)
    {
        this->currentCU = *BinaryEnv::validationTargetsCU->at(this->actualValidationCU);

        // Updating next split solution :
        //   -  If the optimal split is not the specialized action, the optimal split is set to 0
        //   -  Else the optimal split is set to 1
        this->optimal_split = BinaryEnv::validationTargetsOptimalSplits->at(this->actualValidationCU) != this->specializedAction ? 0 : 1;
        this->actualValidationCU++;

        // Looping on the beginning of validation targets
        if (this->actualValidationCU >= NB_VALIDATION_TARGETS)
            this->actualValidationCU = 0;
    }
}

void BinaryEnv::printClassifStatsTable(const Environment& env, const TPG::TPGVertex* bestRoot, const int numGen, std::string const& outputFile)
{
    // Create a new TPGExecutionEngine from the environment
    TPG::TPGExecutionEngine tee(env, nullptr);

    // Change the LearningMode in VALIDATION
    this->reset(0, Learn::LearningMode::VALIDATION);

    // Fill the table
    const int nbClasses = 2;

    uint64_t classifTable[nbClasses][nbClasses] = { 0 };
    uint64_t nbPerClass[nbClasses] = { 0 };
    uint8_t actionID;

    for (uint64_t nbImage = 0; nbImage < this->NB_VALIDATION_TARGETS; nbImage++)
    {
        // Get answer
        uint64_t optimalActionID = this->optimal_split;
        nbPerClass[optimalActionID]++;

        // Execute
        auto path = tee.executeFromRoot(*bestRoot);
        const auto* action = (const TPG::TPGAction*)path.at(path.size() - 1);
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
    validationScore += ((double) classifTable[0][0]) / 5;
    validationScore += (double) classifTable[1][1];
    // Computing ScoreMax :
    double scoreMax = 0.0;
    scoreMax += ((double) nbPerClass[0]) / 5;
    scoreMax += (double) nbPerClass[1];

    // What is the specialized action ?
    std::string speActionName("???");
    switch(this->specializedAction)
    {
        case 0 :
            speActionName = "NP ";
            break;
        case 1 :
            speActionName = "QT ";
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

    // Print the table
    std::ofstream file(outputFile.c_str(), std::ios::app);
    if(file)
    {
        file << "-------------------------------------------------" << std::endl;
        file << "Gen: " << numGen << "   | Score: " << std::setprecision(4) << validationScore/scoreMax*100 << "  (" << validationScore << "/" << scoreMax << ")" << std::endl;
        file << "  OTHER    " << speActionName << "  Nb  |     OTHER      " << speActionName << "    Nb" << std::endl;

        for(int x = 0; x < nbClasses; x++)
        {
            // ----------  Number of CUs ----------
            // Print real class number
            file << x << " ";
            // Print number of guessed instances for each class
            for(int y = 0; y < nbClasses; y++)
            {
                uint64_t nb = classifTable[x][y];
                file << "  " << std::setw(3) << nb << " ";
            }
            // Print total number of class instances
            uint64_t nb = nbPerClass[x];
            file << std::setw(4) << nb;

            // ----------  Normalized ----------
            file << "  |  " << x;
            for(int y = 0; y < nbClasses; y++)
            {
                double nbNorm = (double)classifTable[x][y] / (double)nb * 100;
                file << "  " << std::setw(5) << nbNorm << " ";
            }
            file << std::setw(6) << nb << std::endl;
        }
        file.close();
    }else
    {
        std::cout << "Unable to open the file " << outputFile << "." << std::endl;
    }
}
