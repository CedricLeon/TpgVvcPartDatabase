#include "../include/PartCU.h"
#include <vector>

// ********************************************************************* //
// ************************** GEGELATI FUNCTIONS *********************** //
// ********************************************************************* //

void PartCU::doAction(uint64_t actionID)
{
    // Call to default method to increment classificationTable
    ClassificationLearningEnvironment::doAction(actionID);

    // Loading next CU
    this->LoadNextCU();

    // Si nécessaire pour debugguer : printf des actions choisies pour les 1ere gen
}

std::vector<std::reference_wrapper<const Data::DataHandler>> PartCU::getDataSources()
{
    // Return a vector containing every element constituting the State of the environment
    std::vector<std::reference_wrapper<const Data::DataHandler>> result{this->currentCU};

    return result;
}

void PartCU::reset(size_t seed, Learn::LearningMode mode)
{
    // Reset the classificationTable
    ClassificationLearningEnvironment::reset(seed);

    this->currentMode = mode;

    // RNG Control : Create seed from seed and mode
    //size_t hash_seed = Data::Hash<size_t>()(seed) ^Data::Hash<Learn::LearningMode>()(mode);
    // Reset the RNG
    this->rng.setSeed(/*hash_seed*/seed);

    // Preload the first CU (depending of the current mode)
    this->LoadNextCU();
}

Learn::LearningEnvironment *PartCU::clone() const
{
    return new PartCU(*this);
}

bool PartCU::isCopyable() const
{
    return true; // false : to avoid ParallelLearning (Cf LearningAgent)
}

double PartCU::getScore() const
{
    return ClassificationLearningEnvironment::getScore();
}

bool PartCU::isTerminal() const
{
    // Return if the job is over
    return false;
}



// ********************************************************************* //
// *************************** PartCU FUNCTIONS ************************ //
// ********************************************************************* //

// ****** TRAINING Arguments ******
std::vector<Data::PrimitiveTypeArray<uint8_t> *> *PartCU::trainingTargetsCU = new std::vector<Data::PrimitiveTypeArray<uint8_t>*>; // Array2DWrapper
std::vector<uint8_t> *PartCU::trainingTargetsOptimalSplits = new std::vector<uint8_t>;
// ****** VALIDATION Arguments ******
std::vector<Data::PrimitiveTypeArray<uint8_t>*> *PartCU::validationTargetsCU = new std::vector<Data::PrimitiveTypeArray<uint8_t>*>;  // Array2DWrapper
std::vector<uint8_t> *PartCU::validationTargetsOptimalSplits = new std::vector<uint8_t>;

Data::PrimitiveTypeArray<uint8_t> *PartCU::getRandomCU(uint64_t index, Learn::LearningMode mode) {
    // ------------------ Opening and Reading a random CU file ------------------
    uint32_t next_CU_number = this->rng.getInt32(0, NB_TRAINING_ELEMENTS - 1);
    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number);
    char current_CU_path[100] = "/home/cleonard/Data/dataset_tpg_balanced/dataset_tpg_32x32_27_balanced2/";
    // "D:/dev/InnovR/dataset_tpg_32x32_27/dataset_tpg_32x32_27/" || "/home/cleonard/Data/dataset_tpg_balanced/dataset_tpg_32x32_27_balanced2/"
    char bin_extension[10] = ".bin";
    std::strcat(current_CU_path, next_CU_number_string);
    std::strcat(current_CU_path, bin_extension);

    // Openning the file
    std::FILE *input = std::fopen(current_CU_path, "r");
    if (!input)
    {
        char error_file_path[300] = "File opening failed : ";
        std::strcat(error_file_path, current_CU_path);
        std::perror(error_file_path);
        return nullptr; // return EXIT_FAILURE;
    }

    // Stocking content in a uint8_t tab, first 32x32 uint8_t are CU's pixels values and the 1025th value is the optimal split
    uint8_t contents[32*32+1];
    size_t nbCharRead = std::fread(&contents[0], 1, 32*32+1, input);
    if (nbCharRead != 32*32+1)
        std::perror("File Read failed");
    // Dunno why it fails

    // Important ...
    std::fclose(input);

    // Creating a new PrimitiveTypeArray<uint8_t> and filling it
    auto *randomCU = new Data::PrimitiveTypeArray<uint8_t>(32 * 32);
    for (uint32_t pxlIndex = 0; pxlIndex < 32 * 32; pxlIndex++)
        randomCU->setDataAt(typeid(uint8_t), pxlIndex, contents[pxlIndex]);

    // Updating the corresponding optimal split depending of the current mode
    if (mode == Learn::LearningMode::TRAINING)
        PartCU::trainingTargetsOptimalSplits->push_back(contents[1024]);
    else if (mode == Learn::LearningMode::VALIDATION)
        PartCU::validationTargetsOptimalSplits->push_back(contents[1024]);

    return randomCU;
}

void PartCU::LoadNextCU() {
    // Checking validity is no longer necessary
    if (this->currentMode == Learn::LearningMode::TRAINING)
    {
        this->currentCU = *PartCU::trainingTargetsCU->at(this->actualTrainingCU);

        // Updating next split solution
        this->currentClass = PartCU::trainingTargetsOptimalSplits->at(this->actualTrainingCU);
        this->actualTrainingCU++;

        // Looping on the beginning of training targets
        if (this->actualTrainingCU >= NB_TRAINING_TARGETS)
            this->actualTrainingCU = 0;
    }
    else if (this->currentMode == Learn::LearningMode::VALIDATION)
    {
        this->currentCU = *PartCU::validationTargetsCU->at(this->actualValidationCU);

        // Updating next split solution
        this->currentClass = PartCU::validationTargetsOptimalSplits->at(this->actualValidationCU);
        this->actualValidationCU++;

        // Looping on the beginning of validation targets
        if (this->actualValidationCU >= NB_VALIDATION_TARGETS)
            this->actualValidationCU = 0;
    }
}

void PartCU::printClassifStatsTable(const Environment& env, const TPG::TPGVertex* bestRoot, const int numGen, std::string const outputFile)
{
    // Print table of classification of the best
    TPG::TPGExecutionEngine tee(env, nullptr);

    // Change the MODE
    this->reset(0, Learn::LearningMode::VALIDATION);    // TESTING in MNIST, mais marche pas ici

    // Fill the table
    const int nbClasses = 6;

    uint64_t classifTable[nbClasses][nbClasses] = { 0 };
    uint64_t nbPerClass[nbClasses] = { 0 };
    uint8_t actionID = -1;

    for (uint64_t nbImage = 0; nbImage < this->NB_VALIDATION_TARGETS; nbImage++)
    {
        // Get answer
        uint64_t optimalActionID = this->currentClass;
        nbPerClass[optimalActionID]++;

        // Execute
        auto path = tee.executeFromRoot(*bestRoot);
        const TPG::TPGAction* action = (const TPG::TPGAction*)path.at(path.size() - 1);
        actionID = (uint8_t) action->getActionID();

        // Increment table
        classifTable[optimalActionID][actionID]++;

        // Do action (to trigger image update)
        this->LoadNextCU();
    }

    // Reset the learning mode to TESTING
    this->reset(0, Learn::LearningMode::TESTING);

    // Computing Score :
    uint64_t score = 0;
    for (int i = 0; i < nbClasses; i++)
        score += classifTable[i][i];

    // Print the table
    std::ofstream fichier(outputFile.c_str(), std::ios::app);
    if(fichier)
    {
        fichier << "-------------------------------------------------" << std::endl;
        fichier << "Gen : " << numGen << "   | Score  : " << score << std::endl;
        fichier << "     NP     QT    BTH    BTV    TTH    TTV    Nb" << std::endl;

        for(int x = 0; x < nbClasses; x++)
        {
            fichier << x;
            for(int y = 0; y < nbClasses; y++)
            {
                uint64_t nb = classifTable[x][y]; //this->classificationTable.at(x).at(y);

                int nbChar = (int) (1 + (nb == 0 ? 0 : log10(nb)));
                for(int nbEspace = 0; nbEspace < (nbClasses - nbChar); nbEspace++)
                    fichier << " ";
                fichier << nb << (x == y ? "-" : " ");
            }
            uint64_t nb = nbPerClass[x];
            int nbChar = (int)(1 + (nb == 0 ? 0 : log10(nb)));
            for (int nbEspace = 0; nbEspace < (nbClasses - nbChar); nbEspace++)
                fichier << " ";
            fichier << nb << std::endl;
        }
        fichier.close();
    }else
    {
        std::cout << "Unable to open the file " << outputFile << "." << std::endl;
    }
}
