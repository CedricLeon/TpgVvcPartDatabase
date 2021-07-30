#include "../../include/classification/ClassEnv.h"

// ********************************************************************* //
// ************************** GEGELATI FUNCTIONS *********************** //
// ********************************************************************* //

void ClassEnv::doAction(uint64_t actionID)
{
    // Call to default method to increment classificationTable
    ClassificationLearningEnvironment::doAction(actionID);

    // Loading next CU
    this->LoadNextCU();

    // Si nécessaire pour debugguer : printf des actions choisies pour les 1ere gen
}

std::vector<std::reference_wrapper<const Data::DataHandler>> ClassEnv::getDataSources()
{
    // Return a vector containing every element constituting the State of the environment
    std::vector<std::reference_wrapper<const Data::DataHandler>> result{this->currentCU};

    return result;
}

void ClassEnv::reset(size_t seed, Learn::LearningMode mode)
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

Learn::LearningEnvironment *ClassEnv::clone() const
{
    return new ClassEnv(*this);
}

bool ClassEnv::isCopyable() const
{
    return true; // false : to avoid ParallelLearning (Cf LearningAgent)
}

double ClassEnv::getScore() const
{
    return ClassificationLearningEnvironment::getScore();
}

bool ClassEnv::isTerminal() const
{
    // Return if the job is over
    return false;
}



// ********************************************************************* //
// *************************** ClassEnv FUNCTIONS ************************ //
// ********************************************************************* //

// ****** TRAINING Arguments ******
std::vector<Data::PrimitiveTypeArray2D<uint8_t> *> *ClassEnv::trainingTargetsCU = new std::vector<Data::PrimitiveTypeArray2D<uint8_t>*>; // Array2DWrapper
std::vector<uint8_t> *ClassEnv::trainingTargetsOptimalSplits = new std::vector<uint8_t>;
// ****** VALIDATION Arguments ******
std::vector<Data::PrimitiveTypeArray2D<uint8_t>*> *ClassEnv::validationTargetsCU = new std::vector<Data::PrimitiveTypeArray2D<uint8_t>*>;  // Array2DWrapper
std::vector<uint8_t> *ClassEnv::validationTargetsOptimalSplits = new std::vector<uint8_t>;

void ClassEnv::getRandomCU(Learn::LearningMode mode, const std::string& databasePath)
{
    // ------------------ Opening and Reading a random CU file ------------------
    char datasetPath[100] = "/home/cleonard/Data/CU/CU_32x32_balanced/";
    uint32_t next_CU_number = this->rng.getInt32(0, NB_TRAINING_ELEMENTS - 1);
    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number);
    //char current_CU_path[100];
    //std::strcat(current_CU_path, datasetPath.c_str());
    char bin_extension[10] = ".bin";
    std::strcat(datasetPath, next_CU_number_string);
    std::strcat(datasetPath, bin_extension);

    // Openning the file
    std::FILE *input = std::fopen(datasetPath, "r");
    if (!input)
    {
        char error_file_path[300] = "File opening failed: ";
        std::strcat(error_file_path, datasetPath);
        std::perror(error_file_path);
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
    auto *randomCU = new Data::PrimitiveTypeArray2D<uint8_t>(32, 32);   // 2D Array
    for (uint32_t pxlIndex = 0; pxlIndex < 32 * 32; pxlIndex++)
        randomCU->setDataAt(typeid(uint8_t), pxlIndex, contents[pxlIndex]);

    // Updating the corresponding optimal split depending of the current mode
    if (mode == Learn::LearningMode::TRAINING)
    {
        ClassEnv::trainingTargetsCU->push_back(randomCU);
        ClassEnv::trainingTargetsOptimalSplits->push_back(contents[1024]);
    }
    else if (mode == Learn::LearningMode::VALIDATION)
    {
        ClassEnv::validationTargetsCU->push_back(randomCU);
        ClassEnv::validationTargetsOptimalSplits->push_back(contents[1024]);
    }

}

void ClassEnv::UpdateTargets(uint64_t currentGen, const std::string& databasePath)
{
    // Each ${nbGeneTargetChange} generation, generate new random training targets so that different targets are used
    if (currentGen % NB_GENERATION_BEFORE_TARGETS_CHANGE == 0)
    {
        // ---  Deleting old targets ---
        if (currentGen != 0) // Don't clear trainingTargets before initializing them
        {
            this->reset(this->seed, Learn::LearningMode::TRAINING);
            for (uint64_t idx_targ = 0; idx_targ < NB_TRAINING_TARGETS; idx_targ++)
                delete ClassEnv::trainingTargetsCU->at(idx_targ);   // Targets are allocated in getRandomCUFeaturesFromCSVFile()
            ClassEnv::trainingTargetsCU->clear();
            ClassEnv::trainingTargetsOptimalSplits->clear();
            this->actualTrainingCU = 0;
        }
        else        // Load VALIDATION Targets at the beginning of the training (i == 0)
        {
            for (uint64_t idx_targ = 0; idx_targ < NB_VALIDATION_TARGETS; idx_targ++)
                this->getRandomCU(Learn::LearningMode::VALIDATION, databasePath);
        }

        // ---  Loading next targets ---
        for (uint64_t idx_targ = 0; idx_targ < NB_TRAINING_TARGETS; idx_targ++)
            this->getRandomCU(Learn::LearningMode::TRAINING, databasePath);
    }
}

void ClassEnv::LoadNextCU()
{
    if (this->currentMode == Learn::LearningMode::TRAINING)
    {
        // Load next CU
        this->currentCU = *ClassEnv::trainingTargetsCU->at(this->actualTrainingCU);
        // Updating next split solution
        this->currentClass = ClassEnv::trainingTargetsOptimalSplits->at(this->actualTrainingCU);
        // Increment index
        this->actualTrainingCU++;

        // Looping on the beginning of training targets
        if (this->actualTrainingCU >= NB_TRAINING_TARGETS)
            this->actualTrainingCU = 0;
    }
    else if (this->currentMode == Learn::LearningMode::VALIDATION)
    {
        // Load next CU
        this->currentCU = *ClassEnv::validationTargetsCU->at(this->actualValidationCU);
        // Updating next split solution
        this->currentClass = ClassEnv::validationTargetsOptimalSplits->at(this->actualValidationCU);
        // Increment index
        this->actualValidationCU++;

        // Looping on the beginning of validation targets
        if (this->actualValidationCU >= NB_VALIDATION_TARGETS)
            this->actualValidationCU = 0;
    }
}

void ClassEnv::printClassifStatsTable(const Environment& env, const TPG::TPGVertex* bestRoot, const uint64_t numGen, std::string const& outputFile, bool readable)
{
    // Print table of classification of the best
    TPG::TPGExecutionEngine tee(env, nullptr);

    // Change the MODE
    this->reset(0, Learn::LearningMode::VALIDATION);    // TESTING in MNIST, mais marche pas ici

    // Fill the table
    const int nbClasses = 6;

    uint64_t classifTable[nbClasses][nbClasses] = { {0} };
    uint64_t nbPerClass[nbClasses] = { 0 };
    uint8_t actionID;

    for (uint64_t nbImage = 0; nbImage < this->NB_VALIDATION_TARGETS; nbImage++)
    {
        // Get answer
        uint64_t optimalActionID = this->currentClass;
        nbPerClass[optimalActionID]++;

        // Execute
        auto path = tee.executeFromRoot(*bestRoot);
        const auto* action = (const TPG::TPGAction*)path.at(path.size() - 1);
        actionID = (uint8_t) action->getActionID();

        // Increment table
        classifTable[optimalActionID][actionID]++;

        // Do action (to trigger image update)
        this->LoadNextCU();
    }

    // Reset the learning mode to TESTING
    this->reset(0, Learn::LearningMode::TESTING);

/*    // Computing Score :
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

                int nbChar = (int) (1 + (nb == 0 ? 0 : log10((double) nb)));
                for(int nbEspace = 0; nbEspace < (nbClasses - nbChar); nbEspace++)
                    fichier << " ";
                fichier << nb << (x == y ? "-" : " ");
            }
            uint64_t nb = nbPerClass[x];
            int nbChar = (int)(1 + (nb == 0 ? 0 : log10((double) nb)));
            for (int nbEspace = 0; nbEspace < (nbClasses - nbChar); nbEspace++)
                fichier << " ";
            fichier << nb << std::endl;
        }
        fichier.close();
    }else
    {
        std::cout << "Unable to open the file " << outputFile << "." << std::endl;
    }*/

    // Computing Score and ScoreMax
    double validationScore = 0.0;
    double scoreMax = 0.0;
    for (int i = 0; i < nbClasses; i++)
    {
        validationScore += ((double) classifTable[i][i]);
        scoreMax += ((double) nbPerClass[i]);
    }

    int colWidth = 7;

    // If readable confusion matrix is needed
    if(readable)
    {
        // Print the table
        std::ofstream file(outputFile.c_str(), std::ios::app);
        if (file)
        {
            // Compute total score
            double scoreTot = 0.0;
            for (int i = 0; i < nbClasses; i++)
            {
                double norm = (double) classifTable[i][i] / (double) nbPerClass[i] * 100;
                scoreTot += norm;
            }
            scoreTot /= (double) nbClasses;

            // Print the beginning of the confusion matrix
            file << "-----------------------------------------------------" << std::endl;
            file << "Gen: " << numGen << " | Score: " << std::setprecision(4) << scoreTot << std::endl << std::endl;

            file << std::setw(4) << " " << std::setw(colWidth) << "NS"
                 << std::setw(colWidth) << "QT"  << std::setw(colWidth) << "BTH"
                 << std::setw(colWidth) << "BTV" << std::setw(colWidth) << "TTH"
                 << std::setw(colWidth) << "TTV" << std::setw(colWidth) << "TOT" << std::endl;
            for (int x = 0; x < nbClasses; x++)
            {
                // Print real class number
                file << std::setw(4) << x;

                // Get total number of class instances
                uint64_t nb = nbPerClass[x];

                // Print number of guessed instances for each class
                for (int y = 0; y < nbClasses; y++)
                    file << std::setw(colWidth) << std::setprecision(4) << (double) classifTable[x][y] / (double) nb * 100;

                // Print total number of class instances
                file << std::setw(colWidth) << nb << std::endl;
            }
            file << std::endl;
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
                file << "Features TPG training" << std::endl << std::endl;

                file << std::setw(colWidth) << "Split" << std::setw(colWidth) << "NS"
                     << std::setw(colWidth) << "QT"  << std::setw(colWidth) << "BTH"
                     << std::setw(colWidth) << "BTV" << std::setw(colWidth) << "TTH"
                     << std::setw(colWidth) << "TTV" << std::setw(colWidth) << "TOT" << std::endl;

                file << std::setw(colWidth) << "Total";
                for(unsigned long nb : nbPerClass)
                    file << std::setw(colWidth) << nb;
                file << std::setw(colWidth) << this->NB_VALIDATION_TARGETS << std::endl << std::endl;

                file << std::setw(colWidth) << "Gen" << std::setw(colWidth) << "NS"
                     << std::setw(colWidth) << "QT"  << std::setw(colWidth) << "BTH"
                     << std::setw(colWidth) << "BTV" << std::setw(colWidth) << "TTH"
                     << std::setw(colWidth) << "TTV" << std::setw(colWidth) << "MOY" << std::endl;
            }
            file << std::setw(colWidth) << numGen;
            double scoreTot = 0.0;
            for (int i = 0; i < nbClasses; i++)
            {
                double norm   = (double) classifTable[i][i] / (double) nbPerClass[i] * 100;
                file << std::setw(colWidth) << std::setprecision(4) << norm;
                scoreTot += norm;
            }
            scoreTot /= (double) nbClasses;
            file << std::setw(colWidth) << std::setprecision(4) << scoreTot << std::endl;
            file.close();
        } else
        {
            std::cout << "Unable to open the file " << outputFile << "." << std::endl;
        }
    }
}
