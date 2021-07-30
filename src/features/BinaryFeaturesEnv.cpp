#include "../../include/features/BinaryFeaturesEnv.h"

// ********************************************************************* //
// ************************** GEGELATI FUNCTIONS *********************** //
// ********************************************************************* //

void BinaryFeaturesEnv::doAction(uint64_t actionID)
{
    // Call the doAction() method of the ClassificationLearningEnvironment : Update the reward
    ClassificationLearningEnvironment::doAction(actionID);

    // Load next CU features
    this->LoadNextCUFeatures();
}

std::vector<std::reference_wrapper<const Data::DataHandler>> BinaryFeaturesEnv::getDataSources()
{
    // Return a vector containing every element constituting the State of the environment
    std::vector<std::reference_wrapper<const Data::DataHandler>> result{this->currentState};
    return result;
}

void BinaryFeaturesEnv::reset(size_t seed, Learn::LearningMode mode)
{
    // Reset the classificationTable and the score
    ClassificationLearningEnvironment::reset(seed);

    // Update the LearningMode
    this->currentMode = mode;

    // Reset the RNG
    this->rng.setSeed(seed);

    // Preload the first CU (depending on the current mode)
    this->LoadNextCUFeatures();
}

Learn::LearningEnvironment* BinaryFeaturesEnv::clone() const { return new BinaryFeaturesEnv(*this); }
bool BinaryFeaturesEnv::isCopyable() const { return true; }
double BinaryFeaturesEnv::getScore() const { return ClassificationLearningEnvironment::getScore(); }
bool BinaryFeaturesEnv::isTerminal() const { return false; }


// ********************************************************************* //
// *************************** PartCU FUNCTIONS ************************ //
// ********************************************************************* //

// ****** TRAINING Arguments ******
std::vector<Data::PrimitiveTypeArray<double> *> *BinaryFeaturesEnv::trainingTargetsData = new std::vector<Data::PrimitiveTypeArray<double>*>;
std::vector<uint8_t> *BinaryFeaturesEnv::trainingTargetsSplits = new std::vector<uint8_t>;
// ****** VALIDATION Arguments ******
std::vector<Data::PrimitiveTypeArray<double>*> *BinaryFeaturesEnv::validationTargetsData = new std::vector<Data::PrimitiveTypeArray<double>*>;
std::vector<uint8_t> *BinaryFeaturesEnv::validationTargetsSplits = new std::vector<uint8_t>;

void BinaryFeaturesEnv::getRandomCUFeaturesFromCSVFile(Learn::LearningMode mode, const std::string& databasePath)
{
    // ------------------ Opening and Reading a random CSV file ------------------
    // Generate the path for a random CSV file
    uint32_t next_CU_number = this->rng.getInt32(0, this->NB_DATABASE_ELEMENTS-1);
    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number);
    char CSV_path[100];
    std::strcpy(CSV_path, databasePath.c_str());
    char file_extension[10] = ".csv";
    std::strcat(CSV_path, next_CU_number_string);
    std::strcat(CSV_path, file_extension);

    // Init File pointer
    std::ifstream file;
    // Open the existing file
    file.open(CSV_path, std::ios::in);

    // ------------------ Read the Data from the file as String Vector ------------------
    // Init the vector which will contain the string
    std::vector<std::string> row;
    std::string word;
    if (file.good())
    {
        std::string line;
        getline(file, line);
        // -------- Get the whole file as a line --------
        // Clear the vector
        row.clear();
        // Used for breaking words
        std::istringstream s(line);
        // Read every column data of a row and store it in a string variable, 'word'
        while (std::getline(s, word, ','))
            row.push_back(word);

        /*std::cout << "size : " << row.size() << " row :";
        for(auto & j : row)
            std::cout << " " << j;
        std::cout << std::endl;*/

        // -------- Create and fill the container --------
        // Create a new PrimitiveTypeArray<uint8_t> which will contain 1 CU features
        auto *randomCU = new Data::PrimitiveTypeArray<double>(BinaryFeaturesEnv::NB_FEATURES+1); // +1 for QP
        // Fill it with QP Value and then every features
        randomCU->setDataAt(typeid(double), 0, std::stod(row.at(0)));
        for (uint32_t featuresIdx = 2; featuresIdx < BinaryFeaturesEnv::NB_FEATURES+2; featuresIdx++)
            randomCU->setDataAt(typeid(double), featuresIdx-2, std::stod(row.at(featuresIdx)));

        // -------- Store the features array (currentState) and its split --------
        // Deduce the optimal split from string
        //std::cout << "Nom du split  : \"" << row.at(1) << "\"" << std::endl;
        uint8_t optSplit = getSplitNumber(row.at(1));
        // Store the CU features and the corresponding optimal split depending of the current mode
        if (mode == Learn::LearningMode::TRAINING)
        {
            BinaryFeaturesEnv::trainingTargetsData->push_back(randomCU);
            BinaryFeaturesEnv::trainingTargetsSplits->push_back(optSplit);
        }
        else if (mode == Learn::LearningMode::VALIDATION)
        {
            BinaryFeaturesEnv::validationTargetsData->push_back(randomCU);
            BinaryFeaturesEnv::validationTargetsSplits->push_back(optSplit);
        }

    }/*else
    {
        fprintf(stderr, "File not good : %s\n", CSV_path);
    }*/
    file.close();
}

void BinaryFeaturesEnv::UpdateTargets(uint64_t currentGen, const std::string& databasePath)
{
    // Each ${nbGeneTargetChange} generation, generate new random training targets so that different targets are used
    if (currentGen % NB_GENERATION_BEFORE_TARGETS_CHANGE == 0)
    {
        // ---  Deleting old targets ---
        if (currentGen != 0) // Don't clear trainingTargets before initializing them
        {
            this->reset(this->seed, Learn::LearningMode::TRAINING);
            for (uint64_t idx_targ = 0; idx_targ < NB_TRAINING_TARGETS; idx_targ++)
                delete BinaryFeaturesEnv::trainingTargetsData->at(idx_targ);   // Targets are allocated in getRandomCUFeaturesFromCSVFile()
            BinaryFeaturesEnv::trainingTargetsData->clear();
            BinaryFeaturesEnv::trainingTargetsSplits->clear();
            this->actualTrainingCU = 0;
        }
        else        // Load VALIDATION Targets at the beginning of the training (i == 0)
        {
            for (uint64_t idx_targ = 0; idx_targ < NB_VALIDATION_TARGETS; idx_targ++)
                this->getRandomCUFeaturesFromCSVFile(Learn::LearningMode::VALIDATION, databasePath);
        }

        // ---  Loading next targets ---
        for (uint64_t idx_targ = 0; idx_targ < NB_TRAINING_TARGETS; idx_targ++)
            this->getRandomCUFeaturesFromCSVFile(Learn::LearningMode::TRAINING, databasePath);
    }
}

void BinaryFeaturesEnv::LoadNextCUFeatures()
{
    if (this->currentMode == Learn::LearningMode::TRAINING)
    {
        this->currentState = *BinaryFeaturesEnv::trainingTargetsData->at(this->actualTrainingCU);

        uint8_t optimalSplit = BinaryFeaturesEnv::trainingTargetsSplits->at(this->actualTrainingCU);
        this->updateCurrentClass(optimalSplit);

        this->actualTrainingCU++;

        // Looping on the beginning of training targets
        if (this->actualTrainingCU >= NB_TRAINING_TARGETS)
            this->actualTrainingCU = 0;
    }
    else if (this->currentMode == Learn::LearningMode::VALIDATION)
    {
        this->currentState = *BinaryFeaturesEnv::validationTargetsData->at(this->actualValidationCU);

        uint8_t optimalSplit = BinaryFeaturesEnv::validationTargetsSplits->at(this->actualValidationCU);
        this->updateCurrentClass(optimalSplit);

        this->actualValidationCU++;

        // Looping on the beginning of validation targets
        if (this->actualValidationCU >= NB_VALIDATION_TARGETS)
            this->actualValidationCU = 0;
    }
}

void BinaryFeaturesEnv::updateCurrentClass(uint8_t optimalSplit)
{
    bool updated = false;
    for(int i = 0; !updated && i < (int) this->actions0.size(); i++)
    {
        if (optimalSplit == this->actions0.at(i))
        {
            this->currentClass = 0;
            updated = true;
        }
    }
    if(!updated)
    {
        for(int i = 0; !updated && i < (int) this->actions1.size(); i++)
        {
            if (optimalSplit == this->actions1.at(i))
            {
                this->currentClass = 1;
                updated = true;
            }
        }
    }
}

void BinaryFeaturesEnv::printClassifStatsTable(const Environment& env, const TPG::TPGVertex* bestRoot, const int numGen, std::string const& outputFile, bool readable)
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
        uint64_t optimalActionID = this->currentClass;
        nbPerClass[optimalActionID]++;

        // std::cout << optimalActionID << " : " << nbPerClass[optimalActionID] << std::endl;

        // Execute
        auto path = tee.executeFromRoot(*bestRoot);
        const auto *action = (const TPG::TPGAction *) path.at(path.size() - 1);
        actionID = (uint8_t) action->getActionID();

        // Increment table
        classifTable[optimalActionID][actionID]++;

        // Do action in order to trigger image update and load the next CU
        this->LoadNextCUFeatures();
    }

    // Reset the learning mode to TESTING
    this->reset(0, Learn::LearningMode::TESTING);

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

            file << std::setw(4) << " " << std::setw(colWidth) << "speAct"
                 << std::setw(colWidth) << "OTHERS" << std::endl;
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

                file << std::setw(colWidth) << "Gen" << std::setw(colWidth) << "speAct"
                     << std::setw(colWidth) << "OTHER" << std::endl;
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

uint8_t BinaryFeaturesEnv::getSplitNumber(const std::string& split)
{
    uint8_t splitNumber;
    if(split == "NS")
        splitNumber = 0;
    else if(split == "QT")
        splitNumber = 1;
    else if(split == "BTH")
        splitNumber = 2;
    else if(split == "BTV")
        splitNumber = 3;
    else if(split == "TTH")
        splitNumber = 4;
    else if(split == "TTV")
        splitNumber = 5;
    else
        splitNumber = 6;
    return splitNumber;
}

std::string BinaryFeaturesEnv::getActionName(uint64_t speAct)
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

// *************************************************** GETTERS *****************************************************
uint64_t BinaryFeaturesEnv::getCuHeight() const { return CU_HEIGHT; }
uint64_t BinaryFeaturesEnv::getCuWidth() const { return CU_WIDTH; }
const uint64_t BinaryFeaturesEnv::getNbFeatures() const { return NB_FEATURES; }
const std::vector<uint8_t> &BinaryFeaturesEnv::getActions0() const { return actions0; }
const std::vector<uint8_t> &BinaryFeaturesEnv::getActions1() const { return actions1; }
// *************************************************** SETTERS *****************************************************
void BinaryFeaturesEnv::setCurrentState(const Data::PrimitiveTypeArray<double> &state) { BinaryFeaturesEnv::currentState = state; }
