#include "../../include/features/FeaturesEnv.h"

// ********************************************************************* //
// ************************** GEGELATI FUNCTIONS *********************** //
// ********************************************************************* //

void FeaturesEnv::doAction(uint64_t actionID)
{
    // Call the doAction() method of the ClassificationLearningEnvironment : Update the reward
    ClassificationLearningEnvironment::doAction(actionID);

    // Load next CU features
    this->LoadNextCUFeatures();
}

std::vector<std::reference_wrapper<const Data::DataHandler>> FeaturesEnv::getDataSources()
{
    // Return a vector containing every element constituting the State of the environment
    std::vector<std::reference_wrapper<const Data::DataHandler>> result{this->currentState};
    return result;
}

void FeaturesEnv::reset(size_t seed, Learn::LearningMode mode)
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

Learn::LearningEnvironment *FeaturesEnv::clone() const
{
    return new FeaturesEnv(*this);
}

bool FeaturesEnv::isCopyable() const
{
    return true; // false : to avoid ParallelLearning (Cf. LearningAgent)
}

double FeaturesEnv::getScore() const
{
    return ClassificationLearningEnvironment::getScore();
}

bool FeaturesEnv::isTerminal() const
{
    return false;
}


// ********************************************************************* //
// *************************** PartCU FUNCTIONS ************************ //
// ********************************************************************* //

// ****** TRAINING Arguments ******
std::vector<Data::PrimitiveTypeArray<double> *> *FeaturesEnv::trainingTargetsCUFeatures = new std::vector<Data::PrimitiveTypeArray<double>*>;
std::vector<uint8_t> *FeaturesEnv::trainingTargetsOptimalSplits = new std::vector<uint8_t>;
// ****** VALIDATION Arguments ******
std::vector<Data::PrimitiveTypeArray<double>*> *FeaturesEnv::validationTargetsCUFeatures = new std::vector<Data::PrimitiveTypeArray<double>*>;
std::vector<uint8_t> *FeaturesEnv::validationTargetsOptimalSplits = new std::vector<uint8_t>;

void FeaturesEnv::getRandomCUFeaturesFromOriginalCSVFile(Learn::LearningMode mode, const char current_CU_path[100])
{
    // ------------------ Opening and Reading a random CSV file ------------------
    // Generate the path for a random CSV file
    uint32_t next_CU_number = this->rng.getInt32(0, NB_TRAINING_ELEMENTS - 1);
    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number);
    char CSV_path[100];
    std::strcpy(CSV_path, current_CU_path);
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
    std::string word, temp;

    int i = 0;
    for( std::string line; getline(file, line ); )
    {
        // Don't process first line
        if(i != 0)
        {
            // -------- Get a whole line --------
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
            auto *randomCU = new Data::PrimitiveTypeArray<double>(FeaturesEnv::CSV_FILE_WIDTH+1); // +1 for QP
            // Fill it with QP Value and then every features
            randomCU->setDataAt(typeid(double), 0, std::stod(row.at(1)));
            for (uint32_t featuresIdx = 3; featuresIdx < FeaturesEnv::CSV_FILE_WIDTH+3; featuresIdx++)
                randomCU->setDataAt(typeid(double), featuresIdx-2, std::stod(row.at(featuresIdx)));

            // -------- Store the features array (currentState) and its split --------
            // Deduce the optimal split from string
            //std::cout << "Nom du split  : \"" << row.at(2) << "\"" << std::endl;
            uint8_t optSplit = getSplitNumber(row.at(2));
            // Store the CU features and the corresponding optimal split depending of the current mode
            if (mode == Learn::LearningMode::TRAINING)
            {
                FeaturesEnv::trainingTargetsCUFeatures->push_back(randomCU);
                FeaturesEnv::trainingTargetsOptimalSplits->push_back(optSplit);
            }
            else if (mode == Learn::LearningMode::VALIDATION)
            {
                FeaturesEnv::validationTargetsCUFeatures->push_back(randomCU);
                FeaturesEnv::validationTargetsOptimalSplits->push_back(optSplit);
            }
        }
        i++;
    }
    file.close();
}

void FeaturesEnv::getRandomCUFeaturesFromSimpleCSVFile(Learn::LearningMode mode, const char current_CU_path[100])
{
    // ------------------ Opening and Reading a random CSV file ------------------
    // Generate the path for a random CSV file
    uint32_t next_CU_number = this->rng.getInt32(0, NB_TRAINING_ELEMENTS - 1);
    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number);
    char CSV_path[100];
    std::strcpy(CSV_path, current_CU_path);
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
        auto *randomCU = new Data::PrimitiveTypeArray<double>(FeaturesEnv::CSV_FILE_WIDTH+1); // +1 for QP
        // Fill it with QP Value and then every features
        randomCU->setDataAt(typeid(double), 0, std::stod(row.at(0)));
        for (uint32_t featuresIdx = 2; featuresIdx < FeaturesEnv::CSV_FILE_WIDTH+2; featuresIdx++)
            randomCU->setDataAt(typeid(double), featuresIdx-2, std::stod(row.at(featuresIdx)));

        // -------- Store the features array (currentState) and its split --------
        // Deduce the optimal split from string
        //std::cout << "Nom du split  : \"" << row.at(1) << "\"" << std::endl;
        uint8_t optSplit = getSplitNumber(row.at(1));
        // Store the CU features and the corresponding optimal split depending of the current mode
        if (mode == Learn::LearningMode::TRAINING)
        {
            FeaturesEnv::trainingTargetsCUFeatures->push_back(randomCU);
            FeaturesEnv::trainingTargetsOptimalSplits->push_back(optSplit);
        }
        else if (mode == Learn::LearningMode::VALIDATION)
        {
            FeaturesEnv::validationTargetsCUFeatures->push_back(randomCU);
            FeaturesEnv::validationTargetsOptimalSplits->push_back(optSplit);
        }

    }
    file.close();
}

void FeaturesEnv::UpdatingTargets(uint64_t currentGen, const char current_CU_path[100])
{
    // Each ${nbGeneTargetChange} generation, generate new random training targets so that different targets are used
    if (currentGen % NB_GENERATION_BEFORE_TARGETS_CHANGE == 0)
    {
        // ---  Deleting old targets ---
        if (currentGen != 0) // Don't clear trainingTargets before initializing them
        {
            this->reset(0, Learn::LearningMode::TRAINING);
            for (uint64_t idx_targ = 0; idx_targ < NB_TRAINING_TARGETS; idx_targ++)
                delete FeaturesEnv::trainingTargetsCUFeatures->at(idx_targ);   // targets are allocated in getRandomCUFeaturesFromSimpleCSVFile()
            FeaturesEnv::trainingTargetsCUFeatures->clear();
            FeaturesEnv::trainingTargetsOptimalSplits->clear();
            this->actualTrainingCU = 0;
        }
        else        // Load VALIDATION Targets at the beginning of the training (i == 0)
        {
            for (uint64_t idx_targ = 0; idx_targ < NB_VALIDATION_TARGETS; idx_targ++)
                this->getRandomCUFeaturesFromSimpleCSVFile(Learn::LearningMode::VALIDATION, current_CU_path);
        }

        // ---  Loading next targets ---
        for (uint64_t idx_targ = 0; idx_targ < NB_TRAINING_TARGETS; idx_targ++)
            this->getRandomCUFeaturesFromSimpleCSVFile(Learn::LearningMode::TRAINING, current_CU_path);
    }
}

void FeaturesEnv::LoadNextCUFeatures()
{
    if (this->currentMode == Learn::LearningMode::TRAINING)
    {
        this->currentState = *FeaturesEnv::trainingTargetsCUFeatures->at(this->actualTrainingCU);
        this->currentClass = FeaturesEnv::trainingTargetsOptimalSplits->at(this->actualTrainingCU);
        this->actualTrainingCU++;

        // Looping on the beginning of training targets
        if (this->actualTrainingCU >= NB_TRAINING_TARGETS)
            this->actualTrainingCU = 0;
    }
    else if (this->currentMode == Learn::LearningMode::VALIDATION)
    {
        this->currentState = *FeaturesEnv::validationTargetsCUFeatures->at(this->actualValidationCU);
        this->currentClass = FeaturesEnv::validationTargetsOptimalSplits->at(this->actualValidationCU);
        this->actualValidationCU++;

        // Looping on the beginning of validation targets
        if (this->actualValidationCU >= NB_VALIDATION_TARGETS)
            this->actualValidationCU = 0;
    }
}

void FeaturesEnv::printClassifStatsTable(const Environment& env, const TPG::TPGVertex* bestRoot, const int numGen, std::string const& outputFile, bool readable)
{
    // Create a new TPGExecutionEngine from the environment
    TPG::TPGExecutionEngine tee(env, nullptr);

    // Change the LearningMode in VALIDATION
    this->reset(0, Learn::LearningMode::VALIDATION);

    // Fill the table
    const int nbClasses = 6;

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
                file << std::endl << std::endl;

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

uint8_t FeaturesEnv::getSplitNumber(const std::string& split)
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
