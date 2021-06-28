#include <iostream>
#include <cmath>
#include <cinttypes>
#include <cstdlib>

#include <gegelati.h>

#include "../../include/features/BinaryFeaturesEnv.h"

void importTPG(BinaryFeaturesEnv* le, Environment& env, TPG::TPGGraph& tpg);
Data::PrimitiveTypeArray<double>* getRandomCUFeatures(std::string& datasetPath, BinaryFeaturesEnv* le, std::vector<uint8_t>* splitList);

int main()
{
    std::cout << "Start VVC Partitionning Optimization with binary features TPGs solution" << std::endl;

    int nbEval = 50;
    std::cout << nbEval << " evaluations" << std::endl;
    double moyenne = 0.0;
    for(int i = 0; i < nbEval; i++)
    {
        // ************************************************** INSTRUCTIONS *************************************************

        // Create the instruction set for programs
        Instructions::Set set;

        // double instructions
        auto minus_double = [](double a, double b) -> double { return a - b; };
        auto add_double = [](double a, double b) -> double { return a + b; };
        auto mult_double = [](double a, double b) -> double { return a * b; };
        auto div_double = [](double a, double b) -> double { return a / b; };
        auto max_double = [](double a, double b) -> double { return std::max(a, b); };
        auto ln_double = [](double a) -> double { return std::log(a); };
        auto exp_double = [](double a) -> double { return std::exp(a); };
        auto multByConst_double = [](double a, Data::Constant c) -> double { return a * (double) c; };
        /*auto conv2D_double = [](const Data::Constant coeff[9], const double data[3][3]) -> double {
            double res = 0.0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    res += (double) coeff[i * 3 + j] * data[i][j];
                }
            }
            return res;
        };*/

        // Add those instructions to instruction set
        set.add(*(new Instructions::LambdaInstruction<double, double>(minus_double)));
        set.add(*(new Instructions::LambdaInstruction<double, double>(add_double)));
        set.add(*(new Instructions::LambdaInstruction<double, double>(mult_double)));
        set.add(*(new Instructions::LambdaInstruction<double, double>(div_double)));
        set.add(*(new Instructions::LambdaInstruction<double, double>(max_double)));
        set.add(*(new Instructions::LambdaInstruction<double>(exp_double)));
        set.add(*(new Instructions::LambdaInstruction<double>(ln_double)));
        set.add(*(new Instructions::LambdaInstruction<double, Data::Constant>(multByConst_double)));
        //set.add(*(new Instructions::LambdaInstruction<const Data::Constant[9], const double[3][3]>(conv2D_double)));

        // ******************************************* PARAMETERS AND ENVIRONMENT ******************************************

        // ---------------- Load and initialize parameters from .json file ----------------
        Learn::LearningParameters params;
        File::ParametersParser::loadParametersFromJson(ROOT_DIR "/TPG/params.json", params);

        // Default arguments
        std::vector<uint8_t> actions0 = {0};
        std::vector<uint8_t> actions1 = {1,2,3,4,5};
        size_t seed = 0;
        uint64_t cuHeight = 32;
        uint64_t cuWidth = 32;
        uint64_t nbFeatures = 112;
        uint64_t nbDatabaseElements = 114348*6; // 32x32_balanced database

        // Initialising the number of CUs used
        //uint64_t nbDatabaseElements = 686088;   // Balanced database with 55000 elements of one class and 11000 elements of each other class : 110000
        // Balanced database with full classes : 330000
        // Unbalanced database : 1136424
        // Number of CUs preload changed every nbGeneTargetChange generation for training and load only once for validation
        uint64_t nbTrainingTargets = 10000;
        uint64_t nbGeneTargetChange = 30;
        uint64_t nbValidationTarget = 1000;


        // ************************************************** INSTANCES *************************************************

        // ---------------- Instantiate 6 LearningEnvironments, 6 Environments, 6 TPGGraphs and 6 TPGExecutionEngines ----------------
        // 6 LearningEnvironments are required because each instance own its specializedAction, its target vector, etc...
        // And therefore, each of the following object (Environment, TPGGraph, Engine, ...) depends on the learningEnvironment. So, 6 of each.
        auto *leNP = new BinaryFeaturesEnv({0}, {1,2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                           nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leQT = new BinaryFeaturesEnv({1}, {0,2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                           nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leBTH = new BinaryFeaturesEnv({2}, {0,1,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leBTV = new BinaryFeaturesEnv({3}, {0,1,2,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leTTH = new BinaryFeaturesEnv({4}, {0,1,2,3,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);

        // Instantiate the environment that will embed the LearningEnvironment
        Environment envNP(set, leNP->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envQT(set, leQT->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envBTH(set, leBTH->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envBTV(set, leBTV->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envTTH(set, leTTH->getDataSources(), params.nbRegisters, params.nbProgramConstant);

        // Instantiate the TPGGraph that we will load
        auto tpgNP = TPG::TPGGraph(envNP);
        auto tpgQT = TPG::TPGGraph(envQT);
        auto tpgBTH = TPG::TPGGraph(envBTH);
        auto tpgBTV = TPG::TPGGraph(envBTV);
        auto tpgTTH = TPG::TPGGraph(envTTH);

        // Instantiate the tee that will handle the decisions taken by the TPG
        TPG::TPGExecutionEngine teeNP(envNP);
        TPG::TPGExecutionEngine teeQT(envQT);
        TPG::TPGExecutionEngine teeBTH(envBTH);
        TPG::TPGExecutionEngine teeBTV(envBTV);
        TPG::TPGExecutionEngine teeTTH(envTTH);

        // ---------------- Import TPG graph from .dot file ----------------
        importTPG(leNP, envNP, tpgNP);
        importTPG(leQT, envQT, tpgQT);
        importTPG(leBTH, envBTH, tpgBTH);
        importTPG(leBTV, envBTV, tpgBTV);
        importTPG(leTTH, envTTH, tpgTTH);

        // ---------------- Execute TPG from root on the environment ----------------
        auto rootNP = tpgNP.getRootVertices().front();
        auto rootQT = tpgQT.getRootVertices().front();
        auto rootBTH = tpgBTH.getRootVertices().front();
        auto rootBTV = tpgBTV.getRootVertices().front();
        auto rootTTH = tpgTTH.getRootVertices().front();

        // ************************************************** MAIN RUN *************************************************
        // Path to the global database with 330.000 elements (55.000 of each class)
        std::string datasetBasePath = "/media/cleonard/alex/cedric_TPG-VVC/balanced_datasets/";
        std::string datasetMiddlePath = "x";
        std::string datasetEndPath = "_balanced/";
        std::string datasetPath = datasetBasePath + std::to_string(leNP->getCuHeight()) + datasetMiddlePath + std::to_string(leNP->getCuWidth()) + datasetEndPath;


        auto *dataHandler = new std::vector<Data::PrimitiveTypeArray<double> *>;
        auto *splitList = new std::vector<uint8_t>;

        // Load a vector of 1000 CUs (dataHandler) and their corresponding split (splitList)
        // -------------------------- Load a global vector of 1.000 CUs --------------------------
        for (uint64_t idx_targ = 0; idx_targ < leNP->NB_VALIDATION_TARGETS; idx_targ++) {
            Data::PrimitiveTypeArray<double> *target = getRandomCUFeatures(datasetPath, leNP, splitList);
            dataHandler->push_back(target);
            // Optimal split is stored in splitList inside getRandomCU()
        }

        // -------------------------- Load next CU for all TPG --------------------------
        uint64_t score = 0;
        int actionID;
        int chosenAction = -1;

        // Run NB_VALIDATION_TARGETS times the TPG, each time on a different CU
        for (uint64_t nbCU = 0; nbCU < leNP->NB_VALIDATION_TARGETS; nbCU++)
        {
            // -------------------------- Load next CU for all TPGs --------------------------
            leNP->setCurrentState(*dataHandler->at(nbCU));
            leQT->setCurrentState(*dataHandler->at(nbCU));
            leBTH->setCurrentState(*dataHandler->at(nbCU));
            leBTV->setCurrentState(*dataHandler->at(nbCU));
            leTTH->setCurrentState(*dataHandler->at(nbCU));

            // ************************************** ORDER : NP QT BTH BTV TTH ***************************************
            // -------------------------- Call NP TPG --------------------------
            actionID = (int) ((const TPG::TPGAction *) teeNP.executeFromRoot(* rootNP).back())->getActionID();
            //std::cout << "NP Action : " << actionID << std::endl;
            if(actionID == 1)
                chosenAction = leNP->getActions0().at(0);
            else
            {
                // -------------------------- Call QT TPG --------------------------
                actionID = (int) ((const TPG::TPGAction *) teeQT.executeFromRoot(* rootQT).back())->getActionID();
                //std::cout << "QT Action : " << actionID << std::endl;
                if(actionID == 1)
                    chosenAction = leQT->getActions0().at(0);
                else
                {
                    // -------------------------- Call BTH TPG --------------------------
                    actionID = (int) ((const TPG::TPGAction *) teeBTH.executeFromRoot(* rootBTH).back())->getActionID();
                    //std::cout << "BTH Action : " << actionID << std::endl;
                    if(actionID == 1)
                        chosenAction = leBTH->getActions0().at(0);
                    else
                    {
                        // -------------------------- Call BTV TPG --------------------------
                        actionID = (int) ((const TPG::TPGAction *) teeBTV.executeFromRoot(* rootBTV).back())->getActionID();
                        //std::cout << "BTV Action : " << actionID << std::endl;
                        if(actionID == 1)
                            chosenAction = leBTV->getActions0().at(0);
                        else
                        {
                            // -------------------------- Call TTH TPG --------------------------
                            actionID = (int) ((const TPG::TPGAction *) teeTTH.executeFromRoot(* rootTTH).back())->getActionID();
                            //std::cout << "TTH Action : " << actionID << std::endl;
                            if(actionID == 1)
                                chosenAction = leTTH->getActions0().at(0);
                             // TTH
                        } // BTV
                    } // BTH
                } // QT
            } // NP

            // -------------------------- Update Score --------------------------
            //std::cout << "Action : " << chosenAction << ", real Split : " << (int) splitList->at(nbCU) << std::endl;
            if (chosenAction == (int) splitList->at(nbCU))
                score++;

            //std::cout << "TPG: " << actionID << " Sol: " << (uint64_t) le->getOptimalSplit() << std::endl;
        }
        std::cout << "Score : " << score << "/" << leNP->NB_VALIDATION_TARGETS << std::endl;
        moyenne += (double) score;

        // ---------------- Clean ----------------
        // instructions
        for (unsigned int j = 0; j < set.getNbInstructions(); j++)
            delete (&set.getInstruction(j));
        // LearningEnvironment
        delete leNP;
        delete leQT;
        delete leBTH;
        delete leBTV;
        delete leTTH;
        // Data and solution handlers
        delete dataHandler;
        delete splitList;
    }
    moyenne /= nbEval;
    std::cout << "Score moyen : " << moyenne << "/1000" << std::endl;
    return 0;
}

void importTPG(BinaryFeaturesEnv* le, Environment& env, TPG::TPGGraph& tpg)
{
    try{
        char tpgPath[100] = ROOT_DIR"/TPG/";
        std::string speActionName = le->getActionName(le->getActions0().at(0));
        std::strcat(tpgPath, speActionName.c_str());
        char dotExtension[10] = ".dot";
        std::strcat(tpgPath, dotExtension);
        File::TPGGraphDotImporter dotImporter(tpgPath, env, tpg);
    }catch (std::runtime_error& e){
        std::cout << e.what() << std::endl;
    }
}

Data::PrimitiveTypeArray<double>* getRandomCUFeatures(std::string& datasetPath, BinaryFeaturesEnv* le, std::vector<uint8_t>* splitList)
{

    // ------------------ Opening and Reading a random CSV file ------------------
    // Generate the path for a random CSV file
    uint32_t next_CU_number = rand() % (int) le->NB_DATABASE_ELEMENTS;
    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number);
    char CSV_path[100];
    std::strcpy(CSV_path, datasetPath.c_str());
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
        auto *randomCU = new Data::PrimitiveTypeArray<double>(le->getNbFeatures()+1); // +1 for QP
        // Fill it with QP Value and then every features
        randomCU->setDataAt(typeid(double), 0, std::stod(row.at(0)));
        for (uint32_t featuresIdx = 2; featuresIdx < le->getNbFeatures()+2; featuresIdx++)
            randomCU->setDataAt(typeid(double), featuresIdx-2, std::stod(row.at(featuresIdx)));

        // -------- Store the features array (currentState) and its split --------
        // Deduce the optimal split from string
        //std::cout << "Nom du split  : \"" << row.at(1) << "\"" << std::endl;
        splitList->push_back(le->getSplitNumber(row.at(1)));
        // Store the CU features and the corresponding optimal split depending of the current mode

        return randomCU;
    }

    return NULL;
}
