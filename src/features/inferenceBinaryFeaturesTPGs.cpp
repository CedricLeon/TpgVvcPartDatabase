#include <iostream>
#include <cmath>
#include <cinttypes>
#include <cstdlib>

#include <gegelati.h>

#include "../../include/features/BinaryFeaturesEnv.h"

void importTPG(BinaryFeaturesEnv* le, Environment& env, TPG::TPGGraph& tpg);
Data::PrimitiveTypeArray<double>* getRandomCUFeatures(std::string& datasetPath, BinaryFeaturesEnv* le, std::vector<uint8_t>* splitList);
uint64_t EvaluateLinearWaterFall(BinaryFeaturesEnv* leNP,
                                 TPG::TPGExecutionEngine teeNP,
                                 const TPG::TPGVertex* rootNP,
                                 BinaryFeaturesEnv* leQT,
                                 TPG::TPGExecutionEngine teeQT,
                                 const TPG::TPGVertex* rootQT,
                                 BinaryFeaturesEnv* leBTH,
                                 TPG::TPGExecutionEngine teeBTH,
                                 const TPG::TPGVertex* rootBTH,
                                 BinaryFeaturesEnv* leBTV,
                                 TPG::TPGExecutionEngine teeBTV,
                                 const TPG::TPGVertex* rootBTV,
                                 BinaryFeaturesEnv* leTTH,
                                 TPG::TPGExecutionEngine teeTTH,
                                 const TPG::TPGVertex* rootTTH,
                                 std::vector<Data::PrimitiveTypeArray<double> *>& dataHandler,
                                 std::vector<uint8_t>& splitList);

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
        size_t seed = 0;
        uint64_t cuHeight = 32;
        uint64_t cuWidth = 32;
        uint64_t nbFeatures = 112;
        uint64_t nbDatabaseElements = 114348*6; // 32x32_balanced database
        uint64_t nbTrainingTargets = 10000;
        uint64_t nbGeneTargetChange = 30;
        uint64_t nbValidationTarget = 1000;

        // ************************************************** INSTANCES *************************************************

        // ---------------- Instantiate 6 LearningEnvironments, 6 Environments, 6 TPGGraphs and 6 TPGExecutionEngines ----------------
        // 6 LearningEnvironments are required because each instance own its specializedAction, its target vector, etc...
        // And therefore, each of the following object (Environment, TPGGraph, Engine, ...) depends on the learningEnvironment. So, 6 of each.
        auto *leNP = new BinaryFeaturesEnv({0}, {1,2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                           nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leQT = new BinaryFeaturesEnv({1}, {2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                           nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leBTH = new BinaryFeaturesEnv({2}, {3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leBTV = new BinaryFeaturesEnv({3}, {4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leTTH = new BinaryFeaturesEnv({4}, {5}, seed, cuHeight, cuWidth, nbFeatures,
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
        std::string datasetPath = datasetBasePath
                                + std::to_string(leNP->getCuHeight())
                                + datasetMiddlePath
                                + std::to_string(leNP->getCuWidth())
                                + datasetEndPath;


        auto *dataHandler = new std::vector<Data::PrimitiveTypeArray<double> *>;
        auto *splitList = new std::vector<uint8_t>;
        auto *CUchosen = new std::vector<int>{0,0,0,0,0,0};
        auto *CUset = new std::vector<int>{0,0,0,0,0,0};

        // Load a vector of 1000 CUs (dataHandler) and their corresponding split (splitList)
        // -------------------------- Load a global vector of 1.000 CUs --------------------------
        for (uint64_t idx_targ = 0; idx_targ < leNP->NB_VALIDATION_TARGETS; idx_targ++)
        {
            Data::PrimitiveTypeArray<double> *target = getRandomCUFeatures(datasetPath, leNP, splitList);
            dataHandler->push_back(target);
            // Optimal split is stored in splitList inside getRandomCU()
        }

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

            // ************************************ ORDER : NP QT DIREC HORI VERTI *************************************
            // -------------------------- Call NP TPG --------------------------
            actionID = (int) ((const TPG::TPGAction *) teeNP.executeFromRoot(* rootNP).back())->getActionID();
            //std::cout << "NP Action : " << actionID << std::endl;
            if(actionID == 0)
                chosenAction = 0; //leNP->getActions0().at(0);
            else
            {
                // -------------------------- Call QT TPG --------------------------
                actionID = (int) ((const TPG::TPGAction *) teeQT.executeFromRoot(* rootQT).back())->getActionID();
                //std::cout << "QT Action : " << actionID << std::endl;
                if(actionID == 0)
                    chosenAction = 1; //leQT->getActions0().at(0);
                else
                {
                    // -------------------------- Call DIREC TPG --------------------------
                    actionID = (int) ((const TPG::TPGAction *) teeBTH.executeFromRoot(* rootBTH).back())->getActionID();
                    //std::cout << "BTH Action : " << actionID << std::endl;
                    if(actionID == 0)
                    {
                        // -------------------------- Call HORI TPG --------------------------
                        actionID = (int) ((const TPG::TPGAction *) teeBTV.executeFromRoot(* rootBTV).back())->getActionID();
                        //std::cout << "HORI Action : " << actionID << std::endl;
                        if(actionID == 0)
                            chosenAction = 2; //leBTV->getActions0().at(0);
                        else
                            chosenAction = 4; //leBTV->getActions1().at(0);
                         // HORI
                    }
                    else
                    {
                        // -------------------------- Call VERTI TPG --------------------------
                        actionID = (int) ((const TPG::TPGAction *) teeTTH.executeFromRoot(* rootTTH).back())->getActionID();
                        //std::cout << "VERTI Action : " << actionID << std::endl;
                        if(actionID == 0)
                            chosenAction = 3; //leTTH->getActions0().at(0);
                        else
                            chosenAction = 5; //leTTH->getActions1().at(0);
                         // VERTI
                    } // DIREC
                } // QT
            } // NP

            // -------------------------- Update Score --------------------------
            //std::cout << "Action : " << chosenAction << ", real Split : " << (int) splitList->at(nbCU) << std::endl;
            if (chosenAction == (int) splitList->at(nbCU))
                score++;
            CUchosen->at(chosenAction) ++;
            CUset->at((int) splitList->at(nbCU)) ++;
        }

        /*// Run NB_VALIDATION_TARGETS times the TPG, each time on a different CU
        for (uint64_t nbCU = 0; nbCU < leNP->NB_VALIDATION_TARGETS; nbCU++)
        {
            // -------------------------- Load next CU for all TPGs --------------------------
            leNP->setCurrentState(*dataHandler->at(nbCU));
            leQT->setCurrentState(*dataHandler->at(nbCU));
            leBTH->setCurrentState(*dataHandler->at(nbCU));
            leBTV->setCurrentState(*dataHandler->at(nbCU));
            leTTH->setCurrentState(*dataHandler->at(nbCU));

            // ************************************** ORDER : NP QT BTH BTV TTH ****************************************
            // -------------------------- Call NP TPG --------------------------
            actionID = (int) ((const TPG::TPGAction *) teeNP.executeFromRoot(* rootNP).back())->getActionID();
            //std::cout << "NP Action : " << actionID << std::endl;
            if(actionID == 0)
                chosenAction = leNP->getActions0().at(0);
            else
            {
                // -------------------------- Call QT TPG --------------------------
                actionID = (int) ((const TPG::TPGAction *) teeQT.executeFromRoot(* rootQT).back())->getActionID();
                //std::cout << "QT Action : " << actionID << std::endl;
                if(actionID == 0)
                    chosenAction = leQT->getActions0().at(0);
                else
                {
                    // -------------------------- Call BTH TPG --------------------------
                    actionID = (int) ((const TPG::TPGAction *) teeBTH.executeFromRoot(* rootBTH).back())->getActionID();
                    //std::cout << "BTH Action : " << actionID << std::endl;
                    if(actionID == 0)
                        chosenAction = leBTH->getActions0().at(0);
                    else
                    {
                        // -------------------------- Call BTV TPG --------------------------
                        actionID = (int) ((const TPG::TPGAction *) teeBTV.executeFromRoot(* rootBTV).back())->getActionID();
                        //std::cout << "BTV Action : " << actionID << std::endl;
                        if(actionID == 0)
                            chosenAction = leBTV->getActions0().at(0);
                        else
                        {
                            // -------------------------- Call TTH TPG --------------------------
                            actionID = (int) ((const TPG::TPGAction *) teeTTH.executeFromRoot(* rootTTH).back())->getActionID();
                            //std::cout << "TTH Action : " << actionID << std::endl;
                            if(actionID == 0)
                                chosenAction = leTTH->getActions0().at(0);
                            // TTH
                        } // BTV
                    } // BTH
                } // QT
            } // NP

            // -------------------------- Update Score --------------------------
            std::cout << "Action : " << chosenAction << ", real Split : " << (int) splitList->at(nbCU) << std::endl;
            if (chosenAction == (int) splitList->at(nbCU))
                score++;
        }*/

/*        // -------------------------- Load next CU for all TPG --------------------------
        uint64_t scoreFct = EvaluateLinearWaterFall(leNP, teeNP, rootNP,
                                                 leQT, teeQT, rootQT,
                                                 leBTH, teeBTH, rootBTH,
                                                 leBTV, teeBTV, rootBTV,
                                                 leTTH, teeTTH, rootTTH,
                                                 *dataHandler,
                                                 *splitList);*/

        std::cout << "Score : " << score << "/" << leNP->NB_VALIDATION_TARGETS << ", CU set : ["
                  << CUset->at(0) << ", "
                  << CUset->at(1) << ", "
                  << CUset->at(2) << ", "
                  << CUset->at(3) << ", "
                  << CUset->at(4) << ", "
                  << CUset->at(5) << "], CU chosen : ["
                  << CUchosen->at(0) << ", "
                  << CUchosen->at(1) << ", "
                  << CUchosen->at(2) << ", "
                  << CUchosen->at(3) << ", "
                  << CUchosen->at(4) << ", "
                  << CUchosen->at(5) << "]"
                  << std::endl;
        //std::cout << "  ScoreFct : " << scoreFct << "/" << leNP->NB_VALIDATION_TARGETS << std::endl;
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
        delete CUchosen;
        delete CUset;
    }
    moyenne /= nbEval;
    std::cout << "Score moyen : " << moyenne << "/1000" << std::endl;
    return 0;
}


uint64_t EvaluateLinearWaterFall(BinaryFeaturesEnv* leNP,
                                 TPG::TPGExecutionEngine teeNP,
                                 const TPG::TPGVertex* rootNP,
                                 BinaryFeaturesEnv* leQT,
                                 TPG::TPGExecutionEngine teeQT,
                                 const TPG::TPGVertex* rootQT,
                                 BinaryFeaturesEnv* leBTH,
                                 TPG::TPGExecutionEngine teeBTH,
                                 const TPG::TPGVertex* rootBTH,
                                 BinaryFeaturesEnv* leBTV,
                                 TPG::TPGExecutionEngine teeBTV,
                                 const TPG::TPGVertex* rootBTV,
                                 BinaryFeaturesEnv* leTTH,
                                 TPG::TPGExecutionEngine teeTTH,
                                 const TPG::TPGVertex* rootTTH,
                                 std::vector<Data::PrimitiveTypeArray<double> *>& dataHandler,
                                 std::vector<uint8_t>& splitList)
{
    uint64_t score = 0;
    int actionID;
    int chosenAction = -1;

    // Run NB_VALIDATION_TARGETS times the TPG, each time on a different CU
    for (uint64_t nbCU = 0; nbCU < leNP->NB_VALIDATION_TARGETS; nbCU++)
    {
        // -------------------------- Load next CU for all TPGs --------------------------
        leNP->setCurrentState(*dataHandler.at(nbCU));
        leQT->setCurrentState(*dataHandler.at(nbCU));
        leBTH->setCurrentState(*dataHandler.at(nbCU));
        leBTV->setCurrentState(*dataHandler.at(nbCU));
        leTTH->setCurrentState(*dataHandler.at(nbCU));

        // ************************************** ORDER : NP QT BTH BTV TTH ***************************************
        // -------------------------- Call NP TPG --------------------------
        actionID = (int) ((const TPG::TPGAction *) teeNP.executeFromRoot(* rootNP).back())->getActionID();
        //std::cout << "NP Action : " << actionID << std::endl;
        if(actionID == 0)
            chosenAction = leNP->getActions0().at(0);
        else
        {
            // -------------------------- Call QT TPG --------------------------
            actionID = (int) ((const TPG::TPGAction *) teeQT.executeFromRoot(* rootQT).back())->getActionID();
            //std::cout << "QT Action : " << actionID << std::endl;
            if(actionID == 0)
                chosenAction = leQT->getActions0().at(0);
            else
            {
                // -------------------------- Call BTH TPG --------------------------
                actionID = (int) ((const TPG::TPGAction *) teeBTH.executeFromRoot(* rootBTH).back())->getActionID();
                //std::cout << "BTH Action : " << actionID << std::endl;
                if(actionID == 0)
                    chosenAction = leBTH->getActions0().at(0);
                else
                {
                    // -------------------------- Call BTV TPG --------------------------
                    actionID = (int) ((const TPG::TPGAction *) teeBTV.executeFromRoot(* rootBTV).back())->getActionID();
                    //std::cout << "BTV Action : " << actionID << std::endl;
                    if(actionID == 0)
                        chosenAction = leBTV->getActions0().at(0);
                    else
                    {
                        // -------------------------- Call TTH TPG --------------------------
                        actionID = (int) ((const TPG::TPGAction *) teeTTH.executeFromRoot(* rootTTH).back())->getActionID();
                        //std::cout << "TTH Action : " << actionID << std::endl;
                        if(actionID == 0)
                            chosenAction = leTTH->getActions0().at(0);
                        // TTH
                    } // BTV
                } // BTH
            } // QT
        } // NP

        // -------------------------- Update Score --------------------------
        std::cout << "Action : " << chosenAction << ", real Split : " << (int) splitList.at(nbCU) << std::endl;
        if (chosenAction == (int) splitList.at(nbCU))
            score++;
    }
    return score;
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
