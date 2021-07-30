#include <iostream>
#include <cmath>
#include <cinttypes>
#include <cstdlib>

#include <gegelati.h>

#include "../../include/features/BinaryFeaturesEnv.h"

void importTPG(BinaryFeaturesEnv* le, Environment& env, TPG::TPGGraph& tpg);
Data::PrimitiveTypeArray<double>* getRandomCUFeatures(std::string& datasetPath, BinaryFeaturesEnv* le, std::vector<uint8_t>* splitList);

void ParseAvailableSplitsInVector(std::vector<bool>& actions, std::string str);

void EvaluateAllBinaryParallelFull(size_t seed, uint64_t cuHeight, uint64_t cuWidth, uint64_t nbFeatures,
                                    uint64_t nbDatabaseElements, uint64_t nbTrainingTargets,
                                    uint64_t nbGeneTargetChange, uint64_t nbValidationTarget,
                                    const Instructions::Set& set, Learn::LearningParameters params, std::string datasetPath,
                                    int nbEval, bool debug, std::vector<bool> availableSplits);
void EvaluateDirectionWaterfallSink(size_t seed, uint64_t cuHeight, uint64_t cuWidth, uint64_t nbFeatures,
                                  uint64_t nbDatabaseElements, uint64_t nbTrainingTargets,
                                  uint64_t nbGeneTargetChange, uint64_t nbValidationTarget,
                                  const Instructions::Set& set, Learn::LearningParameters params, std::string datasetPath,
                                  int nbEval, bool debug, std::vector<bool> availableSplits);
void EvaluateLinearWaterfallSink(size_t seed, uint64_t cuHeight, uint64_t cuWidth, uint64_t nbFeatures,
                                  uint64_t nbDatabaseElements, uint64_t nbTrainingTargets,
                                  uint64_t nbGeneTargetChange, uint64_t nbValidationTarget,
                                  const Instructions::Set& set, Learn::LearningParameters params, std::string datasetPath,
                                  int nbEval, bool debug, std::vector<bool> availableSplits);

int main(int argc, char* argv[])
{
    std::cout << "Start VVC Partitionning Optimization with binary features TPGs solution" << std::endl;
    // ******************************************* MAIN ARGUMENTS *******************************************

    // Customizable arguments
    std::string availableSplitsStr = "[]";
    std::vector<bool> availableSplits = {false, false, false, false, false, false};
    size_t seed = 0;
    uint64_t cuHeight = 32;
    uint64_t cuWidth = 32;
    uint64_t nbFeatures = 112;
    uint64_t nbDatabaseElements = 114348*6;
    int nbEval = 50;

    // Debug arguments
    bool debug = false;

    // Const arguments
    uint64_t nbTrainingTargets = 10000;
    uint64_t nbGeneTargetChange = 30;
    uint64_t nbValidationTarget = 1000;

    std::cout << "argc: " << argc << std::endl;
    /*for (int i = 0; i < argc-1; i ++)
        std:: cout << i << ": " << argv[i] << ", ";
    std::cout << argc << ": " << argv[argc] << std::endl;*/
    if (argc == 8)
    {
        ParseAvailableSplitsInVector(availableSplits, (std::string) argv[1]);
        seed = atoi(argv[2]);
        cuHeight = atoi(argv[3]);
        cuWidth = atoi(argv[4]);
        nbFeatures = atoi(argv[5]);
        nbDatabaseElements = atoi(argv[6]);
        nbEval = atoi(argv[7]);
    }
    else
    {
        std::cout << "Arguments were not precised (waiting 8 arguments : availableSplits, seed, cuHeight, cuWidth, nbFeatures, nbDatabaseElements and nbEvaluations). Using default value." << std::endl;
        std::cout << "Example : \"./TPGVVCPartDatabase_binaryFeaturesEnv [0, 1, 2, 3, 4, 5] 0 32 32 112 686088\"" << std::endl ;
    }

    std::cout << std::endl << "---------- Main arguments ----------" << std::endl;
    std::cout << std::setw(13) << "availableSplits (bool):";
    for(auto && availableSplit : availableSplits)
        std::cout << std::setw(4) << availableSplit;
    std::cout << std::endl << std::setw(13) << "seed:" << " " << std::setw(3) << seed << std::endl;
    std::cout << std::setw(13) << "cuHeight:" << " " << std::setw(4) << cuHeight << std::endl;
    std::cout << std::setw(13) << "cuWidth:" << " " << std::setw(4) << cuWidth << std::endl;
    std::cout << std::setw(13) << "nbFeatures:" << " " << std::setw(4) << nbFeatures << std::endl;
    std::cout << std::setw(13) << "nbDTBElements:" << " " << std::setw(4) << nbDatabaseElements << std::endl;
    std::cout << std::setw(13) << "nbEval:" << " " << std::setw(4) << nbEval << std::endl;

    std::cout << std::endl << "Start the training of a TPG based on CU features extraction (CNN)" << std::endl;

    // ************************************************ DATASET PATH ***********************************************
    std::string datasetBasePath = "/media/cleonard/alex/cedric_TPG-VVC/balanced_datasets/";
    std::string datasetMiddlePath = "x";
    std::string datasetEndPath = "_balanced/";
    std::string datasetPath = datasetBasePath += std::to_string(cuHeight)
            += datasetMiddlePath += std::to_string(cuWidth) += datasetEndPath;
    std::cout << datasetPath << std::endl;

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

    // Add those instructions to instruction set
    set.add(*(new Instructions::LambdaInstruction<double, double>(minus_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(mult_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(div_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(max_double)));
    set.add(*(new Instructions::LambdaInstruction<double>(exp_double)));
    set.add(*(new Instructions::LambdaInstruction<double>(ln_double)));
    set.add(*(new Instructions::LambdaInstruction<double, Data::Constant>(multByConst_double)));

    // ******************************************* PARAMETERS AND ENVIRONMENT ******************************************
    // ---------------- Load and initialize parameters from .json file ----------------
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/TPG/params.json", params);

    // EvaluateDirectionWaterfallSink()
    // EvaluateLinearWaterfallSink()
    EvaluateAllBinaryParallelFull(seed, cuHeight, cuWidth, nbFeatures,
                                                nbDatabaseElements, nbTrainingTargets,
                                                nbGeneTargetChange, nbValidationTarget,
                                                set, params, datasetPath, nbEval, debug, availableSplits);
    return 0;
}

void EvaluateAllBinaryParallelFull(size_t seed, uint64_t cuHeight, uint64_t cuWidth, uint64_t nbFeatures,
                                    uint64_t nbDatabaseElements, uint64_t nbTrainingTargets,
                                    uint64_t nbGeneTargetChange, uint64_t nbValidationTarget,
                                    const Instructions::Set& set, Learn::LearningParameters params, std::string datasetPath,
                                    int nbEval, bool debug, std::vector<bool> availableSplits)
{
    // ********************* ONLY FOR INDEPENDANT BINARIES (CASCADE-FULL) *********************
    // Goal: Predict with all TPGs on see how many TPGs said it was their split
    //       If many look if the optimal split is among and if yes, print how many split were selected
    //       +1 if the right split was selected

    std::string const recapFileName("/home/cleonard/dev/TpgVvcPartDatabase/build/InferenceRecapFile.log");
    double moyenneScore = 0.0;
    double moyenneNbSplit = 0.0;

    for(int i = 0; i < nbEval; i++)
    {
        // ************************************************** INSTANCES *************************************************

        // ---------------- Instantiate 6 LearningEnvironments, 6 Environments, 6 TPGGraphs and 6 TPGExecutionEngines ----------------
        auto *leNP  = new BinaryFeaturesEnv({0}, {1,2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                                nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leQT  = new BinaryFeaturesEnv({1}, {0,2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                                nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leBTH = new BinaryFeaturesEnv({2}, {0,1,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                                nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leBTV = new BinaryFeaturesEnv({3}, {0,1,2,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leTTH = new BinaryFeaturesEnv({4}, {0,1,2,3,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leTTV = new BinaryFeaturesEnv({5}, {0,1,2,3,4}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);

        // Instantiate the environment that will embed the LearningEnvironment
        Environment envNP (set, leNP->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envQT (set, leQT->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envBTH(set, leBTH->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envBTV(set, leBTV->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envTTH(set, leTTH->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envTTV(set, leTTV->getDataSources(), params.nbRegisters, params.nbProgramConstant);

        // Instantiate the TPGGraph that we will load
        auto tpgNP  = TPG::TPGGraph(envNP);
        auto tpgQT  = TPG::TPGGraph(envQT);
        auto tpgBTH = TPG::TPGGraph(envBTH);
        auto tpgBTV = TPG::TPGGraph(envBTV);
        auto tpgTTH = TPG::TPGGraph(envTTH);
        auto tpgTTV = TPG::TPGGraph(envTTV);

        // Instantiate the tee that will handle the decisions taken by the TPG
        TPG::TPGExecutionEngine teeNP(envNP);
        TPG::TPGExecutionEngine teeQT(envQT);
        TPG::TPGExecutionEngine teeBTH(envBTH);
        TPG::TPGExecutionEngine teeBTV(envBTV);
        TPG::TPGExecutionEngine teeTTH(envTTH);
        TPG::TPGExecutionEngine teeTTV(envTTV);

        // ---------------- Import TPG graph from .dot file ----------------
        importTPG(leNP,  envNP, tpgNP);
        if(availableSplits[1])
            importTPG(leQT,  envQT, tpgQT);
        if(availableSplits[2])
            importTPG(leBTH, envBTH, tpgBTH);
        if(availableSplits[3])
            importTPG(leBTV, envBTV, tpgBTV);
        if(availableSplits[4])
            importTPG(leTTH, envTTH, tpgTTH);
        if(availableSplits[5])
            importTPG(leTTV, envTTV, tpgTTV);

        // ---------------- Get root from TPG ----------------
        auto rootNP = tpgNP.getRootVertices().front();
        auto rootQT  = availableSplits[1] ? tpgQT.getRootVertices().front()  : nullptr;
        auto rootBTH = availableSplits[2] ? tpgBTH.getRootVertices().front() : nullptr;
        auto rootBTV = availableSplits[3] ? tpgBTV.getRootVertices().front() : nullptr;
        auto rootTTH = availableSplits[4] ? tpgTTH.getRootVertices().front() : nullptr;
        auto rootTTV = availableSplits[5] ? tpgTTV.getRootVertices().front() : nullptr;

        // ************************************************* LOAD DATA *************************************************
        auto *dataHandler = new std::vector<Data::PrimitiveTypeArray<double> *>;
        auto *splitList   = new std::vector<uint8_t>;
        auto *CUchosen    = new std::vector<int>{0,0,0,0,0,0};
        auto *CUset       = new std::vector<int>{0,0,0,0,0,0};

        // Load a vector of 1000 CUs (dataHandler) and their corresponding split (splitList)
        // -------------------------- Load a global vector of 1.000 CUs --------------------------
        for (uint64_t idx_targ = 0; idx_targ < nbValidationTarget; idx_targ++)
        {
            Data::PrimitiveTypeArray<double> *target = getRandomCUFeatures(datasetPath, leNP, splitList);
            dataHandler->push_back(target);
            // Optimal split is stored in splitList inside getRandomCU()
        }

        // ************************************************** MAIN RUN *************************************************
        uint64_t score = 0;
        int actionID;
        auto *chosenSplits = new std::vector<int>;
        auto *howManySplitsChosen = new std::vector<int>{0,0,0,0,0,0,0};

        for (uint64_t nbCU = 0; nbCU < nbValidationTarget; nbCU++)
        {
            // Reset chosen splits vector
            chosenSplits->clear();

            // Load next CU for all TPGs
            leNP->setCurrentState(*dataHandler->at(nbCU));
            leQT->setCurrentState(*dataHandler->at(nbCU));
            leBTH->setCurrentState(*dataHandler->at(nbCU));
            leBTV->setCurrentState(*dataHandler->at(nbCU));
            leTTH->setCurrentState(*dataHandler->at(nbCU));
            leTTV->setCurrentState(*dataHandler->at(nbCU));

            // Call NP
            actionID = (int) ((const TPG::TPGAction *) teeNP.executeFromRoot(* rootNP).back())->getActionID();
            if(actionID == 0)
                chosenSplits->push_back(0); //leNP->getActions0().at(0);

            // Call QT
            if(availableSplits[1])
            {
                actionID = (int) ((const TPG::TPGAction *) teeQT.executeFromRoot(* rootQT).back())->getActionID();
                if(actionID == 0)
                    chosenSplits->push_back(1); //leQT->getActions0().at(0);
            }

            // Call BTH
            if(availableSplits[2])
            {
                actionID = (int) ((const TPG::TPGAction *) teeBTH.executeFromRoot(*rootBTH).back())->getActionID();
                if (actionID == 0)
                    chosenSplits->push_back(2); //leBTH->getActions0().at(0);
            }
            // Call BTV
            if(availableSplits[3])
            {
                actionID = (int) ((const TPG::TPGAction *) teeBTV.executeFromRoot(*rootBTV).back())->getActionID();
                if (actionID == 0)
                    chosenSplits->push_back(3); //leBTV->getActions0().at(0);
            }
            // Call TTH
            if(availableSplits[4])
            {
                actionID = (int) ((const TPG::TPGAction *) teeTTH.executeFromRoot(* rootTTH).back())->getActionID();
                if(actionID == 0)
                    chosenSplits->push_back(4); //leTTH->getActions0().at(0);
            }
            // Call TTV
            if(availableSplits[5])
            {
                actionID = (int) ((const TPG::TPGAction *) teeTTV.executeFromRoot(*rootTTV).back())->getActionID();
                if (actionID == 0)
                    chosenSplits->push_back(5); //leTTV->getActions0().at(0);
            }
            // Debug printing
            if(debug)
                std::cout << "          chosen Splits : {";

            // **** Compute Score and Update CUchosen ****
            for(uint j = 0; j < chosenSplits->size(); j++)
            {
                if(chosenSplits->at(j) == (int) splitList->at(nbCU))
                    score ++;
                CUchosen->at(chosenSplits->at(j))++;
                // Debug printing
                if(debug && j != chosenSplits->size())
                    std::cout << chosenSplits->at(j) << ", ";
            }
            howManySplitsChosen->at(chosenSplits->size()) ++;
            CUset->at((int) splitList->at(nbCU)) ++;

            // Debug printing
            if(debug)
                std::cout << "}" << std::endl;
        } // End 1000 CUs Loop

        // Compute moyNbSplit
        double moyNbSplit = 0;
        for(uint j = 0; j < howManySplitsChosen->size(); j++)
            moyNbSplit += j*howManySplitsChosen->at(j);
        moyNbSplit /= nbValidationTarget;

        // ---------------- Update averages ----------------
        moyenneNbSplit += moyNbSplit;
        moyenneScore += (double) score;

        // ---------------- Print Result ----------------
        std::cout << "Score : " << score << "/" << nbValidationTarget << std::endl;
        std::cout << "    CU set : ["
                  << CUset->at(0) << ", "
                  << CUset->at(1) << ", "
                  << CUset->at(2) << ", "
                  << CUset->at(3) << ", "
                  << CUset->at(4) << ", "
                  << CUset->at(5) << "] " << std::endl;
        std::cout << "    chosen : ["
                  << CUchosen->at(0) << ", "
                  << CUchosen->at(1) << ", "
                  << CUchosen->at(2) << ", "
                  << CUchosen->at(3) << ", "
                  << CUchosen->at(4) << ", "
                  << CUchosen->at(5) << "] " << std::endl;
        std::cout << "    HowManySplits chosen : ["
                  << howManySplitsChosen->at(0) << ", "
                  << howManySplitsChosen->at(1) << ", "
                  << howManySplitsChosen->at(2) << ", "
                  << howManySplitsChosen->at(3) << ", "
                  << howManySplitsChosen->at(4) << ", "
                  << howManySplitsChosen->at(5) << "],"
                  << " en moyenne : " << moyNbSplit << std::endl;

        // ---------------- Clean ----------------
        // LearningEnvironments
        delete leNP; delete leQT; delete leBTH; delete leBTV; delete leTTH; delete leTTV;
        // Data, solution handlers and stats stores
        delete dataHandler; delete splitList; delete CUchosen; delete CUset;
    } // End nbEval Loop

    // ---------------- Compute and Print global result ----------------
    moyenneScore /= nbEval;
    moyenneNbSplit /= nbEval;
    std::cout << "Score moyen : " << moyenneScore << "/1000" << std::endl;
    std::cout << "Nombre de splits sélectionnés moyen : " << moyenneNbSplit << std::endl;

    // Store result in file
    std::ofstream file(recapFileName.c_str(), std::ios::app);
    if (file)
    {
        file << round(moyenneScore*100)/100 << " " << round(moyenneNbSplit * 1000) / 1000 << std::endl;
    }
}

void EvaluateDirectionWaterfallSink(size_t seed, uint64_t cuHeight, uint64_t cuWidth, uint64_t nbFeatures,
                                  uint64_t nbDatabaseElements, uint64_t nbTrainingTargets,
                                  uint64_t nbGeneTargetChange, uint64_t nbValidationTarget,
                                  const Instructions::Set& set, Learn::LearningParameters params, std::string datasetPath,
                                  int nbEval, bool debug, std::vector<bool> availableSplits)
{
    double moyenneScore = 0.0;

    for(int i = 0; i < nbEval; i++)
    {
        // ************************************************** INSTANCES *************************************************

        // ---------------- Instantiate 6 LearningEnvironments, 6 Environments, 6 TPGGraphs and 6 TPGExecutionEngines ----------------
        auto *leNP  = new BinaryFeaturesEnv({0}, {1,2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                           nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leQT  = new BinaryFeaturesEnv({1}, {2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                           nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leBTH = new BinaryFeaturesEnv({2,4}, {3,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leBTV = new BinaryFeaturesEnv({2}, {4}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leTTH = new BinaryFeaturesEnv({3}, {5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);

        // Instantiate the environment that will embed the LearningEnvironment
        Environment envNP(set,  leNP->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envQT(set,  leQT->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envBTH(set, leBTH->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envBTV(set, leBTV->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envTTH(set, leTTH->getDataSources(), params.nbRegisters, params.nbProgramConstant);

        // Instantiate the TPGGraph that we will load
        auto tpgNP  = TPG::TPGGraph(envNP);
        auto tpgQT  = TPG::TPGGraph(envQT);
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
        importTPG(leNP,  envNP, tpgNP);
        importTPG(leQT,  envQT, tpgQT);
        importTPG(leBTH, envBTH, tpgBTH);
        importTPG(leBTV, envBTV, tpgBTV);
        importTPG(leTTH, envTTH, tpgTTH);

        // ---------------- Execute TPG from root on the environment ----------------
        auto rootNP  = tpgNP.getRootVertices().front();
        auto rootQT  = tpgQT.getRootVertices().front();
        auto rootBTH = tpgBTH.getRootVertices().front();
        auto rootBTV = tpgBTV.getRootVertices().front();
        auto rootTTH = tpgTTH.getRootVertices().front();

        // ************************************************* LOAD DATA *************************************************
        auto *dataHandler = new std::vector<Data::PrimitiveTypeArray<double> *>;
        auto *splitList   = new std::vector<uint8_t>;
        auto *CUchosen    = new std::vector<int>{0,0,0,0,0,0};
        auto *CUset       = new std::vector<int>{0,0,0,0,0,0};

        // Load a vector of 1000 CUs (dataHandler) and their corresponding split (splitList)
        // -------------------------- Load a global vector of 1.000 CUs --------------------------
        for (uint64_t idx_targ = 0; idx_targ < nbValidationTarget; idx_targ++)
        {
            Data::PrimitiveTypeArray<double> *target = getRandomCUFeatures(datasetPath, leNP, splitList);
            dataHandler->push_back(target);
            // Optimal split is stored in splitList inside getRandomCU()
        }

        // ************************************************** MAIN RUN *************************************************
        uint64_t score = 0;
        int actionID;
        int chosenAction = -1;

        for (uint64_t nbCU = 0; nbCU < nbValidationTarget; nbCU++)
        {   // -------------------------- Load next CU for all TPGs --------------------------
            leNP->setCurrentState(*dataHandler->at(nbCU));
            leQT->setCurrentState(*dataHandler->at(nbCU));
            leBTH->setCurrentState(*dataHandler->at(nbCU));
            leBTV->setCurrentState(*dataHandler->at(nbCU));
            leTTH->setCurrentState(*dataHandler->at(nbCU));

            // ************************************ ORDER : NP QT DIREC HORI VERTI *************************************
            // -------------------------- Call NP TPG --------------------------
            actionID = (int) ((const TPG::TPGAction *) teeNP.executeFromRoot(* rootNP).back())->getActionID();
            if(actionID == 0)
                chosenAction = 0; //leNP->getActions0().at(0);
            else
            {  // -------------------------- Call QT TPG --------------------------
                actionID = (int) ((const TPG::TPGAction *) teeQT.executeFromRoot(* rootQT).back())->getActionID();
                if(actionID == 0)
                    chosenAction = 1; //leQT->getActions0().at(0);
                else
                {  // -------------------------- Call DIREC TPG --------------------------
                    actionID = (int) ((const TPG::TPGAction *) teeBTH.executeFromRoot(* rootBTH).back())->getActionID();
                    if(actionID == 0)
                    {  // -------------------------- Call HORI TPG --------------------------
                        actionID = (int) ((const TPG::TPGAction *) teeBTV.executeFromRoot(* rootBTV).back())->getActionID();
                        if(actionID == 0)
                            chosenAction = 2; //leBTV->getActions0().at(0);
                        else
                            chosenAction = 4; //leBTV->getActions1().at(0);
                    }
                    else
                    {  // -------------------------- Call VERTI TPG --------------------------
                        actionID = (int) ((const TPG::TPGAction *) teeTTH.executeFromRoot(* rootTTH).back())->getActionID();
                        if(actionID == 0)
                            chosenAction = 3; //leTTH->getActions0().at(0);
                        else
                            chosenAction = 5; //leTTH->getActions1().at(0);
                    } // DIREC
                } // QT
            } // NP
            // **** Debug Printing ****
            if(debug)
                std::cout << "Action : " << chosenAction << ", real Split : " << (int) splitList->at(nbCU) << std::endl;
            // -------------------------- Update Score --------------------------
            if (chosenAction == (int) splitList->at(nbCU))
                score++;
            CUchosen->at(chosenAction) ++; CUset->at((int) splitList->at(nbCU)) ++;
        }

        // ---------------- Print Result ----------------
        std::cout << "Score : " << score << "/" << nbValidationTarget << std::endl;
        std::cout << "    CU set : ["
                  << CUset->at(0) << ", "
                  << CUset->at(1) << ", "
                  << CUset->at(2) << ", "
                  << CUset->at(3) << ", "
                  << CUset->at(4) << ", "
                  << CUset->at(5) << "] " << std::endl;
        std::cout << "    chosen : ["
                  << CUchosen->at(0) << ", "
                  << CUchosen->at(1) << ", "
                  << CUchosen->at(2) << ", "
                  << CUchosen->at(3) << ", "
                  << CUchosen->at(4) << ", "
                  << CUchosen->at(5) << "] " << std::endl;
        moyenneScore += (double) score;

        // ---------------- Clean ----------------
        // LearningEnvironments
        delete leNP; delete leQT; delete leBTH; delete leBTV; delete leTTH;
        // Data, solution handlers and stats stores
        delete dataHandler; delete splitList; delete CUchosen; delete CUset;
    } // End nbEval Loop

    // ---------------- Compute and Print global result ----------------
    moyenneScore /= nbEval;
    std::cout << "Score moyen : " << moyenneScore << "/1000" << std::endl;
}

void EvaluateLinearWaterfallSink(size_t seed, uint64_t cuHeight, uint64_t cuWidth, uint64_t nbFeatures,
                                  uint64_t nbDatabaseElements, uint64_t nbTrainingTargets,
                                  uint64_t nbGeneTargetChange, uint64_t nbValidationTarget,
                                  const Instructions::Set& set, Learn::LearningParameters params, std::string datasetPath,
                                  int nbEval, bool debug, std::vector<bool> availableSplits)
{
    double moyenneScore = 0.0;

    for(int i = 0; i < nbEval; i++)
    {
        // ************************************************** INSTANCES *************************************************
        // ---------------- Instantiate 6 LearningEnvironments, 6 Environments, 6 TPGGraphs and 6 TPGExecutionEngines ----------------
        auto *leNP  = new BinaryFeaturesEnv({0}, {1,2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leQT  = new BinaryFeaturesEnv({1}, {2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leBTH = new BinaryFeaturesEnv({2}, {3,4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leBTV = new BinaryFeaturesEnv({3}, {4,5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
        auto *leTTH = new BinaryFeaturesEnv({4}, {5}, seed, cuHeight, cuWidth, nbFeatures,
                                            nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);

        // Instantiate the environment that will embed the LearningEnvironment
        Environment envNP(set,  leNP->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envQT(set,  leQT->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envBTH(set, leBTH->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envBTV(set, leBTV->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envTTH(set, leTTH->getDataSources(), params.nbRegisters, params.nbProgramConstant);

        // Instantiate the TPGGraph that we will load
        auto tpgNP  = TPG::TPGGraph(envNP);
        auto tpgQT  = TPG::TPGGraph(envQT);
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
        importTPG(leNP,  envNP, tpgNP);
        importTPG(leQT,  envQT, tpgQT);
        importTPG(leBTH, envBTH, tpgBTH);
        importTPG(leBTV, envBTV, tpgBTV);
        importTPG(leTTH, envTTH, tpgTTH);

        // ---------------- Execute TPG from root on the environment ----------------
        auto rootNP  = tpgNP.getRootVertices().front();
        auto rootQT  = tpgQT.getRootVertices().front();
        auto rootBTH = tpgBTH.getRootVertices().front();
        auto rootBTV = tpgBTV.getRootVertices().front();
        auto rootTTH = tpgTTH.getRootVertices().front();

        // ************************************************* LOAD DATA *************************************************
        auto *dataHandler = new std::vector<Data::PrimitiveTypeArray<double> *>;
        auto *splitList   = new std::vector<uint8_t>;
        auto *CUchosen    = new std::vector<int>{0,0,0,0,0,0};
        auto *CUset       = new std::vector<int>{0,0,0,0,0,0};

        // Load a vector of 1000 CUs (dataHandler) and their corresponding split (splitList)
        // -------------------------- Load a global vector of 1.000 CUs --------------------------
        for (uint64_t idx_targ = 0; idx_targ < nbValidationTarget; idx_targ++)
        {
            Data::PrimitiveTypeArray<double> *target = getRandomCUFeatures(datasetPath, leNP, splitList);
            dataHandler->push_back(target);
            // Optimal split is stored in splitList inside getRandomCU()
        }

        // ************************************************** MAIN RUN *************************************************
        uint64_t score = 0;
        int actionID;
        int chosenAction = -1;

        for (uint64_t nbCU = 0; nbCU < nbValidationTarget; nbCU++)
        {   // -------------------------- Load next CU for all TPGs --------------------------
            leNP->setCurrentState(*dataHandler->at(nbCU));
            leQT->setCurrentState(*dataHandler->at(nbCU));
            leBTH->setCurrentState(*dataHandler->at(nbCU));
            leBTV->setCurrentState(*dataHandler->at(nbCU));
            leTTH->setCurrentState(*dataHandler->at(nbCU));

            // ************************************** ORDER : NP QT BTH BTV TTH ***************************************
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
                    // -------------------------- Call BTH TPG --------------------------
                    actionID = (int) ((const TPG::TPGAction *) teeBTH.executeFromRoot(* rootBTH).back())->getActionID();
                    //std::cout << "BTH Action : " << actionID << std::endl;
                    if(actionID == 0)
                        chosenAction = 2; //leBTH->getActions0().at(0);
                    else
                    {
                        // -------------------------- Call BTV TPG --------------------------
                        actionID = (int) ((const TPG::TPGAction *) teeBTV.executeFromRoot(* rootBTV).back())->getActionID();
                        //std::cout << "BTV Action : " << actionID << std::endl;
                        if(actionID == 0)
                            chosenAction = 3; //leBTV->getActions0().at(0);
                        else
                        {
                            // -------------------------- Call TTH TPG --------------------------
                            actionID = (int) ((const TPG::TPGAction *) teeTTH.executeFromRoot(* rootTTH).back())->getActionID();
                            //std::cout << "TTH Action : " << actionID << std::endl;
                            if(actionID == 0)
                                chosenAction = 4; //leTTH->getActions0().at(0);
                                // TTH
                            else
                                chosenAction = 5;
                                // TTV
                        } // BTV
                    } // BTH
                } // QT
            } // NP

            // **** Debug Printing ****
            if(debug)
                std::cout << "Action : " << chosenAction << ", real Split : " << (int) splitList->at(nbCU) << std::endl;
            // -------------------------- Update Score --------------------------
            if (chosenAction == (int) splitList->at(nbCU))
                score++;
            CUchosen->at(chosenAction) ++; CUset->at((int) splitList->at(nbCU)) ++;
        }

        // ---------------- Print Result ----------------
        std::cout << "Score : " << score << "/" << nbValidationTarget << std::endl;
        std::cout << "    CU set : ["
                  << CUset->at(0) << ", "
                  << CUset->at(1) << ", "
                  << CUset->at(2) << ", "
                  << CUset->at(3) << ", "
                  << CUset->at(4) << ", "
                  << CUset->at(5) << "] " << std::endl;
        std::cout << "    chosen : ["
                  << CUchosen->at(0) << ", "
                  << CUchosen->at(1) << ", "
                  << CUchosen->at(2) << ", "
                  << CUchosen->at(3) << ", "
                  << CUchosen->at(4) << ", "
                  << CUchosen->at(5) << "] " << std::endl;
        moyenneScore += (double) score;

        // ---------------- Clean ----------------
        // LearningEnvironments
        delete leNP; delete leQT; delete leBTH; delete leBTV; delete leTTH;
        // Data, solution handlers and stats stores
        delete dataHandler; delete splitList; delete CUchosen; delete CUset;
    } // End nbEval Loop

    // ---------------- Compute and Print global result ----------------
    moyenneScore /= nbEval;
    std::cout << "Score moyen : " << moyenneScore << "/1000" << std::endl;
}

void importTPG(BinaryFeaturesEnv* le, Environment& env, TPG::TPGGraph& tpg)
{
    try{
        char tpgPath[100] = ROOT_DIR"/TPG/";
        std::string speActionName = le->getActionName(le->getActions0().at(0));
        std::strcat(tpgPath, speActionName.c_str());
        char dotExtension[10] = ".dot";
        std::strcat(tpgPath, dotExtension);
        //std::cout << "Import TPG from : \"" << tpgPath << "\"" << std::endl;
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

void ParseAvailableSplitsInVector(std::vector<bool>& actions, std::string str)
{
    std::vector<uint8_t> availableActions;

    // This condition below is not necessary in normal execution conditions
    // But when calling the executable from a script, bash force the extern quotes as part of the string
    // Which fucked up the whole process below. So we erase these quotes when necessary
    if(str[0] == '"' && str[str.size() - 1] == '"')
    {
        str.erase(0, 1);
        str.erase(str.size() - 1);
    }

    // Remove first ('[{]') and last char (']') from str
    //std::cout << str.size() << " : |" << str << "|" << std::endl;
    str.erase(0, 1);
    str.erase(str.size() - 1);
    //std::cout << str.size() << " : |" << str << "|" << std::endl;

    if (str.size() == 1)
        actions.push_back((uint8_t) atoi(str.c_str()));
    else
    {
        while(!str.empty())
        {
            // Using stringstream to convert string into int
            int nb;
            std::stringstream ss;
            ss << str[0];
            ss >> nb;

            availableActions.push_back((uint8_t) nb);
            str.erase(0,3); // 'X, '
            //std::cout << str.size() << " : |" << str << "|" << std::endl;
        }
    }

    std::cout << std::endl << std::setw(13) << "availableSplits:";
    for (auto &act : availableActions)
    {
        std::cout << std::setw(4) << (int) act;
        for(int i = 0; i < 6; i++)
            if(act == i)
                actions[i] = true;
    }
    std::cout << std::endl;
}
