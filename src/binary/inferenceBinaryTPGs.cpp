#include <iostream>
#include <cmath>
#include <thread>
#include <atomic>
#include <cinttypes>
#include <cstdlib>

#include <gegelati.h>

#include "../../include/binary/defaultBinaryEnv.h"
#include "../../include/binary/classBinaryEnv.h"

void importTPG(BinaryClassifEnv* le, Environment& env, TPG::TPGGraph& tpg);
Data::PrimitiveTypeArray2D<uint8_t>* getRandomCU(const char datasetPath[100], BinaryClassifEnv* le, std::vector<uint8_t>* splitList, uint64_t index_targ);
void runOneTPG(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, BinaryClassifEnv* le);

int main(int argc, char* argv[])
{
    std::cout << "Start VVC Partitionning Optimization with binary TPGs solution" << std::endl;
    int nbEval = 50;
    std::cout << nbEval << " evaluations" << std::endl;
    double moyenne = 0.0;
    for(int i = 0; i < nbEval; i++)
    {
        // ************************************************** INSTRUCTIONS *************************************************

        // Create the instruction set for programs
        Instructions::Set set;

        // uint8_t instructions (for pixels values)
        auto minus = [](uint8_t a, uint8_t b) -> double { return a - b; };
        auto add = [](uint8_t a, uint8_t b) -> double { return a + b; };
        auto mult = [](uint8_t a, uint8_t b) -> double { return a * b; };
        auto div = [](uint8_t a, uint8_t b) -> double {
            return a / (double) b;
        }; // cast b to double to avoid div by zero (uint8_t)
        auto max = [](uint8_t a, uint8_t b) -> double { return std::max(a, b); };
        auto multByConst = [](uint8_t a, Data::Constant c) -> double { return a * (double) c; };

        // double instructions (for TPG programs)
        auto minus_double = [](double a, double b) -> double { return a - b; };
        auto add_double = [](double a, double b) -> double { return a + b; };
        auto mult_double = [](double a, double b) -> double { return a * b; };
        auto div_double = [](double a, double b) -> double { return a / b; };
        auto max_double = [](double a, double b) -> double { return std::max(a, b); };
        auto ln_double = [](double a) -> double { return std::log(a); };
        auto exp_double = [](double a) -> double { return std::exp(a); };
        auto multByConst_double = [](double a, Data::Constant c) -> double { return a * (double) c; };
        auto conv2D_double = [](const Data::Constant coeff[9], const uint8_t data[3][3]) -> double {
            double res = 0.0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    res += (double) coeff[i * 3 + j] * data[i][j];
                }
            }
            return res;
        };

        // Add those instructions to instruction set
        set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(minus)));
        set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(add)));
        set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(mult)));
        set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(div)));
        set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(max)));
        set.add(*(new Instructions::LambdaInstruction<uint8_t, Data::Constant>(multByConst)));
        set.add(*(new Instructions::LambdaInstruction<double, double>(minus_double)));
        set.add(*(new Instructions::LambdaInstruction<double, double>(add_double)));
        set.add(*(new Instructions::LambdaInstruction<double, double>(mult_double)));
        set.add(*(new Instructions::LambdaInstruction<double, double>(div_double)));
        set.add(*(new Instructions::LambdaInstruction<double, double>(max_double)));
        set.add(*(new Instructions::LambdaInstruction<double>(exp_double)));
        set.add(*(new Instructions::LambdaInstruction<double>(ln_double)));
        set.add(*(new Instructions::LambdaInstruction<double, Data::Constant>(multByConst_double)));
        set.add(*(new Instructions::LambdaInstruction<const Data::Constant[9], const uint8_t[3][3]>(conv2D_double)));

        // ******************************************* PARAMETERS AND ENVIRONMENT ******************************************

        // ---------------- Load and initialize parameters from .json file ----------------
        Learn::LearningParameters params;
        File::ParametersParser::loadParametersFromJson(ROOT_DIR "/TPG/params.json", params);

        // Initialising the number of CUs used
        uint64_t nbTrainingElements = 110000;   // Balanced database with 55000 elements of one class and 11000 elements of each other class : 110000
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
        auto *leNP = new BinaryClassifEnv({0, 1}, 0, nbTrainingElements, nbTrainingTargets, nbGeneTargetChange,
                                          nbValidationTarget, 0);
        auto *leQT = new BinaryClassifEnv({0, 1}, 1, nbTrainingElements, nbTrainingTargets, nbGeneTargetChange,
                                          nbValidationTarget, 0);
        auto *leBTH = new BinaryClassifEnv({0, 1}, 2, nbTrainingElements, nbTrainingTargets, nbGeneTargetChange,
                                           nbValidationTarget, 0);
        auto *leBTV = new BinaryClassifEnv({0, 1}, 3, nbTrainingElements, nbTrainingTargets, nbGeneTargetChange,
                                           nbValidationTarget, 0);
        //auto *leTTH = new BinaryClassifEnv({0, 1}, 4, nbTrainingElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget, 0);
        auto *leTTV = new BinaryClassifEnv({0, 1}, 5, nbTrainingElements, nbTrainingTargets, nbGeneTargetChange,
                                           nbValidationTarget, 0);

        // Instantiate the environment that will embed the LearningEnvironment
        Environment envNP(set, leNP->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envQT(set, leQT->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envBTH(set, leBTH->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envBTV(set, leBTV->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        //Environment envTTH(set, leTTH->getDataSources(), params.nbRegisters, params.nbProgramConstant);
        Environment envTTV(set, leTTV->getDataSources(), params.nbRegisters, params.nbProgramConstant);

        // Instantiate the TPGGraph that we will load
        auto tpgNP = TPG::TPGGraph(envNP);
        auto tpgQT = TPG::TPGGraph(envQT);
        auto tpgBTH = TPG::TPGGraph(envBTH);
        auto tpgBTV = TPG::TPGGraph(envBTV);
        //auto tpgTTH = TPG::TPGGraph(envTTH);
        auto tpgTTV = TPG::TPGGraph(envTTV);

        // Instantiate the tee that will handle the decisions taken by the TPG
        TPG::TPGExecutionEngine teeNP(envNP);
        TPG::TPGExecutionEngine teeQT(envQT);
        TPG::TPGExecutionEngine teeBTH(envBTH);
        TPG::TPGExecutionEngine teeBTV(envBTV);
        //TPG::TPGExecutionEngine teeTTH(envTTH);
        TPG::TPGExecutionEngine teeTTV(envTTV);

        // ---------------- Import TPG graph from .dot file ----------------
        importTPG(leNP, envNP, tpgNP);
        importTPG(leQT, envQT, tpgQT);
        importTPG(leBTH, envBTH, tpgBTH);
        importTPG(leBTV, envBTV, tpgBTV);
        //importTPG(leTTH, envTTH, tpgTTH);
        importTPG(leTTV, envTTV, tpgTTV);

        // ---------------- Execute TPG from root on the environment ----------------
        auto rootNP = tpgNP.getRootVertices().front();
        auto rootQT = tpgQT.getRootVertices().front();
        auto rootBTH = tpgBTH.getRootVertices().front();
        auto rootBTV = tpgBTV.getRootVertices().front();
        //auto rootTTH = tpgTTH.getRootVertices().front();
        auto rootTTV = tpgTTV.getRootVertices().front();

        // ************************************************** MAIN RUN *************************************************
        // Path to the global database with 330.000 elements (55.000 of each class)
        char datasetPath[100] = "/home/cleonard/Data/dataset_tpg_balanced/dataset_tpg_32x32_27_balanced2/";

        auto *dataHandler = new std::vector<Data::PrimitiveTypeArray2D<uint8_t> *>;
        auto *splitList = new std::vector<uint8_t>;

        // Load a vector of 1000 CUs (dataHandler) and their corresponding split (splitList)
        // -------------------------- Load a global vector of 1.000 CUs --------------------------
        for (uint64_t idx_targ = 0; idx_targ < leNP->NB_VALIDATION_TARGETS; idx_targ++) {
            Data::PrimitiveTypeArray2D<uint8_t> *target = getRandomCU(datasetPath, leNP, splitList, idx_targ);
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
            leNP->setCurrentCu(*dataHandler->at(nbCU));
            leQT->setCurrentCu(*dataHandler->at(nbCU));
            leBTH->setCurrentCu(*dataHandler->at(nbCU));
            leBTV->setCurrentCu(*dataHandler->at(nbCU));
            //leTTH->setCurrentCu(*dataHandler->at(nbCU));
            leTTV->setCurrentCu(*dataHandler->at(nbCU));

            /*// ************************************** ORDER : NP QT BTH BTV TTH TTV ***************************************
            // -------------------------- Call NP TPG --------------------------
            actionID = (int) ((const TPG::TPGAction *) teeNP.executeFromRoot(* rootNP).back())->getActionID();
            //std::cout << "NP Action : " << actionID << std::endl;
            if(actionID == 1)
                chosenAction = leNP->getSpecializedAction();
            else
            {
                // -------------------------- Call QT TPG --------------------------
                actionID = (int) ((const TPG::TPGAction *) teeQT.executeFromRoot(* rootQT).back())->getActionID();
                //std::cout << "QT Action : " << actionID << std::endl;
                if(actionID == 1)
                    chosenAction = leQT->getSpecializedAction();
                else
                {
                    // -------------------------- Call BTH TPG --------------------------
                    actionID = (int) ((const TPG::TPGAction *) teeBTH.executeFromRoot(* rootBTH).back())->getActionID();
                    //std::cout << "BTH Action : " << actionID << std::endl;
                    if(actionID == 1)
                        chosenAction = leBTH->getSpecializedAction();
                    else
                    {
                        // -------------------------- Call BTV TPG --------------------------
                        actionID = (int) ((const TPG::TPGAction *) teeBTV.executeFromRoot(* rootBTV).back())->getActionID();
                        //std::cout << "BTV Action : " << actionID << std::endl;
                        if(actionID == 1)
                            chosenAction = leBTV->getSpecializedAction();
                        else
                        {
                            // -------------------------- Call TTH TPG --------------------------
                            actionID = (int) ((const TPG::TPGAction *) teeTTH.executeFromRoot(* rootTTH).back())->getActionID();
                            //std::cout << "TTH Action : " << actionID << std::endl;
                            if(actionID == 1)
                                chosenAction = leTTH->getSpecializedAction();
                            else
                            {
                                // -------------------------- Call TTV TPG --------------------------
                                actionID = (int) ((const TPG::TPGAction *) teeTTV.executeFromRoot(* rootTTV).back())->getActionID();
                                //std::cout << "TTV Action : " << actionID << std::endl;
                                if(actionID == 1)
                                    chosenAction = leTTV->getSpecializedAction();
                                 // TTV
                            } // TTH
                        } // BTV
                    } // BTH
                } // QT
            } // NP*/

            // ************************************** ORDER : TTV NP QT BTH BTV TTH ***************************************
            // -------------------------- Call TTV TPG --------------------------
            actionID = (int) ((const TPG::TPGAction *) teeTTV.executeFromRoot(*rootTTV).back())->getActionID();
            //std::cout << "TTV Action : " << actionID << std::endl;
            if (actionID == 1)
                chosenAction = leTTV->getSpecializedAction();
            else {
                // -------------------------- Call NP TPG --------------------------
                actionID = (int) ((const TPG::TPGAction *) teeNP.executeFromRoot(*rootNP).back())->getActionID();
                //std::cout << "NP Action : " << actionID << std::endl;
                if (actionID == 1)
                    chosenAction = leNP->getSpecializedAction();
                else {
                    // -------------------------- Call QT TPG --------------------------
                    actionID = (int) ((const TPG::TPGAction *) teeQT.executeFromRoot(*rootQT).back())->getActionID();
                    //std::cout << "QT Action : " << actionID << std::endl;
                    if (actionID == 1)
                        chosenAction = leQT->getSpecializedAction();
                    else {
                        // -------------------------- Call BTH TPG --------------------------
                        actionID = (int) ((const TPG::TPGAction *) teeBTH.executeFromRoot(*rootBTH).back())->getActionID();
                        //std::cout << "BTH Action : " << actionID << std::endl;
                        if (actionID == 1)
                            chosenAction = leBTH->getSpecializedAction();
                        else {
                            // -------------------------- Call BTV / TTH TPG --------------------------
                            actionID = (int) ((const TPG::TPGAction *) teeBTV.executeFromRoot(
                                    *rootBTV).back())->getActionID();
                            //std::cout << "BTV Action : " << actionID << std::endl;
                            if (actionID == 1)
                                chosenAction = leBTV->getSpecializedAction(); //BTV
                            else
                                chosenAction = 4;   // TTH
                            /*else
                            {
                                // -------------------------- Call TTH TPG --------------------------
                                actionID = (int) ((const TPG::TPGAction *) teeTTH.executeFromRoot(* rootTTH).back())->getActionID();
                                //std::cout << "TTH Action : " << actionID << std::endl;
                                if(actionID == 1)
                                    chosenAction = leTTH->getSpecializedAction();
                                // TTH
                            } // BTV*/
                        } // BTH
                    } // QT
                } // NP
            } // TTV

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
        //delete leTTH;
        delete leTTV;
        // Data and solution handlers
        delete dataHandler;
        delete splitList;
    }
    moyenne /= nbEval;
    std::cout << "Score moyen : " << moyenne << "/1000" << std::endl;
    return 0;
}

void importTPG(BinaryClassifEnv* le, Environment& env, TPG::TPGGraph& tpg)
{
    try{
        char tpgPath[100] = ROOT_DIR"/TPG/";
        std::string speActionName = le->getActionName(le->getSpecializedAction());
        std::strcat(tpgPath, speActionName.c_str());
        char dotExtension[10] = ".dot";
        std::strcat(tpgPath, dotExtension);
        File::TPGGraphDotImporter dotImporter(tpgPath, env, tpg);
    }catch (std::runtime_error& e){
        std::cout << e.what() << std::endl;
    }
}

Data::PrimitiveTypeArray2D<uint8_t>* getRandomCU(const char datasetPath[100], BinaryClassifEnv* le, std::vector<uint8_t>* splitList, uint64_t index_targ)
{
    // ------------------ Opening and Reading a random CU file ------------------
    // Generate the path for a random CU
    uint32_t next_CU_number = rand() % (int) le->NB_TRAINING_ELEMENTS; //le->getRng().getInt32(0, (int) le->NB_TRAINING_ELEMENTS - 1);
    if(index_targ == 0)
        std::cout << "next_CU_number : " << next_CU_number << std::endl;
    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number);
    char CU_path[100];
    std::strcpy(CU_path, datasetPath);
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

    // Store the corresponding optimal split
    splitList->push_back(contents[1024]);

    return randomCU;
}

void runOneTPG(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, BinaryClassifEnv* le)
{
    // Get database path
    char datasetPath[100] = "/home/cleonard/Data/binary_datasets/";
    std::string speActionName = le->getActionName(le->getSpecializedAction());
    std::strcat(datasetPath, speActionName.c_str());
    char dataset_extension[10] = "_dataset/";
    std::strcat(datasetPath, dataset_extension);

    // Initialize score to 0
    uint64_t score = 0;

    // Load NB_VALIDATION_TARGETS CUs from the database
    for(uint64_t idx_targ = 0; idx_targ < le->NB_VALIDATION_TARGETS; idx_targ++)
    {
        Data::PrimitiveTypeArray2D<uint8_t>* target = le->getRandomCU(Learn::LearningMode::VALIDATION, datasetPath);
        BinaryClassifEnv::validationTargetsCU->push_back(target);
        // Optimal split is stored in validationTargetsOptimalSplits inside getRandomCU()
    }

    // Update the LearningEnvironment mode in VALIDATION and load first CU
    le->reset(0, Learn::LearningMode::VALIDATION);

    // Run NB_VALIDATION_TARGETS times the TPG, each time on a different CU
    for(uint64_t idx_targ = 0; idx_targ < le->NB_VALIDATION_TARGETS; idx_targ++)
    {
        // Gets the action the TPG would decide with this CU (Either 0: OTHER actions or 1: SPECIFIED action)
        uint64_t actionID=((const TPG::TPGAction *) tee.executeFromRoot(* root).back())->getActionID();
        //std::cout << "TPG: " << actionID << " Sol: " << (uint64_t) le->getOptimalSplit() << std::endl;

        // Update score
        if(actionID == (uint64_t) le->getOptimalSplit())
            score++;

        // Load next CU
        le->LoadNextCU();
    }
    // Print the results
    std::cout << "Score : " << score << "/" << le->NB_VALIDATION_TARGETS << std::endl;
}
