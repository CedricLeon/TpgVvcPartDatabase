#include <iostream>
#include <fstream>

//#define _USE_MATH_DEFINES // To get M_PI

#include <cmath>
#include <thread>
#include <atomic>
#include <cinttypes>

#include <gegelati.h>

#include "../include/PartCU.h"

#ifndef NB_GENERATIONS
#define NB_GENERATIONS 2000
#endif

void getKey(std::atomic<bool>& exit)
{
    std::cout << std::endl;
    std::cout << "Press `q` then [Enter] to exit." << std::endl;
    std::cout.flush();

    exit = false;

    while (!exit) {
        char c;
        std::cin >> c;
        switch (c) {
        case 'q':
        case 'Q':
            exit = true;
            break;
        default:
            printf("Invalid key '%c' pressed.", c);
            std::cout.flush();
        }
    }

    printf("Program will terminate at the end of next generation.\n");
    std::cout.flush();
}

int main()
{
    std::cout << "Start VVC Partitionning with TPG application." << std::endl;

    // Create the instruction set for programs
    Instructions::Set set;
    auto minus = [](uint8_t a, uint8_t b)->double {return a - b; };
    auto add   = [](uint8_t a, uint8_t b)->double {return a + b; };
    auto mult  = [](uint8_t a, uint8_t b)->double {return a * b; };
    auto div   = [](uint8_t a, uint8_t b)->double {return a / (double)b; }; // cast b to double to avoid div by zero (uint8_t)
    auto max   = [](uint8_t a, uint8_t b)->double {return std::max(a, b); };
    auto multByConst = [](uint8_t a, Data::Constant c)->double {return a * (double)c; };

    auto minus_double = [](double a, double b)->double {return a - b; };
    auto add_double   = [](double a, double b)->double {return a + b; };
    auto mult_double  = [](double a, double b)->double {return a * b; };
    auto div_double   = [](double a, double b)->double {return a / b; };
    auto max_double   = [](double a, double b)->double {return std::max(a, b); };
    auto ln_double    = [](double a)->double {return std::log(a); };
    auto exp_double   = [](double a)->double {return std::exp(a); };
    auto multByConst_double = [](double a, Data::Constant c)->double {return a * (double)c; };

    // Convolution (3x3) :
    // - Tableaux de data (2D, cf exemples dans MNIST)
    // - changer dans main (sobelMagn), elle prend un tableau de 3x3
    // - dans le LE, le datasource doit être un primitive type array 2D (Cf MNIST)
    // - ajouter un 2eme paramètre qui précise le bombre de constante pour la convolution

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

    // Init training parameters (load from "/params.json")
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);

    // Initialising number of preloaded CUs
    uint64_t maxNbActionsPerEval = 10*params.maxNbActionsPerEval;                       // 10 000
    uint64_t nbGeneTargetChange = 30;                                                   // 30
    uint64_t nbValidationTarget = 1000;                                                 // 1000

    // Instantiate the LearningEnvironment
    auto *LE = new PartCU({0, 1, 2, 3, 4, 5}, maxNbActionsPerEval, nbGeneTargetChange, nbValidationTarget,  0);

    std::cout << "Number of threads: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Parameters : "<< std::endl;
    std::cout << "  - NB Training Targets = " << maxNbActionsPerEval << std::endl;
    std::cout << "  - NB Validation Targets = " << nbValidationTarget << std::endl;
    std::cout << "  - NB Generation Change = " << nbGeneTargetChange << std::endl;
    std::cout << "  - Ratio Deleted Roots  = " << params.ratioDeletedRoots << std::endl;

    Environment env(set, LE->getDataSources(), params.nbRegisters, params.nbProgramConstant); // Nb de registres dans les programmes

    // Instantiate the TPGGraph that we will load
    //auto tpg = TPG::TPGGraph(env);
    // Create an importer for the best graph and imports it
    //std::cout << "Import graph" << std::endl;
    //File::TPGGraphDotImporter dotImporter(ROOT_DIR "/out_0020.dot", env, tpg);
    //dotImporter.importGraph();

    // Instantiate and Init the Learning Agent (non-parallel : LearningAgent / parallel ParallelLearningAgent)
    Learn::ClassificationLearningAgent la(*LE, set, params);
    //Learn::LearningAgent *la = new Learn::LearningAgent(*LE, set, params);   // USING Non-Parallel Agent to DEBUG
    la.init();

    // Printing every parameters in a .json file
    File::ParametersParser::writeParametersToJson("/home/cleonard/dev/TpgVvcPartDatabase/paramsJson.json", params);
    // "D:/dev/InnovR/TpgVvcPartDatabase/paramsJson.json" || "/home/cleonard/dev/TpgVvcPartDatabase/paramsJson.json"

    // Init the best Policy
    //const TPG::TPGVertex* bestRoot = NULL; // unused ?

    // Start a thread for controlling the loop
#ifndef NO_CONSOLE_CONTROL
    // Console
    std::atomic<bool> exitProgram = true; // (set to false by other thread)
    //std::atomic<uint64_t> generation = 0;

    std::thread threadKeyboard(getKey, std::ref(exitProgram));

    while (exitProgram); // Wait for other thread to print key info.
#else
    std::atomic<bool> exitProgram = false;
#endif

    // Basic logger
    Log::LABasicLogger basicLogger(la);

    // Create an exporter for all graphs
    File::TPGGraphDotExporter dotExporter("out_0000.dot", la.getTPGGraph());

    // Logging best policy stat.
    std::ofstream stats;                                                // Warning : stats is uninitialized
    stats.open("bestPolicyStats.md");
    Log::LAPolicyStatsLogger policyStatsLogger(la, stats);

    // Used as it is, we load 10 000 CUs and we use them for every roots during 5 generations
    // For Validation, 1 000 CUs are loaded and used forever
    // Main training Loop

    std::string const fileClassificationTableName("/home/cleonard/dev/TpgVvcPartDatabase/fileClassificationTableName.txt");
    // "D:/dev/InnovR/TpgVvcPartDatabase/fileClassificationTableName.txt" || "/home/cleonard/dev/TpgVvcPartDatabase/fileClassificationTableName.txt"
    for (int i = 0; i < NB_GENERATIONS && !exitProgram; i++)
    {
        // Each ${nbGeneTargetChange} generation, we generate new random training targets so that different targets are used.
        if (i % nbGeneTargetChange == 0)
        {
            // ---  Deleting old targets ---
            if (i != 0) // Don't clear trainingTargets before initializing them
            {
                LE->reset(i);
                for (uint64_t idx_targ = 0; idx_targ < maxNbActionsPerEval; idx_targ++)
                    delete PartCU::trainingTargetsCU->at(idx_targ);   // targets are allocated in getRandomCU()
                PartCU::trainingTargetsCU->clear();
                PartCU::trainingTargetsOptimalSplits->clear();
                LE->actualTrainingCU = 0;
            }
            else        // Load VALIDATION Targets at the beginning of the training (i == 0)
            {
                for (uint64_t idx_targ = 0; idx_targ < nbValidationTarget; idx_targ++)
                {
                    Data::PrimitiveTypeArray2D<uint8_t>* target = LE->getRandomCU(idx_targ, Learn::LearningMode::VALIDATION);
                    PartCU::validationTargetsCU->push_back(target);
                } 
            }

            // ---  Loading next targets ---
            for (uint64_t idx_targ = 0; idx_targ < maxNbActionsPerEval; idx_targ++)
            {
                Data::PrimitiveTypeArray2D<uint8_t>* target = LE->getRandomCU(idx_targ, Learn::LearningMode::TRAINING);
                PartCU::trainingTargetsCU->push_back(target);
                // Optimal split is saved in LE->trainingTargetsOptimalSplits inside getRandomCU()
            }
        }

        char buff[13];
        sprintf(buff, "out_%04d.dot", i);
        dotExporter.setNewFilePath(buff);
        dotExporter.print();

        la.trainOneGeneration(i);

        /**************************** Printing Classification Table using ugly loop *********************************/
        const TPG::TPGVertex* bestRoot = la.getBestRoot().first;
        LE->printClassifStatsTable(env, bestRoot, i, fileClassificationTableName);

        /**************************** Trying To print Classification Table using getClassificationTable() *********************************/
        
        /*// On recup la best root
        const TPG::TPGVertex *bestRoot = la->getBestRoot().first;
        std::shared_ptr<Learn::EvaluationResult> bestRootResult = la->getBestRoot().second;

        // On relance une évaluation sur cette best root
        LE->reset(0, Learn::LearningMode::VALIDATION);
        auto tee = TPG::TPGExecutionEngine(env);
        const std::vector<const TPG::TPGVertex*> vertices = la->getTPGGraph().getVertices();
        auto iter = std::find(vertices.begin(), vertices.end(), bestRoot);
        auto num = iter - vertices.begin();
        auto job = la->makeJob(num, Learn::LearningMode::VALIDATION);
        Learn::EvaluationResult result = *(la->evaluateJob(tee, *job, i, Learn::LearningMode::VALIDATION, *LE));

        // On affiche la classificationTable
        std::ofstream fichier(fileClassificationTableName.c_str(), std::ios::app);
        if(fichier)
        {
            fichier << "-------------------------------------------" << std::endl;
            fichier << "Gen : " << i << ", score de la best Root : " << bestRootResult->getResult() << std::endl;
            fichier << "     NP     QT    BTH    BTV    TTH    TTV" << std::endl;

            for(int x = 0; x < 6; x++)
            {
                fichier << x;
                for(int y = 0; y < 6; y++)
                {
                    int nb = LE->getClassificationTable().at(x).at(y);

                    int nbChar = (int) (1 + (nb == 0 ? 0 : log10(nb)));
                    for(int nbEspace = 0; nbEspace < (6-nbChar); nbEspace++)
                        fichier << " ";
                    fichier << nb << " ";
                }
                fichier << std::endl;
            }
            fichier.close();
        }else
        {
            std::cout << "Impossible d'ouvrir le fichier fileClassificationTableName." << std::endl;
        }*/
    }

    // After training, keep the best policy
    la.keepBestPolicy();
    dotExporter.setNewFilePath("out_best.dot");
    dotExporter.print();

    TPG::PolicyStats ps;
    ps.setEnvironment(la.getTPGGraph().getEnvironment());
    ps.analyzePolicy(la.getBestRoot().first);
    std::ofstream bestStats;
    bestStats.open("out_best_stats.md");
    bestStats << ps;
    bestStats.close();
    stats.close();

    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++)
        delete (&set.getInstruction(i));
    delete LE;

#ifndef NO_CONSOLE_CONTROL
    // Exit the thread
    std::cout << "Exiting program, press a key then [enter] to exit if nothing happens.";
    threadKeyboard.join();
#endif

    return 0;
}
