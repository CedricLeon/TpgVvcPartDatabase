#include <iostream>
#include <cmath>
#include <thread>
#include <cinttypes>

#include <gegelati.h>

#include "../../include/features/FeaturesEnv.h"

int main(int argc, char* argv[])
{
    std::cout << "Start TPGVVCPartDatabase : training a full (6 actions) TPG based on CU features extraction (CNN)." << std::endl;

    // ************************************************** INSTRUCTIONS *************************************************

    // Create the instruction set for programs
    Instructions::Set set;

    // double instructions (for TPG programs)
    auto minus_double = [](double a, double b)->double {return a - b; };
    auto add_double   = [](double a, double b)->double {return a + b; };
    auto mult_double  = [](double a, double b)->double {return a * b; };
    auto div_double   = [](double a, double b)->double {return a / b; };
    auto max_double   = [](double a, double b)->double {return std::max(a, b); };
    auto ln_double    = [](double a)->double {return std::log(a); };
    auto exp_double   = [](double a)->double {return std::exp(a); };
    auto multByConst_double = [](double a, Data::Constant c)->double {return a * (double)c; };
    auto conv2D_double = [](const Data::Constant coeff[9], const double data[3][3])->double {
        double res = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                res += (double)coeff[i * 3 + j] * data[i][j];
            }
        }
        return res;
    };

    // Add those instructions to instruction set
    set.add(*(new Instructions::LambdaInstruction<double, double>(minus_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(mult_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(div_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(max_double)));
    set.add(*(new Instructions::LambdaInstruction<double>(exp_double)));
    set.add(*(new Instructions::LambdaInstruction<double>(ln_double)));
    set.add(*(new Instructions::LambdaInstruction<double, Data::Constant>(multByConst_double)));
    set.add(*(new Instructions::LambdaInstruction<const Data::Constant[9], const double[3][3]>(conv2D_double)));


    // ******************************************* PARAMETERS AND ENVIRONMENT ******************************************

    // ---------------- Loading and initializing parameters ----------------
    // Init training parameters (load from "/params.json")
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);

    // Initialising the number of CUs used
    uint64_t nbTrainingElements = 60000;    // Balanced database with equal of each class : 60000 (10000 (6: TTV), 12000 (5: NP), 15000 (4: QT), 20000 (3: BTH),  30000 (2: BTV et TTH))à
    // Database with 55000 elements of one class and 11000 elements of each other class : 110000
    // Balanced database with full classes : 329999
    // Unbalanced database : 1136424
    // Number of CUs preload changed every nbGeneTargetChange generation for training and load only once for validation
    uint64_t nbTrainingTargets  = 10000;
    uint64_t nbGeneTargetChange = 30;
    uint64_t nbValidationTarget = 1000;

    // Extracting the seed parameter from main arguments
    int seed = 0;
    if (argc > 1)
    {
        seed = atoi(argv[1]);
        std::cout << "SEED is precised: " << seed << std::endl;
    }
    else
    {
        std::cout << "SEED was not precised, using default value: " << seed << std::endl;
    }

    // ---------------- Instantiate Environment and Agent ----------------
    // LearningEnvironment
    auto *LE = new FeaturesEnv({0, 1, 2, 3, 4, 5}, nbTrainingElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget, (size_t) seed);
    // Creating a second environment used to compute the classification table
    Environment env(set, LE->getDataSources(), params.nbRegisters, params.nbProgramConstant);

    // Instantiate and Init the Learning Agent (non-parallel : LearningAgent / parallel ParallelLearningAgent)
    Learn::ParallelLearningAgent la(*LE, set, params);
    la.init();

    // ---------------- Initialising paths ----------------
    char datasetPath[100] = "/home/cleonard/Data/features/balanced1/";
    //const char parametersPrintPath[100] = "/home/cleonard/dev/TpgVvcPartDatabase/build/jsonParams.json";
    std::string const fileClassificationTableName("/home/cleonard/dev/TpgVvcPartDatabase/fileClassificationTable.txt");
    std::string const fullConfusionMatrixName("/home/cleonard/dev/TpgVvcPartDatabase/fullClassifTable.txt");

    // ---------------- Printing training overview  ----------------
    std::cout << "This TPG uses CU features and has 6 actions" << std::endl << std::endl;
    std::cout << "Number of threads: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Parameters: "<< std::endl;
    std::cout << "  - NB Training Targets   = " << nbTrainingTargets << std::endl;
    std::cout << "  - NB Validation Targets = " << nbValidationTarget << std::endl;
    std::cout << "  - NB Generation Change  = " << nbGeneTargetChange << std::endl;
    std::cout << "  - Ratio Deleted Roots   = " << params.ratioDeletedRoots << std::endl;

    // Printing every parameters in a .json file
    //File::ParametersParser::writeParametersToJson(parametersPrintPath, params);

    // ************************************************ LOGS MANAGEMENT ************************************************

    // Create a basic logger
    Log::LABasicLogger basicLogger(la);

    // Create an exporter for all graphs
    File::TPGGraphDotExporter dotExporter("out_0000.dot", la.getTPGGraph());

    // Logging best policy stat.
    std::ofstream stats;
    stats.open("bestPolicyStats.md");
    Log::LAPolicyStatsLogger policyStatsLogger(la, stats);


    // *********************************************** MAIN TRAINING LOOP **********************************************

    // Used as it is, we load 10 000 CUs and we use them for every roots during 30 generations
    // For Validation, 1 000 CUs are loaded and used forever

    for (uint64_t i = 0; i < params.nbGenerations; i++)
    {
        // Update Training and Validation targets depending on the generation
        LE->UpdatingTargets(i, datasetPath);

        // Save best generation policy
        char buff[20];
        sprintf(buff, "out_%" PRIu64 ".dot", i);
        dotExporter.setNewFilePath(buff);
        dotExporter.print();

        // Train
        la.trainOneGeneration(i);

        // Print Classification Table
        const TPG::TPGVertex* bestRoot = la.getBestRoot().first;
        LE->printClassifStatsTable(env, bestRoot, (int) i, fileClassificationTableName, false);
        LE->printClassifStatsTable(env, bestRoot, (int) i, fullConfusionMatrixName, true);
    }

    // ************************************************** TRAINING END *************************************************
    // After training, keep the best policy
    la.keepBestPolicy();
    dotExporter.setNewFilePath("out_best.dot");
    dotExporter.print();
    // Store stats
    TPG::PolicyStats ps;
    ps.setEnvironment(la.getTPGGraph().getEnvironment());
    ps.analyzePolicy(la.getBestRoot().first);
    std::ofstream bestStats;
    bestStats.open("out_best_stats.md");
    bestStats << ps;
    bestStats.close();

    // close logs file
    stats.close();

    // cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++)
        delete (&set.getInstruction(i));
    delete LE;

    return 0;
}

