#include <iostream>

#include <cmath>
#include <thread>
#include <atomic>
#include <cinttypes>

#include <gegelati.h>

#include "../include/defaultBinaryEnv.h"
#include "../include/classBinaryEnv.h"

#ifndef NB_GENERATIONS
#define NB_GENERATIONS 2000
#endif
/**
 * \brief Manage training run : press 'q' or 'Q' to stop the training
 *
 * \param[exit] used by the threadKeyboard (set to false by other thread)
 */
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
    std::cout << "Start VVC Partitionning Optimization with binary TPGs solution." << std::endl;

    // ******************* INSTRUCTIONS *******************

    // Create the instruction set for programs
    Instructions::Set set;
    // uint8_t instructions (for pixels values)
    auto minus = [](uint8_t a, uint8_t b)->double {return a - b; };
    auto add   = [](uint8_t a, uint8_t b)->double {return a + b; };
    auto mult  = [](uint8_t a, uint8_t b)->double {return a * b; };
    auto div   = [](uint8_t a, uint8_t b)->double {return a / (double)b; }; // cast b to double to avoid div by zero (uint8_t)
    auto max   = [](uint8_t a, uint8_t b)->double {return std::max(a, b); };
    auto multByConst = [](uint8_t a, Data::Constant c)->double {return a * (double)c; };

    // double instructions (for TPG programs)
    auto minus_double = [](double a, double b)->double {return a - b; };
    auto add_double   = [](double a, double b)->double {return a + b; };
    auto mult_double  = [](double a, double b)->double {return a * b; };
    auto div_double   = [](double a, double b)->double {return a / b; };
    auto max_double   = [](double a, double b)->double {return std::max(a, b); };
    auto ln_double    = [](double a)->double {return std::log(a); };
    auto exp_double   = [](double a)->double {return std::exp(a); };
    auto multByConst_double = [](double a, Data::Constant c)->double {return a * (double)c; };
    auto conv2D_double = [](const Data::Constant coeff[9], const uint8_t data[3][3])->double {
        double res = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                res += (double)coeff[i * 3 + j] * data[i][j];
            }
        }
        return res;
    };

    // Add those instructions to instruction set
    // uint8_t
    set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(minus)));
    set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(add)));
    set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(mult)));
    set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(div)));
    set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(max)));
    set.add(*(new Instructions::LambdaInstruction<uint8_t, Data::Constant>(multByConst)));
    // double
    set.add(*(new Instructions::LambdaInstruction<double, double>(minus_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(mult_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(div_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(max_double)));
    set.add(*(new Instructions::LambdaInstruction<double>(exp_double)));
    set.add(*(new Instructions::LambdaInstruction<double>(ln_double)));
    set.add(*(new Instructions::LambdaInstruction<double, Data::Constant>(multByConst_double)));
    set.add(*(new Instructions::LambdaInstruction<const Data::Constant[9], const uint8_t[3][3]>(conv2D_double)));


    // ******************* PARAMETERS AND ENVIRONMENT *******************

    // Init training parameters (load from "/params.json")
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);

    // Initialising number of preloaded CUs
    uint64_t nbTrainingElements = 110000;
    // Balanced database with 55000 elements of one class and 11000 elements of each other class : 110000
    // Balanced database with full classes : 329999
    // Unbalanced database : 1136424
    uint64_t nbTrainingTargets = 10*params.maxNbActionsPerEval;                         // 10 000
    uint64_t nbGeneTargetChange = 30;                                                   // 30
    uint64_t nbValidationTarget = 1000;                                                 // 1000

    // Initialising paths
    const char databasePath[100] = "/home/cleonard/Data/binary_datasets/NP_dataset/";
    // /home/cleonard/Data/binary_datasets/NP_dataset/ || /home/cleonard/Data/dataset_tpg_balanced/dataset_tpg_32x32_27_balanced2/
    const char parametersPrintPath[100] = "/home/cleonard/dev/TpgVvcPartDatabase/build/paramsJson.json";
    std::string const fileClassificationTableName("/home/cleonard/dev/TpgVvcPartDatabase/fileClassificationTableName.txt");

    // Instantiate the LearningEnvironment
    auto *LE = new BinaryClassifEnv({0, 1}, 1, nbTrainingElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget, 2021);

    std::cout << "Number of threads: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Parameters : "<< std::endl;
    std::cout << "  - NB Training Targets   = " << nbTrainingTargets << std::endl;
    std::cout << "  - NB Validation Targets = " << nbValidationTarget << std::endl;
    std::cout << "  - NB Generation Change  = " << nbGeneTargetChange << std::endl;
    std::cout << "  - Ratio Deleted Roots   = " << params.ratioDeletedRoots << std::endl;

    Environment env(set, LE->getDataSources(), params.nbRegisters, params.nbProgramConstant);

    // Instantiate and Init the Learning Agent (non-parallel : LearningAgent / parallel ParallelLearningAgent)
    Learn::ParallelLearningAgent la(*LE, set, params);
    //Learn::LearningAgent *la = new Learn::LearningAgent(*LE, set, params);   // USING Non-Parallel Agent to DEBUG
    la.init();

    // Printing every parameters in a .json file
    File::ParametersParser::writeParametersToJson(parametersPrintPath, params);


    // ******************* CONSOLE CONTROL *******************

    // Start a thread to control the loop
#ifndef NO_CONSOLE_CONTROL
    std::atomic<bool> exitProgram = true; // (set to false by other thread)
    std::thread threadKeyboard(getKey, std::ref(exitProgram));
    while (exitProgram); // Wait for other thread to print key info.
#else
    std::atomic<bool> exitProgram = false;
#endif


    // ******************* LOGS MANAGEMENT *******************

    // Create a basic logger
    Log::LABasicLogger basicLogger(la);

    // Create an exporter for all graphs
    File::TPGGraphDotExporter dotExporter("out_0000.dot", la.getTPGGraph());

    // Logging best policy stat.
    std::ofstream stats;                                                // Warning : stats is uninitialized
    stats.open("bestPolicyStats.md");
    Log::LAPolicyStatsLogger policyStatsLogger(la, stats);


    // ******************* TRAINING LOOP *******************

    // Used as it is, we load 10 000 CUs and we use them for every roots during 5 generations
    // For Validation, 1 000 CUs are loaded and used forever

    for (int i = 0; i < NB_GENERATIONS && !exitProgram; i++)
    {
        // Update Training and Validation targets depending on the generation
        LE->UpdatingTargets(i, databasePath);

        // Save best generation policy
        char buff[13];
        sprintf(buff, "out_%04d.dot", i);
        dotExporter.setNewFilePath(buff);
        dotExporter.print();

        // Train
        la.trainOneGeneration(i);

        // Print Classification Table
        const TPG::TPGVertex* bestRoot = la.getBestRoot().first;
        LE->printClassifStatsTable(env, bestRoot, i, fileClassificationTableName);
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

