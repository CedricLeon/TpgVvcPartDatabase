#include <iostream>

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
    auto minus = [](uint8_t a, uint8_t b)->uint8_t {return a - b; };
    auto add   = [](uint8_t a, uint8_t b)->uint8_t {return a + b; };
    auto mult  = [](uint8_t a, uint8_t b)->uint8_t {return a * b; };
    //auto div   = [](uint8_t a, uint8_t b)->uint8_t {return a / b; };
    auto max   = [](uint8_t a, uint8_t b)->uint8_t {return std::max(a, b); };
    auto ln    = [](uint8_t a)->uint8_t {return std::log(a); };
    auto exp   = [](uint8_t a)->uint8_t {return std::exp(a); };

    auto minus_double = [](double a, double b)->double {return a - b; };
    auto add_double = [](double a, double b)->double {return a + b; };
    auto mult_double = [](double a, double b)->double {return a * b; };
    auto div_double = [](double a, double b)->double {return a / b; };
    auto max_double = [](double a, double b)->double {return std::max(a, b); };
    auto ln_double = [](double a)->double {return std::log(a); };
    auto exp_double = [](double a)->double {return std::exp(a); };

    // Add those instructions to instruction set
    set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(minus)));
    set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(add)));
    set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(mult)));
    //set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(div)));
    set.add(*(new Instructions::LambdaInstruction<uint8_t, uint8_t>(max)));
    set.add(*(new Instructions::LambdaInstruction<uint8_t>(exp)));
    set.add(*(new Instructions::LambdaInstruction<uint8_t>(ln)));

    set.add(*(new Instructions::LambdaInstruction<double, double>(minus_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(add_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(mult_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(div_double)));
    set.add(*(new Instructions::LambdaInstruction<double, double>(max_double)));
    set.add(*(new Instructions::LambdaInstruction<double>(exp_double)));
    set.add(*(new Instructions::LambdaInstruction<double>(ln_double)));

    // Init training parameters (load from "/params.json")
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);

    // Initialising number of preloaded CUs
    uint64_t nbTargetsLoaded = params.maxNbActionsPerEval * 10;                  // 10 000
    uint8_t nbGeneTargetChange = 30;                                             // 5

    // Instantiate the LearningEnvironment
    PartCU LE({0, 1, 2, 3, 4, 5}, nbTargetsLoaded, nbGeneTargetChange, 0);

    std::cout << "Number of threads: " << std::thread::hardware_concurrency() << std::endl;

    // Instantiate and Init the Learning Agent (non-parallel : LearningAgent / parallel ParallelLearningAgent)
    Learn::ParallelLearningAgent la(LE, set, params);
    //Learn::LearningAgent la(LE, set, params);   // USING Non-Parallel Agent to DEBUG
    la.init();

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
    std::ofstream stats;
    stats.open("bestPolicyStats.md");
    Log::LAPolicyStatsLogger policyStatsLogger(la, stats);

    // Used as it is (nbGeneTargetChange is commented in loops), we load 10 000 CUs and we use them for every roots during 5 generations
    // Main training Loop
    for (int i = 0; i < NB_GENERATIONS && !exitProgram; i++)
    {
        // Each ${nbGeneTargetChange} generation, we generate new random training targets so that different targets are used.
        if (i % nbGeneTargetChange == 0)
        {
            // ---  Deleting old targets ---
            if (i != 0) // Don't clear trainingTargets before initializing them
            {
                LE.reset(i);
                for (uint64_t idx_targ = 0; idx_targ < nbTargetsLoaded/* *nbGeneTargetChange */; idx_targ++)
                    delete PartCU::trainingTargetsCU[idx_targ];   // targets are allocated in getRandomCU()
                PartCU::trainingTargetsCU.clear();
                PartCU::trainingTargetsOptimalSplits.clear();
                LE.actualCU = 0;
            }

            // ---  Loading next targets ---
            for (uint64_t idx_targ = 0; idx_targ < nbTargetsLoaded/* *nbGeneTargetChange*/; idx_targ++)
            {
                Data::PrimitiveTypeArray<uint8_t>* target = LE.getRandomCU();
                PartCU::trainingTargetsCU.push_back(target);
                // Optimal split is saved in LE.trainingTargetsOptimalSplits inside getRandomCU()
            }
        }

        char buff[13];
        sprintf(buff, "out_%04d.dot", i);
        dotExporter.setNewFilePath(buff);
        dotExporter.print();

        la.trainOneGeneration(i);
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

#ifndef NO_CONSOLE_CONTROL
    // Exit the thread
    std::cout << "Exiting program, press a key then [enter] to exit if nothing happens.";
    threadKeyboard.join();
#endif

    return 0;
}
