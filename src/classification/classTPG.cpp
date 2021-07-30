#include <iostream>
#include <cmath>
#include <thread>
#include <cinttypes>

#include <gegelati.h>

#include "../../include/classification/ClassEnv.h"

int main()
{
    std::cout << "Start TPGVVCPartDatabase : training a full (6 actions) classification TPG." << std::endl;

    // ************************************************** INSTRUCTIONS *************************************************
    // Create the instruction set for programs
    Instructions::Set set;

    // Create the uint8_t instructions
    auto minus = [](uint8_t a, uint8_t b)->double {return a - b; };
    auto add   = [](uint8_t a, uint8_t b)->double {return a + b; };
    auto mult  = [](uint8_t a, uint8_t b)->double {return a * b; };
    auto div   = [](uint8_t a, uint8_t b)->double {return a / (double)b; }; // cast b to double to avoid div by zero (uint8_t)
    auto max   = [](uint8_t a, uint8_t b)->double {return std::max(a, b); };
    auto multByConst = [](uint8_t a, Data::Constant c)->double {return a * (double)c; };
    auto mean2 = [](const uint8_t data[2][2])->double
    {
        uint8_t sum = 0;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                sum += data[i][j];
        return sum;
    };
    auto mean3 = [](const uint8_t data[3][3])->double
    {
        uint8_t sum = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                sum += data[i][j];
        return sum;
    };
    auto mean4 = [](const uint8_t data[4][4])->double
    {
        uint8_t sum = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                sum += data[i][j];
        return sum;
    };
    auto mean5 = [](const uint8_t data[5][5])->double
    {
        uint8_t sum = 0;
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                sum += data[i][j];
        return sum;
    };
    auto var2 = [mean2](const uint8_t data[2][2])->double
    {
        double sum = 0;
        double dataTmp[2][2];

        // Compute mean
        double mean = mean2(data);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                // Subtract mean from elements
                dataTmp[i][j] = data[i][j] - mean;
                // Square each term
                dataTmp[i][j] *= dataTmp[i][j];
            }
        }

        // Take sum
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                sum += data[i][j];

        return sum / (2 * 2);
    };
    auto var3 = [mean3](const uint8_t data[3][3])->double
    {
        double sum = 0;
        double dataTmp[3][3];

        // Compute mean
        double mean = mean3(data);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                // Subtract mean from elements
                dataTmp[i][j] = data[i][j] - mean;
                // Square each term
                dataTmp[i][j] *= dataTmp[i][j];
            }
        }

        // Take sum
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                sum += data[i][j];

        return sum / (3 * 3);
    };
    auto var4 = [mean4](const uint8_t data[4][4])->double
    {
        double sum = 0;
        double dataTmp[4][4];

        // Compute mean
        double mean = mean4(data);

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                // Subtract mean from elements
                dataTmp[i][j] = data[i][j] - mean;
                // Square each term
                dataTmp[i][j] *= dataTmp[i][j];
            }
        }

        // Take sum
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                sum += data[i][j];

        return sum / (4 * 4);
    };
    auto var5 = [mean5](const uint8_t data[5][5])->double
    {
        double sum = 0;
        double dataTmp[5][5];

        // Compute mean
        double mean = mean5(data);

        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                // Subtract mean from elements
                dataTmp[i][j] = data[i][j] - mean;
                // Square each term
                dataTmp[i][j] *= dataTmp[i][j];
            }
        }

        // Take sum
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                sum += data[i][j];

        return sum / (5 * 5);
    };

    // Create the double instructions
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
    set.add(*(new Instructions::LambdaInstruction<const uint8_t[2][2]>(mean2)));
    set.add(*(new Instructions::LambdaInstruction<const uint8_t[2][2]>(var2)));
    set.add(*(new Instructions::LambdaInstruction<const uint8_t[3][3]>(mean3)));
    set.add(*(new Instructions::LambdaInstruction<const uint8_t[3][3]>(var3)));
    set.add(*(new Instructions::LambdaInstruction<const uint8_t[4][4]>(mean4)));
    set.add(*(new Instructions::LambdaInstruction<const uint8_t[4][4]>(var4)));
    set.add(*(new Instructions::LambdaInstruction<const uint8_t[5][5]>(mean5)));
    set.add(*(new Instructions::LambdaInstruction<const uint8_t[5][5]>(var5)));

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


    // ******************************************* PARAMETERS AND ENVIRONMENT ******************************************

    // ---------------- Loading and initializing parameters ----------------
    // Init training parameters (load from "/params.json")
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);

    // Printing every parameters (even default ones) in a .json file
    //File::ParametersParser::writeParametersToJson("/home/cleonard/dev/TpgVvcPartDatabase/build/paramsJson.json", params);

    // Initialising number of preloaded CUs
    uint64_t nbTrainingTargets = 10000;
    uint64_t nbGeneTargetChange = 30;
    uint64_t nbValidationTarget = 1000;
    size_t seed = 0;

    // ---------------- Instantiate Environment and Agent ----------------
    // LearningEnvironment
    auto *LE = new ClassEnv({0, 1, 2, 3, 4, 5}, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget,  seed);
    // Creating a second environment used to compute the classification table
    Environment env(set, LE->getDataSources(), params.nbRegisters, params.nbProgramConstant);

    // The BaseLearningAgent template parameter is the LearningAgent from which the ClassificationLearningAgent inherits.
    // This template notably enable selecting between the classical and the ParallelLearningAgent.
    // Instantiate and Init the Learning Agent (non-parallel : LearningAgent / parallel ParallelLearningAgent)
    Learn::ClassificationLearningAgent la(*LE, set, params);
    la.init();

    // ---------------- Initialising paths ----------------
    const std::string datasetPath = "/home/cleonard/Data/CU/CU_32x32_balanced/";
    //"/media/cleonard/alex/cedric_TPG-VVC/balanced_datasets/32x32_balanced/";
    const std::string fileClassificationTableName("/home/cleonard/dev/TpgVvcPartDatabase/fileClassificationTable.txt");
    const std::string fullConfusionMatrixName("/home/cleonard/dev/TpgVvcPartDatabase/fullClassifTable.txt");

    // ---------------- Printing training overview  ----------------
    std::cout << "Number of threads: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Parameters : "<< std::endl;
    std::cout << "  - NB Training Targets   = " << nbTrainingTargets << std::endl;
    std::cout << "  - NB Validation Targets = " << nbValidationTarget << std::endl;
    std::cout << "  - NB Generation Change  = " << nbGeneTargetChange << std::endl;
    std::cout << "  - Ratio Deleted Roots   = " << params.ratioDeletedRoots << std::endl;

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
    for (uint64_t i = 0; i < params.nbGenerations; i++)
    {
        // Update Training and Validation targets depending on the generation
        LE->UpdateTargets(i, datasetPath);

        // Save best generation policy (spend unnecessary computation resources)
        //char buff[20];
        //sprintf(buff, "out_%" PRIu64 ".dot", i);
        //dotExporter.setNewFilePath(buff);
        //dotExporter.print();

        // Train
        la.trainOneGeneration(i);

        // Print Classification Table
        const TPG::TPGVertex* bestRoot = la.getBestRoot().first;
        LE->printClassifStatsTable(env, bestRoot, i, fileClassificationTableName, false);
        LE->printClassifStatsTable(env, bestRoot, i, fullConfusionMatrixName, true);
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

    // Close logs file
    stats.close();

    // Cleanup
    for (unsigned int i = 0; i < set.getNbInstructions(); i++)
        delete (&set.getInstruction(i));
    delete LE;

    return 0;
}
