#include <iostream>
#include <cmath>
#include <thread>
#include <cinttypes>

#include <gegelati.h>

#include "../../include/features/BinaryFeaturesEnv.h"

void ParseStringInVector(std::vector<uint8_t>& actions, std::string str)
{
    // This condition below is not necessary in normal execution conditions
    // But when calling the executable from a script, bash force the extern quotes as part of the string
    // Which fucked up the whole process below. So we erase these quotes when necessary
    if(str[0] == '"' && str[str.size() - 1] == '"')
    {
        str.erase(0, 1);
        str.erase(str.size() - 1);
    }

    // Remove first ('{') and last char ('}') from str
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

            actions.push_back((uint8_t) nb);
            str.erase(0,2);
            //std::cout << str.size() << " : |" << str << "|" << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
    // ******************************************* MAIN ARGUMENTS EXTRACTION *******************************************

    // Example : "./TPGVVCPartDatabase_binaryFeaturesEnv {0} {1,2,3,4,5} 0 32 32 112 686088"

    // Default arguments
    std::vector<uint8_t> actions0 = {0};
    std::vector<uint8_t> actions1 = {1,2,3,4,5};
    size_t seed = 0;
    uint64_t cuHeight = 32;
    uint64_t cuWidth = 32;
    uint64_t nbFeatures = 112;
    uint64_t nbDatabaseElements = 100000; // binary balanced databases
    std::string actName = "NP";
    //114348*6 : 32x32_balanced database

    //std::cout << "argc: " << argc << std::endl;
    if (argc == 9)
    {
        actions0.clear(); actions1.clear();
        ParseStringInVector(actions0, (std::string) argv[1]);
        ParseStringInVector(actions1, (std::string) argv[2]);
        seed = atoi(argv[3]);
        cuHeight = atoi(argv[4]);
        cuWidth = atoi(argv[5]);
        nbFeatures = atoi(argv[6]);
        nbDatabaseElements = atoi(argv[7]);
        actName = argv[8];
    }
    else
    {
        std::cout << "Arguments were not precised (waiting 7 arguments : actions0, actions1, seed, cuHeight, cuWidth, nbFeatures and nbDatabaseElements). Using default value." << std::endl;
        std::cout << "Example : \"./TPGVVCPartDatabase_binaryFeaturesEnv {0} {1,2,3,4,5} 0 32 32 112 686088\"" << std::endl ;
    }
    std::cout << std::endl << "---------- Main arguments ----------" << std::endl;
    //std::cout << "argv[1]: " << argv[1] << ", argv[2]: " << argv[2] << std::endl;
    std::cout << std::setw(13) << "actions0 :";
    for (auto &act : actions0)
        std::cout << std::setw(4) << (int) act;
    std::cout << std::endl << std::setw(13) << "actions1 :";
    for (auto &act : actions1)
        std::cout << std::setw(4) << (int) act;
    std::cout << std::endl << std::setw(13) << "seed :" << " " << std::setw(3) << seed << std::endl;
    std::cout << std::setw(13) << "cuHeight :" << " " << std::setw(4) << cuHeight << std::endl;
    std::cout << std::setw(13) << "cuWidth :" << " " << std::setw(4) << cuWidth << std::endl;
    std::cout << std::setw(13) << "nbFeatures :" << " " << std::setw(4) << nbFeatures << std::endl;
    std::cout << std::setw(13) << "actName :" << " " << std::setw(4) << actName << std::endl;

    std::cout << std::endl << "Start the training of a TPG based on CU features extraction (CNN)" << std::endl;

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
/*    auto conv2D_double = [](const Data::Constant coeff[9], const double data[3][3])->double {
        double res = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                res += (double)coeff[i * 3 + j] * data[i][j];
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

    // ---------------- Loading and initializing parameters ----------------
    // Init training parameters (load from "/params.json")
    Learn::LearningParameters params;
    File::ParametersParser::loadParametersFromJson(ROOT_DIR "/params.json", params);

    // Initialising the number of CUs used
    //uint64_t nbDatabaseElements = 60000;    // Balanced database with equal of each class : 60000 (10000 (6: TTV), 12000 (5: NP), 15000 (4: QT), 20000 (3: BTH),  30000 (2: BTV et TTH))à
    // Database with 55000 elements of one class and 11000 elements of each other class : 110000
    // Balanced database with full classes : 329999
    // Unbalanced database : 1136424
    // Number of CUs preload changed every nbGeneTargetChange generation for training and load only once for validation
    uint64_t nbTrainingTargets  = 10000;
    uint64_t nbGeneTargetChange = 30;
    uint64_t nbValidationTarget = 1000;

    // ---------------- Instantiate Environment and Agent ----------------
    // LearningEnvironment
    auto *LE = new BinaryFeaturesEnv(actions0, actions1, seed, cuHeight, cuWidth, nbFeatures, nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);
    // Creating a second environment used to compute the classification table
    Environment env(set, LE->getDataSources(), params.nbRegisters, params.nbProgramConstant);

    // Instantiate and Init the Learning Agent (non-parallel : LearningAgent / parallel ParallelLearningAgent)
    Learn::ParallelLearningAgent la(*LE, set, params);
    la.init();

    // ---------------- Initialising paths ----------------
    // "/media/cleonard/alex/cedric_TPG-VVC/balanced_datasets/" || "/home/cleonard/Data/features/"
    std::string datasetBasePath = "/media/cleonard/alex/cedric_TPG-VVC/balanced_datasets/";
    std::string datasetMiddlePath = "x";
    std::string datasetType = "_perso/";
    std::string datasetEndPath = "/";
    std::string datasetPath = datasetBasePath
                            + std::to_string(LE->getCuHeight())
                            + datasetMiddlePath
                            + std::to_string(LE->getCuWidth())
                            + datasetType
                            + actName
                            + datasetEndPath;
    std::string const fileClassificationTableName("/home/cleonard/dev/TpgVvcPartDatabase/fileClassificationTable.txt");
    std::string const fullConfusionMatrixName("/home/cleonard/dev/TpgVvcPartDatabase/fullClassifTable.txt");

    // ---------------- Printing training overview  ----------------
    std::cout << "This TPG uses CU features and has 2 actions" << std::endl;
    std::cout << "It is trained on the database: " << datasetPath << std::endl << std::endl;
    std::cout << "Number of threads: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Parameters: "<< std::endl;
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

        // Save best generation policy
        //char buff[20];
        //sprintf(buff, "out_%" PRIu64 ".dot", i);
        //dotExporter.setNewFilePath(buff);
        //dotExporter.print();

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

