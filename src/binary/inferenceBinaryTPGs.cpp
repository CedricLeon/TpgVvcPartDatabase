#include <iostream>
#include <cmath>
#include <thread>
#include <atomic>
#include <cinttypes>
#include <cstdlib>

#include <gegelati.h>

#include "../../include/binary/defaultBinaryEnv.h"
#include "../../include/binary/classBinaryEnv.h"

void runOneTPG(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, BinaryDefaultEnv* le);

int main(int argc, char* argv[])
{
    std::cout << "Start VVC Partitionning Optimization with binary TPGs solution" << std::endl;

    // ************************************************** INSTRUCTIONS *************************************************

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
                                            // Balanced database with full classes : 329999
                                            // Unbalanced database : 1136424
                                            // Number of CUs preload changed every nbGeneTargetChange generation for training and load only once for validation
    uint64_t nbTrainingTargets  = 10000;
    uint64_t nbGeneTargetChange = 30;
    uint64_t nbValidationTarget = 1000;

    // The action the binary TPG will be specialized in (0: NP, 1: QT, 2: BTH, 3:BTV, 4: TTH, 5: TTV)
    int speAct = 4;

    // ---------------- Instantiate LearningEnvironment, Environment and TPGGraph ----------------
    auto *le = new BinaryDefaultEnv({0, 1}, speAct, nbTrainingElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget, 0);
    // Instantiate the environment that will embed the LearningEnvironment
    Environment env(set, le->getDataSources(), params.nbRegisters, params.nbProgramConstant);

    // Instantiate the TPGGraph that we will load
    auto tpg = TPG::TPGGraph(env);

    // Instantiate the tee that will handle the decisions taken by the TPG
    TPG::TPGExecutionEngine tee(env);

    // ---------------- Import TPG graph from .dot file ----------------
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

    // ---------------- Execute TPG from root on the environment ----------------
    auto root = tpg.getRootVertices().front();
    runOneTPG(root, tee, le);

    // ---------------- Clean ----------------
    for (unsigned int i = 0; i < set.getNbInstructions(); i++)
        delete (&set.getInstruction(i));
    delete le;

    return 0;
}

void runOneTPG(const TPG::TPGVertex* root, TPG::TPGExecutionEngine& tee, BinaryDefaultEnv* le)
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
        Data::PrimitiveTypeArray2D<uint8_t>* target = le->getRandomCU(idx_targ, Learn::LearningMode::VALIDATION, datasetPath);
        BinaryDefaultEnv::validationTargetsCU->push_back(target);
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
