#include "../include/PartCU.h"
#include <chrono>       // use to time list initialisation
#include <vector>
#include <algorithm>   // for random_shuffle

// ********************************************************************* //
// ************************** GEGELATI FUNCTIONS *********************** //
// ********************************************************************* //

void PartCU::doAction(uint64_t actionID)
{
    // Pour le soft dans VVC on resize avant l'action

    // Managing the reward (+1 if the action match the best split, else +0)
    if(actionID == optimal_split)
        score++;

    // Loading next CU
    LoadNextCU();
}

std::vector<std::reference_wrapper<const Data::DataHandler>> PartCU::getDataSources()
{
    // Return a vector containing every element constituting the State of the environment
    auto result = std::vector<std::reference_wrapper<const Data::DataHandler>>();
    result.push_back(this->currentCU);

    return result;
}

void PartCU::reset(size_t seed, Learn::LearningMode mode)
{
    // RNG Control : Create seed from seed and mode
    size_t hash_seed = Data::Hash<size_t>()(seed) ^ Data::Hash<Learn::LearningMode>()(mode);
    // Reset the RNG
    this->rng.setSeed(hash_seed);

    // Reset the count of CU
    nbSplitsJob = 0;

    // Reset the score
    score = 0;

    // Preload the first CU
    if (!LoadNextCU())
    {
       std::cout << "LoadNextCU() : returned false ..." << std::endl;
    }
}

Learn::LearningEnvironment* PartCU::clone() const
{
    return new PartCU(*this);
}

bool PartCU::isCopyable() const
{
    return true;
}

double PartCU::getScore() const
{
    // Return the RDO Cost ? (for VVC software)
    return score;
}

bool PartCU::isTerminal() const
{
    // Poser la question à Karol : return compteDeSplit == maxNbActionsPerEval ?

    // Return if the job is over
    return nbSplitsJob == MAX_NB_ACTIONS_PER_EVAL;
}

// ********************************************************************* //
// *************************** PartCU FUNCTIONS ************************ //
// ********************************************************************* //


void PartCU::InitRandomList()
{
    // Init liste aléatoire
    // NB total elements in the database : 1 420 531
    // If we keep 20% of the elements for the verification that let 1 136 424 elements for training

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    // std::cout << "My vector max size : " << this->CU_list.max_size() << std::endl;

    // ------------------------------ Filling Time ------------------------------
    for(uint32_t i = 0; i < NB_TRAINING_ELEMENTS; i++)  // Extremly fast ...
        this->CU_list.push_back(i);

    // ------------------------------ Shuffle Time ------------------------------
    // Is it really necessary to pick CU randomly and make sure that a CU isn't picked 2 times ?
    // If the TPG picks a CU already tried in another job that change nothing ?

    // Default : using std::random_shuffle
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(this->CU_list.begin(), this->CU_list.end(), std::default_random_engine(seed));

    // Else, switch 2 randomly (with rng => GEGELATI determinist library) picked elements X times :
    /*uint32_t position = 0;
    uint32_t next_position = 0;
    for(uint32_t i = 0; i < NB_TRAINING_ELEMENTS*10; ++i)  // Make 10 browses of CU_list
    {
        position      = this->rng.getInt32(0, NB_TRAINING_ELEMENTS-1);
        next_position = this->rng.getInt32(0, NB_TRAINING_ELEMENTS-1);
        iter_swap(this->CU_list.begin() + position, this->CU_list.begin() + next_position);
    }*/

    // Print Time :
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::cout << "Time : " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << std::endl;
}

bool PartCU::LoadNextCU()
{
    // Update the PATH to the next CU
    uint32_t next_CU_number = this->rng.getInt32(0, NB_TRAINING_ELEMENTS-1);

    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number /*this->CU_list[next_CU_number]*/);

    // Dunno why strcat threw exeptions, this is dirty but works
    char current_CU_path[100] = "D:/dev/InnovR/dataset_tpg_32x32_27/dataset_tpg_32x32_27/";
    char bin_extension[10] = ".bin";
    std::strcat(current_CU_path, next_CU_number_string);
    std::strcat(current_CU_path, bin_extension);

    // Open the file
    std::FILE* input = std::fopen(current_CU_path, "r");
    if (!input) // Check validity
    {
        std::perror("File opening failed");
        //std::cout << "Cannot open file : " << current_CU_path << std::endl;
        return EXIT_FAILURE;
    }
    uint8_t contents[32*32+1];
    std::fread(&contents[0], 1, 32*32+1, input);

    // Important ! ^^
    std::fclose(input);

    // Get the solution
    this->optimal_split = contents[1024];

    // Load the image in the dataSource
    for (uint32_t pxlIndex = 0; pxlIndex < 32*32; pxlIndex++)
    {
        try
        {
            this->currentCU.setDataAt(typeid(uint8_t), pxlIndex, contents[pxlIndex]);
        }
        catch (std::invalid_argument& e) {
            std::cout << e.what() << " LoadNextCU( pxlIndex : " << pxlIndex << ")" << std::endl;
            return false;
        }
        catch (std::out_of_range& e) {
            std::cout << e.what() << " LoadNextCU( pxlIndex : " << pxlIndex << ")" << std::endl;
            return false;
        }
    }
    return true;
}

/***********************************************************************
    Ce TPG prendra en entrée un CU de 32x32 et fournira en sortie une action, un choix de split.
    La reward sera calculée à partir du RDO-Cost mais inversé (Plus le RDO-Cost est petit mieux c'est, mais le TPG essaie de maximiser par défaut donc getScore() rendra un truc type 1/RDO-Cost : à détailler)
    Pour l'entrainement on le lance dans le dossier D:\dev\InnovR\dataset_tpg_32x32_27\dataset_tpg_32x32_27 et on le fait tourner sur les 1.4 M de fichiers .bin ?

    Demander à Alexandre si il prend juste le 32x32 pixels ou 32x32 plus quelques pixels sur les bords (padding)

    Dans les fichiers .bin :
    - 32x32 valeurs sous format uint_8

    1 seul TPG, 1 seule taille et 4 ou 5 décisions

    Globalement les trucs à faire :
     - Les Actions :
        - NP  (0) : Non-Partitionnement
        - QT  (1) : Quad-Tree Partitionning
        - BTH (2) : Binary-Tree  Horizontal
        - BTV (3) : Binary-Tree  Vertical
        - TTH (4) : Ternary-Tree Horizontal
        - TTV (5) : Ternary-Tree Vertical

     - Régler le params.json pour faire environ 1000 évaluations de CU par Job
        "nbIterationsPerJob = 1" / "maxNbActionsPerEval = 1000" : les changer pour faire 1000 évaluations / découpes de CU par Job
     - (Mais dans le vrai entrainement une partie sera la découpe d'un CU et le nb max d'action)

     - La fonction doAction() :
        - Compare l'action choisit par le TPG et la compare à celle de l'encodeur / database
        - Update le score en conséquences (+1 si bonne décision, 0 sinon comme pré-entrainement et après on utilisera le RD-Cost pour le vrai entrainement dans VVC)
        - Change de CU, de façon aléatoire dans la base de données réservée pour l'entrainement (en garder une partie pour la validation) 80% / 20%

     - Une fonction nextCU() :
        - Va récupérer le prochain .bin et met à jour le currentState ?

     - Récupérer le TPG à la fin de l'entrainement ou au milieu si pause :
        - main.c quand il export des fichiers .dot donc à la fin on récupère "out_best.dot"
        - Et ensuite au lieu de de faire un init TPG on fait un dot importer avec le fichier .dot voulu

    - Regarder Mnist pour finir de coder le TPG (Attention à ClassificationLearner::doAction())


    Note Global 18 / Vidéo 16 (attention durée +6 /-4 ? ) / Rapport : résumé + intro moyen reste


    Résultat :
        - Si c'est encourageant on verra


    STAGE REGARDER LES PROCEDURES :
        - Dates : 2 premières semaines de Aout labo fermé (télétravail ?)
        - Attendre résultat des mobi avant procédures
        - Stage de plus je suis là longtemps mieux c'est

     - Une fonction MiseAEchelle() :      ****** Pas utile pour ce soft, database contient que des CU 32x32 ********
        - Sur-échantillonne le CU si il est trop petit pour qu'il soit de taille 32x32

************************************************************************/
