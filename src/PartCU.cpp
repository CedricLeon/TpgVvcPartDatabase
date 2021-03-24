#include "../include/PartCU.h"
#include <chrono>       // use to time list initialisation
#include <vector>

// ********************************************************************* //
// ************************** GEGELATI FUNCTIONS *********************** //
// ********************************************************************* //

void PartCU::doAction(uint64_t actionID) {
    // Pour le soft dans VVC on resize avant l'action

    // Managing the reward (+1 if the action match the best split, else +0)
    if (actionID == optimal_split)
        score++;

    // Loading next CU
    LoadNextCU();
}

std::vector<std::reference_wrapper<const Data::DataHandler>> PartCU::getDataSources() {
    // Return a vector containing every element constituting the State of the environment
    std::vector<std::reference_wrapper<const Data::DataHandler>> result{this->currentCU};

    return result;
}

void PartCU::reset(size_t seed, Learn::LearningMode mode) {
    // RNG Control : Create seed from seed and mode
    size_t hash_seed = Data::Hash<size_t>()(seed) ^Data::Hash<Learn::LearningMode>()(mode);
    // Reset the RNG
    this->rng.setSeed(hash_seed);

    // Reset the score
    score = 0;
    
    // Update the mode
    this->currentMode = mode;

    // Preload the first CU (depending of the current mode)
    LoadNextCU();
}

Learn::LearningEnvironment *PartCU::clone() const {
    return new PartCU(*this);
}

bool PartCU::isCopyable() const {
    return true; // false : pour eviter qu'il se lance en parallel (Cf LearningAgent)
}

double PartCU::getScore() const {
    // Return the RDO Cost ? (for VVC software)
    return (double) score;
}

bool PartCU::isTerminal() const {
    // Return if the job is over
    return false;
}

// ********************************************************************* //
// *************************** PartCU FUNCTIONS ************************ //
// ********************************************************************* //

// ****** TRAINING Arguments ******
std::vector<Data::PrimitiveTypeArray<uint8_t> *> *PartCU::trainingTargetsCU = new std::vector<Data::PrimitiveTypeArray<uint8_t>*>; // Array2DWrapper
std::vector<uint8_t> *PartCU::trainingTargetsOptimalSplits = new std::vector<uint8_t>;
// ****** VALIDATION Arguments ******
std::vector<Data::PrimitiveTypeArray<uint8_t>*> *PartCU::validationTargetsCU = new std::vector<Data::PrimitiveTypeArray<uint8_t>*>;  // Array2DWrapper
std::vector<uint8_t> *PartCU::validationTargetsOptimalSplits = new std::vector<uint8_t>;

Data::PrimitiveTypeArray<uint8_t> *PartCU::getRandomCU(uint64_t index, Learn::LearningMode mode) {
    // ------------------ Opening and Reading a random CU file ------------------
    uint32_t next_CU_number = this->rng.getInt32(0, NB_TRAINING_ELEMENTS - 1);
    char next_CU_number_string[100];
    std::sprintf(next_CU_number_string, "%d", next_CU_number);
    char current_CU_path[100] = "/home/cleonard/Data/dataset_tpg_32x32_27/";
    // "D:/dev/InnovR/dataset_tpg_32x32_27/dataset_tpg_32x32_27/" || "/home/cleonard/Data/dataset_tpg_32x32_27/"
    char bin_extension[10] = ".bin";
    std::strcat(current_CU_path, next_CU_number_string);
    std::strcat(current_CU_path, bin_extension);

    // Openning the file
    std::FILE *input = std::fopen(current_CU_path, "r");
    if (!input) {
        std::perror("File opening failed");
        return nullptr; // return EXIT_FAILURE;
    }

    // Stocking content in a uint8_t tab, first 32x32 uint8_t are CU's pixels values and the 1025th value is the optimal split
    uint8_t contents[32*32+1];
    /*int nbCharRead = */std::fread(&contents[0], 1, 32*32+1, input);
    /*if (nbCharRead != 32*32+1)
        std::perror("File Read failed");*/
    // Dunno why it fails

    // Important ...
    std::fclose(input);

    // Creating a new PrimitiveTypeArray<uint8_t> and filling it
    Data::PrimitiveTypeArray<uint8_t> *randomCU = new Data::PrimitiveTypeArray<uint8_t>(32 * 32);
    for (uint32_t pxlIndex = 0; pxlIndex < 32 * 32; pxlIndex++)
        randomCU->setDataAt(typeid(uint8_t), pxlIndex, contents[pxlIndex]);

    // Updating the corresponding optimal split depending of the current mode
    if (mode == Learn::LearningMode::TRAINING)
        PartCU::trainingTargetsOptimalSplits->push_back(contents[1024]);
    else if (mode == Learn::LearningMode::VALIDATION)
        PartCU::validationTargetsOptimalSplits->push_back(contents[1024]);

    return randomCU;
}

void PartCU::LoadNextCU() {
    // Checking validity is no longer necessary
    if (this->currentMode == Learn::LearningMode::TRAINING)
    {
        this->currentCU = *PartCU::trainingTargetsCU->at(this->actualTrainingCU);
        
        // Updating next split solution
        this->optimal_split = PartCU::trainingTargetsOptimalSplits->at(this->actualTrainingCU);
        this->actualTrainingCU++;
        
        // Looping on the beginning of training targets
        if (this->actualTrainingCU >= NB_TRAINING_TARGETS)
            this->actualTrainingCU = 0;
    }
    else if (this->currentMode == Learn::LearningMode::VALIDATION)
    {
        this->currentCU = *PartCU::validationTargetsCU->at(this->actualValidationCU);

        // Updating next split solution
        this->optimal_split = PartCU::validationTargetsOptimalSplits->at(this->actualValidationCU);
        this->actualValidationCU++;

        // Looping on the beginning of validation targets
        if (this->actualValidationCU >= NB_VALIDATION_TARGETS)
            this->actualValidationCU = 0;
    }
}

/***********************************************************************
    Ce TPG prendra en entrée un CU de 32x32 et fournira en sortie une action, un choix de split.
    La reward sera calculée à partir du RDO-Cost mais inversé (Plus le RDO-Cost est petit mieux c'est, mais le TPG essaie de maximiser par défaut donc getScore() rendra un truc type 1/RDO-Cost : à détailler)
    Pour l'entrainement on le lance dans le dossier D:\dev\InnovR\dataset_tpg_32x32_27\dataset_tpg_32x32_27 et on le fait tourner sur les 1.4 M de fichiers .bin ?

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

        changer instructions en uint8_t
        en ouvrir max nb val (1000), les garder en mémoire (vector) pour toutes les roots (dans nbActions, 
        dans le main rajouter entre 2 générations pour chargeer 1000 nouvelles images
        et a chaque reset le reloadCU va se balader dans la liste des 1000 CUs

        pour verif : dans le params.json le DO_VALIDATion = true 
        changer de dataset en fonction du training / learning mode (Cf lien)
        Cf params pour le json

************************************************************************/
