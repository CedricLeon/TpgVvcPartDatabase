#include "../include/PartCU.h"

// ********************************************************************* //
// ************************** GEGELATI FUNCTIONS *********************** //
// ********************************************************************* //

void PartCU::doAction(uint64_t actionID)
{
    // Action !
}

std::vector<std::reference_wrapper<const Data::DataHandler>> PartCU::getDataSources()
{
    // Return a vector containing every element constituting the State of the environment
    auto result = std::vector<std::reference_wrapper<const Data::DataHandler>>();
    result.push_back(this->board);
    //result.push_back(something);
    return result;
}

void PartCU::reset(size_t seed, Learn::LearningMode mode)
{
    // RNG Control :
    // Create seed from seed and mode
    size_t hash_seed = Data::Hash<size_t>()(seed) ^ Data::Hash<Learn::LearningMode>()(mode);
    // Reset the RNG
    this->rng.setSeed(hash_seed);

    // Reset the environment ?
    //this->initGame();
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
    // Return the RDO Cost ?
    //return score;
}

bool PartCU::isTerminal() const
{
    // Return if the run is over
    // return win || lost;
}

// ********************************************************************* //
// *************************** PartCU FUNCTIONS ************************ //
// ********************************************************************* //

/***********************************************************************
    Ce TPG prendra en entrée un CU de 32x32 et fournira en sortie une action, un choix de split.
    La reward sera calculée à partir du RDO-Cost mais inversé (Plus le RDO-Cost est petit mieux c'est, mais le TPG essaie de maximiser par défaut donc getScore() rendra un truc type 1/RDO-Cost : à détailler)
    Pour l'entrainement on le lance dans le dossier D:\dev\InnovR\dataset_tpg_32x32_27\dataset_tpg_32x32_27 et on le fait tourner sur les 1.4 M de fichiers .bin ?

    Globalement les trucs à faire :
     - Les Actions :
        - NP  (0) : Non-Partitionnement
        - QT  (1) : Quad-Tree Partitionning (Trouver une façon de lui interdire ou de le pénaliser s'il a l'information qu'un
                                             autre type de split a déjà était fait précédemment)
        - BTH (2) : Binary-Tree  Horizontal
        - BTV (3) : Binary-Tree  Vertical
        - TTH (4) : Ternary-Tree Horizontal
        - TTV (5) : Ternary-Tree Vertical

     - Une fonction Partition() :
        - a accès au CU dont elle doit choisir le partitionnement (le current State serait un vector de 32x32=1024 éléments ?)
        - en fonction de l'action choisit, split le CU et renvoie 2, 3 ou 4 vectors contenant la valeur des pixels des sous-CU ?

     - Une fonction updateNextState() :
        - va récupérer le prochain .bin et met à jour le currentState ?


     - Une fonction MiseAEchelle() :      ****** Pas utile pour ce soft, database contient que des CU 32x32 ********
        - Sur-échantillonne le CU si il est trop petit pour qu'il soit de taille 32x32

************************************************************************/
