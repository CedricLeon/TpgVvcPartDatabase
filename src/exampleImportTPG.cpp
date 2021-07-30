
// ----- La gestion des instructions du TPG -----
// On crée le set d'instructions
Instructions::Set set;
// On crée les instructions, j'ai quasi toujours utilisé les mêmes, ça devrait pas changer
auto minus_double = [](double a, double b) -> double { return a - b; };
auto add_double = [](double a, double b) -> double { return a + b; };
auto mult_double = [](double a, double b) -> double { return a * b; };
auto div_double = [](double a, double b) -> double { return a / b; };
auto max_double = [](double a, double b) -> double { return std::max(a, b); };
auto ln_double = [](double a) -> double { return std::log(a); };
auto exp_double = [](double a) -> double { return std::exp(a); };
auto multByConst_double = [](double a, Data::Constant c) -> double { return a * (double) c; };
// On ajoute les instructions au set
set.add(*(new Instructions::LambdaInstruction<double, double>(minus_double)));
set.add(*(new Instructions::LambdaInstruction<double, double>(add_double)));
set.add(*(new Instructions::LambdaInstruction<double, double>(mult_double)));
set.add(*(new Instructions::LambdaInstruction<double, double>(div_double)));
set.add(*(new Instructions::LambdaInstruction<double, double>(max_double)));
set.add(*(new Instructions::LambdaInstruction<double>(exp_double)));
set.add(*(new Instructions::LambdaInstruction<double>(ln_double)));
set.add(*(new Instructions::LambdaInstruction<double, Data::Constant>(multByConst_double)));

// ----- On vient charger les paramètres du TPG depuis le params.json file -----
// On crée l'objet qui va contenir les paramètres
Learn::LearningParameters params;
// On les charge
File::ParametersParser::loadParametersFromJson(ROOT_DIR "/TPG/params.json", params);


// ----- Les autres paramètres, à part la taille des CUs et le nombre de features normalement tu t'en fous un peu -----
// Les 5 ci-dessous changent avec les entraînements (selon la dtb)
size_t seed = 0;
uint64_t cuHeight = 32;
uint64_t cuWidth  = 32;
uint64_t nbFeatures = 112;
uint64_t nbDatabaseElements = 114348*6;
// Ceux-là sont toujours pareils
uint64_t nbTrainingTargets = 10000;
uint64_t nbGeneTargetChange = 30;
uint64_t nbValidationTarget = 1000;

// ----- Création du contexte d'import -----
// Le learning environment que j'ai créé, il va falloir que je te passe ma classe
auto *le  = new BinaryFeaturesEnv({0}, {1,2,3,4,5}, seed, cuHeight, cuWidth, nbFeatures, nbDatabaseElements, nbTrainingTargets, nbGeneTargetChange, nbValidationTarget);

// L'environment qui va contenir le TPG, il a besoin des instructions, des paramètres et de l'endroit ou il va pouvoir trouver l'input des TPGs  à chaque appel, à savoir dans mon learningEnvironment
Environment env(set, le->getDataSources(), params.nbRegisters, params.nbProgramConstant);

// On crée un TPGGraph qu'on va remplir après l'import
auto tpg = TPG::TPGGraph(env);

// On crée l'ExecutionEngine qui va prendre les décisions à travers le TPGGraph
TPG::TPGExecutionEngine tee(env);

// ----- Import depuis le .dot -----
// On import le fichier dans ces 3 derniers objets (je te mets direct la fonction que j'ai faite, je l'utilise beaucoup)
importTPG(le,  env, tpg);

void importTPG(BinaryFeaturesEnv* le, Environment& env, TPG::TPGGraph& tpg)
{
    try{
        // Je crée le chemin qui contient le .dot (c'est un peu vener pour pas grand chose, si tu veux tester mets le en absolu)
        char tpgPath[100] = ROOT_DIR"/TPG/";
        std::string speActionName = le->getActionName(le->getActions0().at(0));
        std::strcat(tpgPath, speActionName.c_str());
        char dotExtension[10] = ".dot";
        std::strcat(tpgPath, dotExtension);
        //std::cout << "Import TPG from : \"" << tpgPath << "\"" << std::endl;

        // J'appelle le DotImporter, qui peut foirer, d'où le try / catch
        File::TPGGraphDotImporter dotImporter(tpgPath, env, tpg);
    }catch (std::runtime_error& e){
        std::cout << e.what() << std::endl;
    }
}

// ----- Récupération de la racine du graphe -----
// On récupère la root du graph (depuis laquelle on va lancer l'execution)
auto rootNP = tpgNP.getRootVertices().front();

// ----- Données d'entrées, toi ce sera certainement autre chose -----
// On crée un datahandler dans lequel on vient mettre des CUs
// En soit toi tu feras certainement autrement, mais pour mes tests en inférence je le remplissait avec 1000 CUs et j'itérais dessus
auto *dataHandler = new std::vector<Data::PrimitiveTypeArray<double> *>;
// Et du  coup j'ai un vecteur avec les solutions
auto *splitList = new std::vector<uint8_t>;
// Je le rempli avec mes CUs
for (uint64_t idx_targ = 0; idx_targ < nbValidationTarget; idx_targ++)
{
    // Si tu veux la fonction qui importe le CU depuis le .csv je l'ai
    Data::PrimitiveTypeArray<double> *target = getRandomCUFeatures(datasetPath, leNP, splitList);
    dataHandler->push_back(target);
}

// ----- Exécution -----
// Et enfin on execute le TPG  et on récupère l'action choisie
// Pour ça faut d'abord mettre à jour son currentState :
le->setCurrentState(*dataHandler->at(index_du_CU_Courant));
// On l'exécute et on récupère son Action
int actionID = (int) ((const TPG::TPGAction *) tee.executeFromRoot(* root).back())->getActionID();

// Et normalement c'est good
