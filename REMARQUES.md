# Remarques

## Validation Set
Lors du trainOnegeneration(), si l'option **doValidation** est activée, les roots sont d'abord évaluée, puis selectionnée, puis les survivantes sont évaluée avec LearningMode::Validation. Dans ton implémentation, les imagettes présentées lors de l'évaluation sont identiques à celles présentées lors du training, ce qui ne doit pas être le cas. 

Les imagettes du Validation set doivent normalement être différentes de celles utilisées pour le training afin de vérifier qu'il n'y ait pas 'd'over-fitting' sur le training set. L'over-fitting, c'est quand l'IA arrive très bien à faire une tâche sur le training set, mais ne parvient pas à obtenir les même score sur le validation set. On dit alors que l'IA ne "généralise" pas bien.

## "Volatilité" du score
Avec un set de 1000 imagettes qui est renouvelé toutes les 5 générations, on se rend compte que le score est très sensible au changement de set d'imagette. En effet, toutes les 5 générations, donc à chaque changement du set de 1000 imagettes, on voit que le score fait des sauts, parfois de 30 à 40 points. Cela semble indiquer que la meilleur root voit son score changer fortement lorsque le jeu d'imagette est modifié.

C'est assez gênant, car cela veut dire qu'à la faveur d'un set d'imagette "favorable", une root "moins" forte peu potentiellement devenir la plus forte le temps de 5 générations, et provoquer la disparition de root pourtant plus "stables", mais pour lesquelles le set d'imagette courant n'est pas forcément favorable.

Pour résoudre cela, je pense qu'il faut augmenter considérablement le nombre d'imagettes utilisées à chaque chargement.

## maxNbEvaluationPerPolicy
Le paramètre `maxNbEvaluationPerPolicy` permet de fixer le nombre d'évaluation (en mode Learn::LearningMode::Training) pendant lequel le score d'une root sera mis à jour.
Par exemple, 

- A sa première évaluation, la root obtient un score de 10.0, son score moyenné est 10.0
- A la seconde évaluation, la root obtient un score de 15.0, son score moyenné est 12.5 (= (15*1 + 10*1) / 2)
- A la troisième évaluation, la root obtient un score de 11.0, son score moyenné est 12.0 (= (11*1 + 12.5*2) / 3)

Une fois arrivé à autant d'évaluation que défini par `maxNbEvaluationPerPolicy` , le score de la root (en mode Training) ne sera plus évalué, et le score enregistré sera retenu. Cela permet de gagner du temps sur les futures évaluations.

Dans ton cas de figure, le problème est que chaque root est évaluée 1 fois par génération, et le `maxNbEvaluationPerPolicy` est fixé à 10. Etant donné que le data-set est mis à jour toutes les 5 générations, il y a de grande chance qu'une root soit réévaluée plusieurs fois (i.e. sur plusieurs générations successives) sur le même set de 1000 imagettes. Cela n'est évidemment pas très bon pour la validité du score obtenu. De plus, comme le score est très volatile d'un set d'imagettes à un autre, le score calculé n'est pas forcément très représentatif.

Pour résoudre ce problème, il faudrait faire en sorte que le jeu d'imagettes présentées à chaque génération ne soit pas le même. Pour cela, je pense que tu peux facilement charger quelques dizaine de milliers d'imagettes en mémoire, et les présenter aléatoirement aux roots pendant de nombreuses générations.

**On load 10 000 images et on les présentes pendant 5 générations** 

## Pistes d'améliorations futures

### Amélioration des perfs
* Utilisation du nouveau Array2DWrapper à la place du `PrimitiveTypeArray `(pas forcément ultra conséquent)
* Faire du `trainingTargetCU` un attribut statique. (cela évitera le temps nécessaire à sa copie lors du clonage du learningEnvironment.) **DONE**

### Amélioration du score
* Changer les paramètres pour qu'ils soient plus favorables. (à discuter)
  * plus de roots (~2000)
  * moins de survivants (~ratioDeletedRoots ~0.95)

**2000 roots - 0.90 ratioDeletedRoots** (Kelly le met à 0.5 pour la diversité, mais il en faut pas trop : pour l'instant test avec 0.90)

* Ajouter de nouvelles instructions (à discuter) **(peut etre plus tard : convolution)**
* Essayer d'hériter du ClassificationLearningEnvironment. **(peut etre plus tard)**
* Essayer de faire des TPG spécifiques à chaque type de découpe (i.e. Action binaire) plutôt que d'en faire un pour toutes les actions. **(On verra plus tard mais on pourra même faire un truc plus meta avec un 7ème TPG qui manage la sortie des 6 autres)**



boucler sur les optimal splits après le chargement pour afficher l'équilibre du set

faire un autre vector de validations et de 1000



# Lancer Entrainement sur la machine 24 coeurs

#### Se log sur la machine

- aller sur  *vpn.insa-rennes.fr*

- Si pas fait, installer le .exe pour windows (pour la sécu)

- Se login sur *SSH HTML5 v2 Dynamique IETR*

- pc-eii21.insa-rennes.fr

- identifiant et mdp INSA



````bash
cmake .. -DCMAKE_BUILD_TYPE=Release             // sous Linux
cmake --build . --target runTests -- -j      	// build gegelati et vérifie qu'elle fonctionne
												// -- pour dire que les arguments suivants sont pour GCC
./bin/runTests
sudo cmake --build . --target install -- -j
````

#### Trick pour lancer une exec via SSH et qu'il continue

````bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release _DTESTING=1
cmake --build .
nohup ./nom_de_lexec > logs.txt &
// Pour voir ou il en est
tail -f logs.txt
// (ou cat)

// Pour l'arreter
htop
*F5* Pour trouver le process parent puis *F9* (fait un kill)
````

