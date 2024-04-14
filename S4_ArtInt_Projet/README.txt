
    #### grid.pyw ####
    ##################

#### Résumé

Code pour le cours S4 '3IN1007' Intelligence artificielle
Projet de groupe 
Fournit une interface de type 'grille' (matrice n*n) via tkinter.
Fournit un éditeur rudimentaire pour colorer la grille.
Fournit un "moteur de jeu".

#### Comment faire

    ## Éditeur

1. Double-cliquer sur 'editor.pyw'
    Cela va ouvrir une fenêtre (interface GUI) et charger le fichier 
    'test.json'.
2. Modifier la grille
    Clic-gauche et clic-droit sur les cases pour changer leur couleur.
3. Sauvegarder/charger
    Avec le menu ou les raccourcis-clavier suivants : 
    "Ctrl+n" pour créer une nouvelle grille.
    "Ctrl+o" pour ouvrir un fichier.
    "Ctrl+s" pour sauvegarder la grille.
    "Ctrl+Alt+s" pour sauvegarder dans un nouveau fichier.
    
Note : L'éditeur ne permet pas de modifier la taille de la grille ou 
       le dictionnaire de couleurs. Pour cela, il faut modifier 
       manuellement le '.json' (clés 'colors' et 'size').

    ## Jeu
    
1. Double-cliquer sur 'meteor.py'
    Cela va ouvrir une fenêtre (interface GUI) et charger le fichier 
    'test.json'.
2. Presser les touches [Control]+[n]
    Cela va lancer la partie (une case "rouge" devrait apparaître).
3. Utiliser les flèches directionnelles ou [wasd] pour déplacer le joueur.
    L'objectif est de récupérer toutes les personnes (cases bleues) 
    avant de rejoindre un abri (cases jaunes). Le score final est affiché en 
    console.

Note : pas de retour après une action (pour l'agent).
Note : pas de chargement/sauvegarde de partie. 
Note : pas de météore.


#### 14.04.2024

Modifié 'meteor.py' avec la classe 'Meteor' et son intégration dans 'Game'.
Également ajouté les méthodes 'direction','distance' et 'expand' à 'Grid'.

Finalement abandonné une classe à part pour le joueur.
Il faudrait de toute manière intégrer la transparence au canvas pour gérer 
des combinaisons complexes, et tkinter ne le fait pas tout seul. Donc pas 
de véhicules et autres.

Version semi-aléatoire en place, les météores tombent de façon aléatoire 
mais toutes les cases deviennent nécessairement des obstacles. En l'état 
le jeu semble satisfaisant.

La méthode "Grid.expand" est le plus gros ajout : il permet de 'scanner' les
alentours d'une case, pour dessiner les météores ou sélectionner une case 
libre. 
"Grid.distance" donne la distance entre deux cases et "Grid.direction" 
retourne un tuple (x,y) avec la direction à prendre pour aller d'un point 
à un autre. 

Il faut encore déterminer l'information renvoyée à l'agent (et la manière)
mais en l'état on peut considérer le jeu lui-même comme fini.
(On pourra modifier au besoin.)

J'allais oublier : génération aléatoire du joueur, des abris et des personnes
à sauver s'ils ne sont pas déjà placés sur la carte manuellement.
(Pas encore vraiment testé.)

#### 06.04.2024

Ajouté 'meteor.py' avec la classe 'Game'. 

La classe est très rudimentaire, notamment pour gérer le joueur, et ne 
génère pas encore de météore (ni ne charge/sauvegarde de partie).

Les fonctions pour l'agent ("mU,mD,mL,mR") sont en place mais il n'y a rien
pour connaître l'état de l'environnement. Retourner un dictionnaire ? Ou 
laisser l'agent lire la grille par lui-même ?
Mieux vaut supposer qu'il faille transmettre l'information à l'agent, comme 
pour le *snake*.

#### 03.04.2024

Ajouté 'Draw.get_v_pos()'. Testé une solution ne recréant que les 
rectangles visibles (dans 'Draw.refresh()'). 

C'est un échec. Il y a un clignotement lors du scrolling et les gains, s'ils 
sont significatifs sur des grilles de +10'000 cases, sont nuls voire négatifs
pour les petites grilles.
Il serait encore possible de modifier la structure de 'Grid' (en ne conservant 
que les cases "utiles" en mémoire) mais ça n'a plus le moindre intérêt.

#### 31.03.2024

Déplacé la classe 'Editor' dans un nouveau fichier 'grid.pyw'.

Tenté d'optimiser encore 'refresh()' mais désormais, sans changer 
fondamentalement la structure, tkinter a atteint ses limites.

#### 29.03.2024

Modifié 'Cell', 'Draw' et ajouté la classe 'Editor'.

Désormais 'Draw' met l'image à jour toutes les 17 millisecondes ('self.ts')
    via la méthode 'self.refresh()'. Au lieu de recréer tous les rectangles,
    on change les coordonnées des triangles existants (on suppose qu'on ne
    rajoute ni n'enlève aucune case). Aussi, une liste 'self.l_refresh' 
    fournit les cases à modifier lors d'un "refresh". Cela oblige à appeler
    "Draw.cell_refresh()' de l'extérieur quand on modifie une case.
'Draw' a également une méthode 'Draw.get_cell(x,y)' qui, des coordonnées de 
    la souris, renvoie la case correspondante. 
    
La classe 'Editor' se contente d'ajouter les appels souris/clavier. 
    - clic-gauche : appelle 'self.next_col(e)' qui change la couleur de la 
                    case. 
    - clic-droit :  appelle 'self.prev_col(e)' qui change la couleur dans 
                    le sens inverse.
    - ctrl+s :      sauvegarde la grille dans le json.
    - ctrl+alt+s :  sauvegarde en demandant un nom de fichier.
    - ctrl+o :      charge un fichier json (en demandant le nom du fichier).
    - ctrl+n :      crée une nouvelle grille.

Note : Le dictionnaire de couleurs doit être modifié manuellement 
       dans le json ('draw'>'colors').

#### 28.03.2024

Mis en place les classes 'Cell', 'Grid' et 'Draw'.
Mis en place les méthodes pour sauvegarder/charger un json (load/save)
    pour 'Grid' et 'Draw'.
Créé un json de test 'test.json'.

Lorsqu'on double-clique sur 'grid.pyw', la condition '__main__' crée 
    une instance 'Draw' qui elle-même crée une instance 'Grid' (self.gr).
'Draw' met en place l'interface graphique via 'self.build_gui()' puis 
    dessine un rectangle par case via 'self.draw()'.
La méthode 'self.on_resize()' change la taille des cases ('self.sz') et 
    l'espace entre les cases ou "padding" (via 'self.get_pad()', 1/10 de 
    la taille) ainsi qu'un "offset" pour centrer la grille ('self.ox/oy').

Note: Le GUI peine à gérer les grandes grilles.
      Testé avec 100*100, déjà du lag; testé avec 1000*1000, ça ne réagit 
      presque plus.
Note: 'Grid' a une méthode 'self.get(x,y)' qui renvoie la case à la position
      (x,y).