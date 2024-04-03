
    #### grid.pyw ####
    ##################

#### Résumé

Code pour le cours S4 '3IN1007' Intelligence artificielle
Projet de groupe 

Fournit une interface de type 'grille' (matrice n*n) via tkinter.
#### Comment faire

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