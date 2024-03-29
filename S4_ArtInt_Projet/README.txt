
    #### grid.pyw ####
    ##################

#### Résumé

Code pour le cours S4 '3IN1007' Intelligence artificielle
Projet de groupe 

Fournit une interface de type 'grille' (matrice n*n) via tkinter.

#### Comment faire

1. Modifier manuellement le json
    1.1. Vérifier le dictionnaire de couleurs
    1.2. Vérifier la taille de la grille
2. Double-cliquer sur 'grid.pyw'
3. Modifier la grille avec "clic-gauche"
4. Sauvegarder avec "Ctrl+s"

1. Modifier manuellement le json
    Les dernières lignes de 'grid.pyw', au moment de créer l'instance
    'Editor', lui passent en chemin le fichier 'test.json'.
  1.1. Vérifier le dictionnaire de couleurs
    L'une des clés du fichier est 'colors' de type dict<str:str>
    où les clés sont des codes ("obstacle","personne",etc.)
    et les valeurs des codes couleur ("white", "rgb(250,128,114)", 
    "#6495ed",etc.).
    Note: la clé vide "" est la valeur des cases par défaut.
    Note: éviter les couleurs aggressives pour les yeux...
  1.2. Vérifier la taille de la grille
    Une autre clé du fichier est 'size' de type list<int> où chaque 
    nombre est la longueur de la matrice (largeur, hauteur).
    Note: La méthode 'Draw.get_size()' présuppose que toutes les lignes 
          de la matrice ont la même longueur.
2. Double-cliquer sur 'grid.pyw'
    Les librairies importées sont toutes natives dans Python 3. Il devrait 
    donc y avoir une fenêtre qui s'ouvre avec un canvas (et des barres de 
    défilement sur les côtés) et la grille du json affichée dedans.
3. Modifier la grille avec "clic-gauche"
    Un clic-gauche sur une case change sa couleur.
    Plus précisément, le clic-gauche change le code de la case puis demande
    au GUI de se mettre à jour. 
4. Sauvegarder avec "Ctrl+s".
    Cela va sauvegarder la grille modifiée dans "test.json".
    Il faut ensuite renommer les fichiers manuellement, etc.

#### 28.03.2024

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
    - ctrl+s :      sauvegarde la grille dans le json.
    
Note : L'éditeur ne permet pas de choisir dans quoi sauvegarder. 
       Il y a bien une propriété 'self.f' en place mais rien pour 
       "sauvegarder sous..." ni "charger..." ou "nouveau...".
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