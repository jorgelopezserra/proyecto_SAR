# versión 1.2

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel


## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"

def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")
    
    if modelname == "SBERT": return SentenceBertEmbeddingModel()    
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all',
                  'positional', "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria pero
        	puedes añadir más variables si las necesitas. 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.docs = {} # diccionario de documentos --> clave: entero(docid),  valor: ruta del fichero.
        self.articles = {} # hash de articulos --> clave string (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(r"\W+") # expresion regular para hacer la tokenizacion
        self.show_all = False # valor por defecto, se cambia con self.set_showall()

        # PARA LA AMPLIACION
        self.semantic = None
        self.positional = False
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = []
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta
        
        
        
        

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_semantic_threshold(self, v:float):
        """

        Cambia el umbral para la búsqueda semántica.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic es False el umbral no tendrá efecto.

        """
        self.semantic_threshold = v

    def set_semantic_ranking(self, v:bool):
        """

        Cambia el valor de semantic_ranking.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic_ranking es True se hará una consulta binaria y los resultados se rankearán por similitud semántica.

        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

            
    def load_semantic_model(self, modelname:str=SEMANTIC_MODEL):
        """
    
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez
        
        """
        if self.model is None:
            print(f"loading {modelname} model ... ",end="", file=sys.stderr)             
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)

            

    def update_chuncks(self, txt:str, artid:int):
        """
        
        Añade los chuncks (frases en nuestro caso) del texto "txt" correspondiente al articulo "artid" en la lista de chuncks
        Pasos:
            1 - extraer los chuncks de txt, en nuestro caso son las frases. Se debe utilizar "sent_tokenize" de la librería "nltk"
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb.
        
        """

        #1 - completar

        #2 - completar

        pass             
        

    def create_kdtree(self):
        """
        
        Crea el tktree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        
        """
        print(f"Creating kdtree ...", end="")
	    # completar
        print("done!")


        
    def solve_semantic_query(self, query:str):
        """

        Resuelve una consulta utilizando el modelo semántico.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultados, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - si el último resultado tiene una distancia <= self.semantic_threshold 
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - también se puede salir si recuperamos todos los embeddings
            5 - tenemos una lista de chuncks que se debe pasar a artículos
        """

        self.load_semantic_model()
        
        # COMPLETAR

        # 1
        # 2
        # 3
        # 4
        # 5


    def semantic_reranking(self, query:str, articles: List[int]):
        """

        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir de los chuncks se deben obtener los artículos
            3 - si entre los artículos recuperados NO estan todos los obtenidos por la RI binaria
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """
        
        self.load_semantic_model()
        # COMPLETAR
        # 1
        # 2
        # 3
        # 4
    

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:
            self.load_semantic_model()


        file_or_dir = Path(root)

        docid = 0 #identificador del documento indexado

        if file_or_dir.is_file():
            # is a file
            self.index_file(root, docid)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname, docid)
                        docid += 1
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        #####################################################
        ## COMPLETAR SI ES NECESARIO FUNCIONALIDADES EXTRA ##
        #####################################################
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []   #nombres de las secciones y subsecciones
        txt_secs = ''    #texto de las secciones y subsecciones

        for sec in article['sections']:
            #se añade el texto de cada sección añadiendo antes el nombre de la sección
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'

            #se añade el texto de las subsecciónes añadiendo antes el nombre de la subsección
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'

            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos

        #se añade una clave 'all' con el texto completo del artículo (titulo + resumen + texto de las secciones y subsecciones)
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs

        #nombre de las secciones y subsecciones
        article['section-name'] = '\n'.join(sec_names)

        return article


    def index_file(self, filename:str, docid:int):
        """

        Indexa el contenido de un fichero.

        input: 
            "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia
            "docid" es un identificador del documento que se indexa, necesario para el índice invertido   

        MODIFICA SELF.INDEX, SELF.DOCS Y SELF.ARTICLES
        Depende de self.positional para el tipo de indexado:                                    self.index[term][1].items() ---> (artid, posiciones)
            Si self.positional es True el formato de self index es termino --> [frecuencia del término, {artid: [posiciones del término en el artículo]}]
            Si self.positional es False el formato de self index es termino --> [artid de los artículos en los que aparece]      

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.positional se debe ampliar el indexado

        """
        
        self.docs[docid] = filename # se añade el documento al diccionario de documentos, con su docid y su ruta


        #i es un identificador de artículo dentro de un docuemento.
        for i, line in enumerate(open(filename)):
            #se convierte la línea del fichero en un diccionario con la información del artículo
            j = self.parse_article(line)

            if self.already_in_index(j):
                continue #si el artículo ya ha sido indexado, se salta
            else:
                self.urls.add(j['url']) #si la url no ha sido indexada, se añade a las urls procesadas

            
            # identificador del artículo y del documento, se puede saber el documento al que
            #pertenece y que artículo dentro del documento es
            article_id = f'{docid}_{i}'
            self.articles[article_id] = (docid, i) # se añade el artículo al diccionario de artículos, es una tupla del docid y la línea dentro del documento


            #Tokenizamos el texto a indexar, que es la clave DEFAULT_FIELD, el método tokenize pasa todo a minúsculas
            # y elimina los símbolos no alfanuméricos. Función ya dada
            texto_tokenizado = self.tokenize(j[self.DEFAULT_FIELD])

            """
            Actualizamos el índice invertido, comprobamos si el término ya está en el índice, si no lo añadimos con una posting list vacía.
            Añadimos el artículo a la posting list del término.
            """
            num_term = 0 #contador del número de término del artículo, necesario para el indexado posicional
            for term in texto_tokenizado:
                #si no está, añadimos la posting list con el contador a 0 y la lista de artículos en las que aparece

                if self.positional:
                    if term not in self.index:
                        #creamos la entrada del término como la frecuencia del término y un diccionario con los artículos en los que aparece y sus posiciones en cada artículo
                        self.index[term] = [0, {}]
                    self.index[term][0] += 1 # incrementamos el contador de apariciones del término
                    dict_articulos = self.index[term][1]

                    #si el artículo no está en la posting list, creamos la lista de posiciones vacía
                    if article_id not in dict_articulos:
                        dict_articulos[article_id] = []
                    
                    #añadimos la posición del término en el artículo
                    dict_articulos[article_id].append(num_term)
                    #ACLARACIÓN: como son diccionarios, al hacer dict_articulos = self.index[term][1] estamos enlazando las variables,
                    # todo cambio hecho a dict_articulos lo estamos haciendo en self.index[term][1]
                    num_term += 1
                else:
                    if term not in self.index:
                        self.index[term] = []
                    if article_id not in self.index[term]: #si el artículo no está en la posting list, lo añadimos
                        self.index[term].append(article_id) 



    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        print("=" * 40)
        print("Number of indexed files:", len(self.docs))
        print("-" * 40)
        print("Number of indexed articles:", len(self.articles))
        print("-" * 40)
        print("Number of vocabulary terms:", len(self.index))
        print("-" * 40)
        if self.positional:
            print("Positional index: YES")
        else:
            print("Positional index: NO")
        print("-" * 40)
        if self.semantic:
            print("Semantic index: YES")
            print("Number of chunks:", len(self.chuncks))
        else:
            print("Semantic index: NO")
        print("=" * 40)



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        
        if query is None or len(query) == 0:
            return []

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        result = None  # posting list acumulada

        # Separamos los tokens respetando las comillas dobles
        # re.findall extrae: secuencias entre comillas O palabras sueltas
        tokens = re.findall(r'"[^"]+"|\S+', query)

        i = 0
        while i < len(tokens):
            token = tokens[i]
            negate = False

            # Comprobamos si el token actual es NOT
            if token.upper() == 'NOT':
                negate = True
                i += 1
                if i >= len(tokens):
                    break
                token = tokens[i]

            # Comprobamos si es una búsqueda posicional (entre comillas)
            if token.startswith('"') and token.endswith('"'):
                # Extraemos los términos dentro de las comillas y los tokenizamos
                phrase = token[1:-1]  # quitamos las comillas
                terms = self.tokenize(phrase)
                posting = self.get_positionals(terms)
            else:
                # Búsqueda normal de un término
                term = self.tokenize(token)
                if len(term) == 0:
                    i += 1
                    continue
                posting = self.get_posting(term[0])

            # Si hay NOT, invertimos la posting list
            if negate:
                posting = self.reverse_posting(posting)

            # Acumulamos con AND
            if result is None:
                result = posting
            else:
                result = self.and_posting(result, posting)

            i += 1

        if result is None:
            return []

        return result

    def get_posting(self, term:str):
        """

        Devuelve la posting list asociada a un termino.
        Puede llamar self.get_positionals: para las búsquedas posicionales.


        param:  "term": termino del que se debe recuperar la posting list.

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        if term not in self.index:
            return []
        
        if self.positional:
            # self.index[term] = [frecuencia, {artid: [posiciones]}]
            return list(self.index[term][1].keys())
        else:
            # self.index[term] = [artid1, artid2, ...]
            return self.index[term]



    def get_positionals(self, terms:str):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LAS BÚSQUESAS POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.

        return: posting list

        """

        #################################
        ## COMPLETAR PARA POSICIONALES ##
        #################################
        if not terms:
            return []

        # Empezamos con los artículos del primer término
        primer_term = terms[0]
        if primer_term not in self.index:
            return []
        
        # artículos candidatos: los que contienen el primer término
        candidatos = set(self.index[primer_term][1].keys())

        for offset, term in enumerate(terms[1:], start=1):
            if term not in self.index:
                return []
            
            dict_term = self.index[term][1]  # {artid: [posiciones]}
            nuevos_candidatos = set()

            for artid in candidatos:
                if artid not in dict_term:
                    continue
                
                pos_primero = set(self.index[primer_term][1][artid])
                pos_term = set(dict_term[artid])

                # comprobamos si alguna posición del primer término tiene
                # el término actual exactamente en +offset
                if pos_primero & {p - offset for p in pos_term}:
                    nuevos_candidatos.add(artid)

            candidatos = nuevos_candidatos

        return list(candidatos)



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list
        si self.positional es True, entonces el formato de la posting list es [frecuencia del término, {artid: [posiciones del término en el artículo]}]
        si es False, entonces el formato es [artid de los artículos]


        return: posting list con todos los artid exceptos los contenidos en p

        """
        result = []
        articles = list(self.articles.keys())
        i = 0
        j = 0
        while i < len(articles):
            if j < len(p):
                if articles[i] == p[j]:
                    i += 1
                    j += 1
                else:
                    doc_a, art_a = self.articles[articles[i]]
                    doc_p, art_p = self.articles[p[j]]
                    if (doc_a < doc_p) or (doc_a == doc_p and art_a < art_p):
                        result.append(articles[i])
                        i += 1
                    else:
                        j += 1
            else:
                result.append(articles[i])
                i += 1
        return result



    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular

        si self.positional es True, entonces el formato de la posting list es [frecuencia del término, {artid: [posiciones del término en el artículo]}]
        si es False, entonces el formato es [artid de los artículos]


        return: posting list con los artid incluidos en p1 y p2

        PROBLEMA: Al haber considerado los identificadores de los artículos como strings, hemos creado problemas de orden.
        Por ejemplo, 2_122 sería menor a 2_3, pese a que uno sea el artículo 122 del documento 2 y el otro sea el artículo 3 del mismo documento

        SOLUCIÓN: Extraemos el identificador del documento y del artículo de cada elemento de las posting lists, al estar en el diccionario self.articles
        el coste de esta operación es constante.

        """
        

        ret = []
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                ret.append(p1[i])
                i += 1
                j += 1
            else:
                doc_1, art_1 = self.articles[p1[i]]
                doc_2, art_2 = self.articles[p2[j]]
                if (doc_1 < doc_2) or (doc_1 == doc_2 and art_1 < art_2):
                    i += 1
                else:
                    j += 1
        return ret
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################






    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular

        si self.positional es True, entonces el formato de la posting list es [frecuencia del término, {artid: [posiciones del término en el artículo]}]
        si es False, entonces el formato es [artid de los artículos]


        return: posting list con los artid incluidos de p1 y no en p2

        Mismo PROBLEMA y SOLUCIÓN que en el método and_posting, explicado detalladamente en el comentario inicial del método y
        dentro del método and_posting 

        """
        ret = []
        i, j = 0, 0
        while i < len(p1):
            if j < len(p2):
                if p1[i] == p2[j]:
                    i += 1
                    j += 1
                else:
                    doc_1, art_1 = self.articles[p1[i]]
                    doc_2, art_2 = self.articles[p2[j]]
                    if (doc_1 < doc_2) or (doc_1 == doc_2 and art_1 < art_2):
                        ret.append(p1[i])
                        i += 1
                    else:
                        j += 1
            else:
                ret.append(p1[i])
                i += 1
        return ret
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################





    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result = self.solve_query(query)
                result = len(result)
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        
        ################
        ## COMPLETAR  ##
        ################
        result = self.solve_query(query)

        print(f"Query: {query}")
        print(f"Number of results: {len(result)}")

        if not self.show_all:
            result = result[:self.SHOW_MAX]

        for orden, artid in enumerate(result):
            docid, i = self.articles[artid]
            filename = self.docs[docid]

            # releemos el fichero y cogemos la línea i
            with open(filename) as f:
                for num, line in enumerate(f):
                    if num == i:
                        article = self.parse_article(line)
                        break

            print(f"{orden}\t{artid}\t{article['title']}\t{article['url']}")

        return len(result)



