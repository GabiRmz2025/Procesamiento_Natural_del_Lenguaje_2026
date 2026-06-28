from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer, MWETokenizer, RegexpTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk
import string 
import unicodedata
import re


#Clase para procesar texto
class NLPTextCleaningPipeline:
    def __init__(self, idioma = 'spanish', opciones_normalizar = {"urls","correos","menciones","hashtags","numeros","emojis","puntuacion","espacios"},
                 remover_acentos= True, remover_stopwords= True, 
                 normalizar_unicode=True, metodo_stem = None,
                 expresiones = None                 
                ):
       
        self.idioma = idioma
        self.stopwords = set(stopwords.words('spanish' if idioma=='spanish' else 'english'))

        self.opciones_a_normalizar = opciones_normalizar
        
        self.word_t = word_tokenize
        self.sent_t = sent_tokenize
        self.tweet_t = TweetTokenizer()
        self.MWE_t = MWETokenizer(expresiones or [], separator='_') #(Multi-Word Expression Tokenizer) une palabras para formar una expresión compuesta.
        self.regexp_t = RegexpTokenizer(r'\w+') #(Regular Expression Tokenizer) utiliza expresiones regulares y las separa o extrae los tokens de un texto.

        self.porter = PorterStemmer()
        self.lancaster = LancasterStemmer()
        self.snow_en = SnowballStemmer('english') #Stemming con idioma inglés
        self.snow_es = SnowballStemmer('spanish') #Stemming con idioma español
        
        self.lematizer = WordNetLemmatizer()
 
#=======================================================================================          
    #Función que limpia los textos - corpus 
    def Normalizar(self, texto):
        if self.opciones_a_normalizar is None:
            opciones = {"urls", "correos", "menciones", "hashtags", "numeros", "emojis", "puntuacion", "espacios"}
        if "urls"       in self.opciones_a_normalizar: texto = re.sub(r'https?://\S+', '', texto)
        if "correos"    in self.opciones_a_normalizar: texto = re.sub(r'[\w.]+@[\w.]+\.[a-zA-Z]{2,}', '', texto)
        if "menciones"  in self.opciones_a_normalizar: texto = re.sub(r'@\w+', '', texto)
        if "hashtags"   in self.opciones_a_normalizar: texto = re.sub(r'#\w+', '', texto)
        if "numeros"    in self.opciones_a_normalizar: texto = re.sub(r'\S*\d\S*', '', texto)
        if "emojis"     in self.opciones_a_normalizar: texto = re.sub(r'[^\x00-\x7Fáéíóúüñ¡¿ÁÉÍÓÚÜÑa-zA-Z \n.,!?]', '', texto)
        if "puntuacion" in self.opciones_a_normalizar: texto = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ]', ' ', texto, flags=re.UNICODE)
        if "espacios"   in self.opciones_a_normalizar: texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto.lower()

#=======================================================================================
    #Función que Elimina tildes y diacríticos usando normalización Unicode.         
    def quitar_acentos(self, texto):      
        nfkd = unicodedata.normalize('NFKD', texto)
        return ''.join(c for c in nfkd if not unicodedata.combining(c))

#=======================================================================================   
    #Función que normaliza texto y quita acentos
    def preprocesar(self, texto):
        texto = self.Normalizar(texto)
        texto = self.quitar_acentos(texto)
        return texto

#=======================================================================================  
    #Función para eliminar los Stopwords
    def eliminar_stopwords(self, tokens):
        return [
            token
            for token in tokens
            if token.lower() not in self.stopwords
        ]

#=======================================================================================        
    #Función para tokenizar texto con LNTK
    def tokenizar(self, texto, metodo = "word"):
            
        if metodo == "word":
            return self.word_t(texto)       
        elif metodo == "sent":
            return self.sent_t(texto)
        elif metodo == "tweet":
            return self.tweet_t.tokenize(texto)
        elif metodo == "regexp":
            return self.regexp_t.tokenize(texto)
        elif metodo == "mwe":
            tokens = self.word_t(texto.lower())
            return self.MWE_t.tokenize(tokens)
        else:
            raise ValueError(f"Método '{metodo}' no válido")
    
#=======================================================================================
    #Función de Stemming con NLTK, obtener la raíz de la palabra    
    def stemming(self, tokens, metodo = "snowball"):
        
        if metodo == "porter":
            return [self.porter.stem(t) for t in tokens]
        elif metodo == "lancaster":
            return [self.lancaster.stem(t) for t in tokens]  
        elif metodo == "snowball":    
            stemmer = (self.snow_es
                if self.idioma == "spanish"
                else self.snow_en)    
            return [stemmer.stem(t) for t in tokens]  
  
        else:
            raise ValueError(f"Método '{metodo}' no válido")            

#=======================================================================================
    #Función para Lematizar con NLTK
    def lematizar(self, tokens):
           
        return [self.lematizer.lemmatize(t) for t in tokens]           

#=======================================================================================    
    #Función para Pipeline principal, etapas:PREPROCESAR (NORMALIZAR Y QUITAR ACENTOS), TOKENIZAR, ELIMINAR STOPWORDS.
    def pipeline(self, texto):   
        texto = self.preprocesar(texto)    
        tokens = self.tokenizar(texto)
        vocabulario_original = len(set(tokens))
        tokens_sin_stop = self.eliminar_stopwords(tokens)    
        return tokens_sin_stop   
        
#=======================================================================================    
    #Función para procesar en el PIPELINE principal; STEMMING Y LEMATIZAR
    def pipeline_vocabulario(self, texto):
        texto = self.preprocesar(texto)
        tokens = self.tokenizar(texto)
        vocabulario_original = len(set(tokens))
        tokens_sin_stop = self.eliminar_stopwords(tokens)
        vocabulario_sin_stop = len(set(tokens_sin_stop))
        stems = self.stemming(tokens_sin_stop)
        vocabulario_stem = len(set(stems))
        lemas = self.lematizar(tokens_sin_stop)
        vocabulario_lemma = len(set(lemas))
    
        return {
            "Original": vocabulario_original,
            "Sin Stopwords": vocabulario_sin_stop,
            "Stemming": vocabulario_stem,
            "Lematizacion": vocabulario_lemma
        }