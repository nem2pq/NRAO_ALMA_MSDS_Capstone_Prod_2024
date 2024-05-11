# module_name.py

# Define your functions, classes, or variables here.
class AlmaTextMiner:

    def __init__(self):
        self.LinearClassifier = load('path_to/linear_classifer')
        self.LDA = load('path_to/lDA')
        self.topic_cluster = pd.read_csv('path_to/topic_cluster_table')
        self.topic_measurement = pd.read_csv('path_to/topic_measurement_table')
        self.Band_Classifier = load('path_to/classifier')
        self.tfidf_vect_LC = load('path_to/tf-idfLC')
        self.tfidf_vect_NB = load('path_to/tf-idfNB')




    def build_recommendation(self, title, abstract):
        text = title + '. ' + abstract
        text = self._preprocess(text)
        LC_val = self.get_LC_pred(text)
        if LC_val == 'continuum':
            return 'This project was classified as a continuum.'
        else:
            topic = self.get_LDA_pred(text)
            topic_clusters = self.topic_cluster[self.topic_cluster['topic'] == topic]
            band_pred = self.get_band_pred(text)
            # subset LDA clusters with band pred
            
            # Come up with return string
            # 'This project was classified as a line project. It was assigned to LDA topic number _
            # and was classified as using band ____. These frequency ranges were observed in this LDA
            # topic and band:'
            
        return True



    def _preprocess(self, text):
        text = text.lower() 
        text = text.strip()
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
        text = re.sub('\s+', ' ', text)  
        text = re.sub(r'\[[0-9]*\]',' ',text) 
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d',' ',text) 
        text = re.sub(r'\s+',' ',text) 
        return text

    
    def get_LC_pred(self, text):
        vect = self.tfidf_vect_LC.transform(text)
        pred = self.LinearClassifier.predict(vect)
        if pred == 1:
            return 'line'
        else:
            return 'continuum'
       
    
    def get_LDA_pred(self, text):
        topic = self.LDA.predict(text)
        return topic
    
    def get_band_pred(self, text):
        vect = self.tfidf_vect_NB.transform(text)
        pred = self.Band_Classifier.predict(vect)
        return pred
        
        

    


        
    
