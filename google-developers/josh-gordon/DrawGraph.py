from sklearn import tree
import pydotplus

def drawDecisionMakingGraph (classifier, features, targets, fileName):
    dot_data = tree.export_graphviz(classifier, out_file=None, 
                            feature_names=features,  
                            class_names=targets,  
                            filled=True, rounded=True,  
                            special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_pdf(fileName+".pdf") 