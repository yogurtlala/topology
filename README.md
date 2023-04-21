# topology:blush::blush:
 Extract topology feature from image by persistence homolgy and  build model to classify.  
 
 * **topo.py:** propose an plug-and-play modules to get topo feature from img.(topo feature: the representation of topology such as betti curve and landscape);run main() to generate the topology feature file(xx.npy) extracted from img.    
 * **topo_classify.py:** build model (such as PCA and SVM) to classify just using topology feature file.
 
