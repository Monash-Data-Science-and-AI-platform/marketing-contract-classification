import sklearn.metrics as skm
import numpy as np

def result_process(scaled_pred,val_labels,txt_path,counter):
    #scaled_pred: prediction by model scaled with sigmoid layer
    #val_labels: true labels of the validations set
    #txt_path: path for txt file for the result to print
    #counter: external counter for the epoch of the training

    #scale the predictions to 0 and 1
    scaled_pred[scaled_pred>0.5]=1
    scaled_pred[scaled_pred<=0.5]=0

    #produce the confusion matrix
    confusion_matrix = skm.multilabel_confusion_matrix(val_labels, scaled_pred)#

    #generate the report
    report=skm.classification_report(val_labels,scaled_pred)#

    #get the shape of confusio matrix for printing
    dimension=confusion_matrix.shape

    with open(txt_path, "a") as f:
        f.write("Epoch %d" %counter)
        f.write("\n")
        f.write('%s\n' %report)
        f.write("\n")
        
        #iterate to extract 2D vector from 3D vector
        for j in range(dimension[0]): #iterate across the 1st dimension
            np.savetxt(f, confusion_matrix[j], fmt='%i')
            f.write("\n")