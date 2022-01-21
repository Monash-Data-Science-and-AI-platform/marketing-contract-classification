README
________________________________________________________________________________________________________________________________________________________
Summary:
This README file will give an overview of the whole model, the file structures of this model, and the ways to fine-tune and pre-train the model


________________________________________________________________________________________________________________________________________________________
Conda environment is included in environment.yml
________________________________________________________________________________________________________________________________________________________
Model and transformers used:
1. nlpaueb/bert-base-uncased-contracts (pre-trained model)
2. TFBertForSequenceClassification (for fine-tuning and inference)
3. TFBertForMaskedLM(for further pre-training)
________________________________________________________________________________________________________________________________________________________
Overview:
This model consist of 3 top-level files, which are: 

1. fine_tuning.py, which is for fine-tuning the model
2. pre_training.py, which is for pre-training the model for further fine-tuning
3. inference.py, which is for making inference based on the fine-tuned model
4. plot.py, which is for plotting f1-score and weighted-average against epoch

The model also consist of several folders, which are:

1. example: contains the example dataset and guide to run the python scripts with the example dataset
2. inference: contains the python module and output for inference.py
3. fine_tuning_saved: folder for saving fine-tuned model
4. modules: folder contains modules for fine-tuning and pre-training


________________________________________________________________________________________________________________________________________________________
File structures and relationships:

1. fine_tuning.py
-fine_tuning.py is use for fine-tuning the bert-based-uncased contracts model
-for experimentations, no changes is needed in this file, all parameters and paths are changed via json files
-Parameters are changed via parameter.json in modules/fine_tuning
-Paths are changed via path.json in modules/fine_tuning
-Output of the fine-tuning(loss,validation loss, epochs and confusion matrix) are printed to output_result folder
-all custom modules are contained in modules/fine_tuning 

2. pre_training.py
-pre_training.py is use to further pre-train the nlpaueb/bert-base-uncased-contracts model
-for experimentations, no changes is needed in this file, all parameters and paths are changed via json files
-Parameters and paths are changed via parameter.json in modules/pre_training
-all custom modules are contained in modules/pre_training

3. inference.py
-inference.py is use for making inference based on a fine-tuned model of nlpaueb/bert-base-uncased-contracts
-for inferencing, all parameters are changed in inference/inference.json
-no additonal custom modules are required

4. plot.py
-plot.py is use to creating plot based on sckitlearn classification report
-all the data come from the raw data output file in .json format after running fine_tuning.py
-all parameter and path are defined in the json file in modules/plot/plot.json
________________________________________________________________________________________________________________________________________________________
Inferencing:

-for inferencing, in inference/inference.json:
-change "keys" to the keys suited to the fine-tuned model, the current fine-tuned model support keys of "DC","CA_1","CA_2","R&R","IE","Flex","None"
-change the "data_file_path" to the .xlsx or .csv file containing the sentences for inferencing
-if the inferencing model is changed:
	-change "model_path" to the .h5 file which the fine-tuned model is saved
	-change "config_path" to the config.json which is saved along with the fine-tuned model
-lastly, change the output file to a desired output text file's path

-run the inference.py file

________________________________________________________________________________________________________________________________________________________
Fine-tuning:

-First, in modules/fine_tuning/parameter.json
	-define the "project_name" and "wandb_entity"
	-change "learning_rate" if needed to test on different learning_rate
	-change "keys" to suite the current experiment(with/without "None")(code will check whether there is "None" or not, if not, it will append it automatically)
	-in "loss_function": use "binary_crossentropy" for both multi-class classification and single-class classification
	-change the graph parameters:
		-"xlabel": label for x-axis of the output graph
		-"ylabel": label for y-axis of the output graph
		-"title": title	for the output graph
		
	-change "epochs" if needed
-Then, in modules/fine_tuning/path.json
	-define the .xlsx file for training in "training_data_path"
	-and .xlsx file for validation in "validation_data_path"
	-change "output_file_path" to the desired .txt file for the result to output
	-change "graph_output_path" to the desired .png file to output the graph
	-change "raw_data_output_path" to the desired .json file to output the raw data in json format
	-change "save_model_path" and "save_config_path" to the desired path for the model and its config.json to save
	-make sure that both "save_model_path" and "save_config_path" are paths to a folder but not specific file

-Run fine-tune.py for the fine-tuning of model


________________________________________________________________________________________________________________________________________________________
Pre-training:

-in modules/pre_training/parameter.json
	-define the "project_name" and "wandb_entity"
	-change "learning_rate" if needed to test on different learning_rate
	-define the "preTraining_data_path" to the .csv or .xlsx file for pre-training
	-define the path for the pre-trained model to be saved in "save_model_path"

________________________________________________________________________________________________________________________________________________________
Additonal details for fine-tuning + pre-training:

-after pre-training nlpaueb/bert-base-uncased-contracts, in modules/fine_tuning/path.json:
	-"model_path" to the saved pre-trained model( also found in modules/pre_training/parameter.json, "save_model_path")
	-then, run the fine-tuning.py to get the result for fine-tuning+pre-training

________________________________________________________________________________________________________________________________________________________
Additonal details for single class labelling vs multi-class labelling

-change the "keys" in modules/fine_tuning/parameter.json from multi-label to single-label( example: "keys":["DC","IE","None"] to "keys":["DC"])
		
________________________________________________________________________________________________________________________________________________________
M3 MASSIVE speicfic issues:
1. TypeError: Invalid keyword argument(s) in `compile`: {'steps_per_execution'}
-This is due to the default tensorflow-gpu 2.2.0 does not support {'steps_per_execution'} in model.fit()
-Solution:
	-download tensorflow-gpu 2.4.0 or later version of tensorflow
	-in the slurm-gpu-job-script:
		-module unload cuda
		-module load cuda/11.0 
	#tensorflow-gpu 2.4.0 requires cuda 11.0, if other version of tensorflow is used, please refer to https://www.tensorflow.org/install/source#tested_build_configurations
	#to get the correct cuda version

2. Weird [Errno 2] No such file or directory or ModuleNotFoundError (even though module is installed in miniconda)
-Solution: Do not define module load tensorflow in slurm-gpu-job-script

3. unable to read .xlsx fie
-Solutions: 
	-install openpyxl in miniconda
	-In the .py file, define:  import openpyxl
	-in the line for pd.read_excel, define the argument in pd.read_excel, engine='openpyxl'

4. Resource exhaust error
-For V100 GPU, pre-training only allow up until batch size of 16(for pre-training)


5. https://huggingface.co/nlpaueb/bert-base-uncased-contracts(or similar url) takes too long to response
-Solutions: wait for a while a resubmit the job

6. The command /usr/local/sv2/dev/jupyter/jupyter_params.py {jobid} on host m3t102 timed out. Is the application server OK?
   Attempting to create an SSH tunnel tom3t101 via m3.massive.org.au unexpectedly returned an error message. Please report this via the contact us link. The error message was Unable to create the secure tunnel. Please try again.
-Solutions: 	-wait for a while after launching the jupyter lab, and then click connect
		-if error message pops-up, click connect again, repeat until you connected to the jupyter lab

-reasons: jupyter process is very slow to startup. The job begins but it takes some time before the server is actually running









