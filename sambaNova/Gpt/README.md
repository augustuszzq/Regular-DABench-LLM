= Compile and train GPT model

=== Check your Python version

Running the tutorial requires Python 3.9 or later.

=== Check your SambaFlow installation

You must have the `sambaflow` package installed to run this example and any of
the tutorial examples.

NOTE: This tutorial requires Sambaflow 1.17 or later.

. To check if the package is installed, run this command:
* For Ubuntu Linux
+
[source,console,subs="verbatim,quotes"]
----
$ dpkg -s sambaflow
----
* For Red Hat Enterprise Linux
+
[source,console,subs="quotes"]
----
$ rpm -qi sambaflow
----
. Examine the output and verify that the SambaFlow version that you are running
matches the documentation you are using.
. If you see a message that `sambaflow` is not installed, contact your system administrator.

=== Clone the repository
----
$ cd $HOME
$ git clone https://github.com/augustuszzq/dataflow-arch-test.git
$ cd dataflow-arch-test/sambaNova/Gpt
----

* At the top level, you see the README and the following files:
** `generative_train.py` is the Python code for training the model

//release.list and requirements.txt not needed if full SambaFlow installed
* The `configuration` folder includes files to support training and inference.
** `gpt2_small_config.json` has some configuration information. The file overrides certain Hugging Face settings to improve performance on RDU.




=== Download the dataset
see detail in https://github.com/sambanova/generative_data_prep

=== Train the model

Before you run training with this model, you need:

* A dataset that's compatible with the model.
* A configuration file that amends the Hugging Face model configuration to optimize it
for RDU.
* The model Python code, `generative_train.py` in this tutorial.
You use the same `generative_train.py` file for compilation and training.
You will later use a different file to run inference.

SambaFlow has a `run` command that supports both training runs and inference runs.

* By default, `run` performs a training run.
* Specify `--inference` for an inference run, discussed further below.

Run the following command to start the training session:

First, set the variables. Don't copy these commands, the values depend on your environment:

[source,console,subs="verbatim,quotes"]
----
$ export PEF=<path/to/pef>
$ export DATADIR=<path/to/dataset>
----

Run the training script:

[source,console,subs="verbatim,quotes"]
----
$ SAMBA_SEED=256 python3 generative_train.py run \
--max_seq_length 1024 \
--batch-size 16 \
--config_name configuration/gpt2_small_config.json \
--weight_decay 0.1 \
--max_grad_norm_clip 1.0 \
--data_dir $DATADIR/generative_tuning_sst2/hdf5/ \
--checkpoint_name $OUTPUT_FOLDER/train_movie_review_checkpoint.pt \
--model_name_or_path gpt2 \
--steps 800 \
--min_eval_acc 0.87  \
--log-level error \
--pef $PEF
----


Most of the arguments are set to the same value during compilation and training.
The following arguments are required or expected during training:

* `pef`: The `pef` argument is required for training. It specifies a PEF file,
 which was the output of compilation.

Certain arguments are expected by this GPT-2 model (and defined in the model's code). Many of these arguments are used by most models.

* `data_dir`. Location of the dataset. Some pre-processing is usually necessary.
* `checkpoint_name`. If you restart training from a checkpoint, name of the checkpoint.
** Each training run saves a checkpoint file that's named after
the number of steps (e.g. `800.pt`)
** To failure proof your training run, run in batches and pass in a checkpoint
to refine your model.
* `steps`. Number of steps to run. Each step is a complete iteration through forward, backward, and optimization.
* `min_eval_acc`. Argument to use during a test run. During a test run, you pass in
the trained model and a dataset that includes labels. Your code can then check
if the outputs map to the expected outputs and ensure that this model
meets this minimum evaluation accuracy.


