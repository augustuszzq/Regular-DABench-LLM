export DUMP_ROOT=/home/ziyu/Sambatune
export DUMP_ROOT=/home/ziyu/sambanova/out_generation

sambatune --modes benchmark instrument run -- /home/ziyu/sambanova/tutorials/generative_nlp/tune.yaml > log.txt 2>&1
sambatune --modes benchmark instrument run -- /home/ziyu/sambanova/tutorials/generative_nlp/tune_1.yaml > log_1.txt 2>&1

sambatune_ui --directory /home/ziyu/Sambatune/artifact_root/sambatune_gen  --port 8568
ssh -L 8568:localhost:8568 ziyu@sambanova.alcf.anl.gov
ssh -L 8568:localhost:8568 -N sn30-r1-h1

lsof -i -P -n | grep gunicorn