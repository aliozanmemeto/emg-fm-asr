# Cluster environment for the SLURM wrappers.  Adjust to your account; the wrappers `source` this file.
export EMGFM_PYTHON=${EMGFM_PYTHON:-/data/chi-gpu8/ge62niq/ser-replication-egemaps-aux/.venv-portable/bin/python}
# extra site-packages (jiwer, rapidfuzz, peft, unidecode) if they are not in the interpreter's environment
export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}/data/chi-gpu8/ge62niq/emg-cbm/.pylibs
# Hugging Face cache holding ibm-granite/granite-4.0-1b-speech (download once, then run offline).
# Set unconditionally: on this cluster the login profile presets HF_HUB_CACHE to a path on another node,
# and sbatch propagates the login environment into the job.
export HF_HOME=/data/chi-gpu8/ge62niq/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
