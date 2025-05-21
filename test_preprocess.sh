#$ -l tmem=64G,h_vmem=64G
#$ -l h_rt=05:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Preprocess_data

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source CriDiff_env/bin/activate
export PATH="CriDiff_env/bin:$PATH"

cd preprocess
python3 preprocess_data.py

date