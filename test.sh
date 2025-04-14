#$ -l gpu=true
#$ -l tmem=64G
#$ -l h_rt=500:0:0

#$ -pe gpu 4
#$ -R y

#$ -wd /cluster/proejct7/ProsRegNet_CellCount/Profound/
#$ -N finetune

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source CriDiff_env/bin/activate
export PATH="CriDiff_env/bin:$PATH"

python3 MY_TEST.py   