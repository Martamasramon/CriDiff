#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=120:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Pretrain_IQT

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source CriDiff_env/bin/activate
export PATH="CriDiff_env/bin:$PATH"

cd diffusion_attn
python3 train.py --use_T2W --results_folder './pretrain_down4' --down 4

date