problem_formulation=dmp
algo=sa

for benchmark in adaptec1 adaptec2 adaptec3 adaptec4 bigblue1 bigblue3
do
python ../src/main.py \
    --name=ISPD2005_HPO_SA_GP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=1 \
    --eval_gp_hpwl=True \
    --n_sampling_repeat=10 \
    --max_evals=200 \
    --max_eval_time=72 \
    --n_macro=1000000 \
    --sampling=random \
    --mutation=random_resetting 
done