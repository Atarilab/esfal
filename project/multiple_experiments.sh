sizes=( 0.70 0.65 0.60 0.55 )
# sizes=( 0.50 )

# for size in ${sizes[@]}
# do 
#     for i in {0..99} 
#     do 
#         python run_experiments.py --mode kin --size_ratio $size --id $i
#     done

#     for i in {0..99} 
#     do 
#         python run_experiments.py --mode kin --offset 1 --size_ratio $size --id $i
#     done

#     for i in {0..99} 
#     do 
#         python run_experiments.py --mode dyn --size_ratio $size --id $i
#     done

#     for i in {0..99} 
#     do 
#         python run_experiments.py --mode dyn --offset 1 --size_ratio $size --id $i
#     done
# done

# alphas=( 0.0 0.2 0.4 0.6 0.8 1.0 )
# betas=( 0.0 0.2 0.4 0.6 0.8 1.0 )
alphas=( 0.2 )
betas=( 0.4 )

for alpha in ${alphas[@]}
do 
    for beta in ${betas[@]}
    do 
        for i in {0..99} 
        do 
            python run_experiments.py --mode dyn --offset 1 --size_ratio 0.50 --alpha $alpha --beta $beta --id $i
        done
    done
done
