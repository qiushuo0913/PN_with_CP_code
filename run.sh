## non-iid case
# python online_CP_4_seq.py --seed 7 --ratio_list 0.1 0.005 0.15;
# python online_CP_4_seq.py --seed 7 --ifinter_CP 0 --ratio_list 0.1 0.005 0.15;
# python naive_4_seq.py --seed 7 --ratio_list 0.1 0.005 0.15;


## iid case
python online_CP_4_seq.py --seed 7;
python online_CP_4_seq.py --seed 7 --ifinter_CP 0;
python naive_4_seq.py --seed 7;
