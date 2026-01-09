# record of every command run in this project
module use /projects/community/modulefiles
module load git
conda activate seqfeaturizer

python compute_features.py \
    --fasta /projects/f_ak1833_1/jliu/to/to_Mahmudur/ASDVAR_Stephan/99.Processed/fa_variants.even.fasta \
    --out-dir features_asdvar_even \
    --features 5mer polyA_polyT_GC fimo_summary dna_shape \
    --motif-db encode=/home/mr2320/seqfeaturizer_py/motif_dbs/encode_motifs.meme \
    --motif-db hg19=/home/mr2320/seqfeaturizer_py/motif_dbs/hg19_motifs.meme \
    --fimo-p 1e-4 \
    --fimo-window 20 \
    --fimo-keep-hits \
    --dnashape rohs \
    --dnashape-timeout 1200

python compute_features.py \
    --fasta /projects/f_ak1833_1/jliu/to/to_Mahmudur/ASDVAR_Stephan/99.Processed/fa_variants.odd.fasta \
    --out-dir features_asdvar_odd \
    --features 5mer polyA_polyT_GC fimo_summary dna_shape \
    --motif-db encode=/home/mr2320/seqfeaturizer_py/motif_dbs/encode_motifs.meme \
    --motif-db hg19=/home/mr2320/seqfeaturizer_py/motif_dbs/hg19_motifs.meme \
    --fimo-p 1e-4 \
    --fimo-window 20 \
    --fimo-keep-hits \
    --dnashape rohs \
    --dnashape-timeout 1200

python compute_diff.py \
    --a features_asdvar_odd/summary.csv \
    --b features_asdvar_even/summary.csv \
    --pairs /projects/f_ak1833_1/jliu/to/to_Mahmudur/ASDVAR_Stephan/99.Processed/logfc_stephan.tsv \
    --pairs-col Names \
    --pairs-split '|' \
    --out asdvar_feature_diffs.csv