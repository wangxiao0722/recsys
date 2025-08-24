N=$1
infile=$2
outfile="${infile%.csv}_sample.csv"

awk -v N="$N" '
  NR==1 {header=$0; next}
  NR-1 <= N {res[NR-1]=$0; count=NR-1}
  NR-1 > N {
    r = int(rand()*(NR-1)) + 1
    if (r <= N) res[r]=$0
  }
  END {
    print header
    for(i=1;i<=N;i++) print res[i]
  }
' "$infile" > "$outfile"

echo "✅ Sampled $N rows from $infile -> $outfile"

