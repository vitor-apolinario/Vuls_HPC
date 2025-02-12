#! /bin/tcsh
# chmod +x run_hpc.sh
# rm err/*
# rm out/*
foreach ERR (`seq 0.5 0.1 0.5`)
  foreach COR (none)
    foreach VAR (`seq 0 1 9`)
      bsub -q standard -W 5000 -n 2 -o ./out/$COR.out.%J -e ./err/$COR.err.%J /share/tjmenzie/zyu9/miniconda2/bin/python2.7 runner.py error_hpcc_continue $COR $ERR $VAR
    end
  end
end
