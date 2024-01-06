import pandas as pd
from progdiff.progdiff import progdiff


def main(param2val):
    """This function is run by Ludwig on remote workers."""

    # TODO can this be detected automatically?
    run_location = "ludwig_local"  # ludwig_local or ludwig_cluster

    performance = progdiff(param2val, run_location)
    print(performance)

    series_list = []
    for k, v in performance.items():
        s = pd.Series(v, index=k)
        s.name = 'took'
        series_list.append(s)
    return series_list
