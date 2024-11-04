import dill
import src
import sys
from exastar.time_series import TimeSeries
from exastar.genome import dt_genome
from exastar.genome.component.dt_node import DTNode
from exastar.genome.visitor import graphviz_visitor_dt
import check_buy_sell

"""
Temp file to run and display the best selected genomes, used for testing.
"""

with open('./output/test/2916.genome', 'rb') as file:
    loaded_data = dill.load(file)
    graph = graphviz_visitor_dt.GraphVizVisitorDT("src", "test", loaded_data)
    test = graph.visit()
    test.render(directory=".")

    csv_filename = (["test_run.csv"])
    input_series_names = [
        "VOl_CHANGE",
        "TURNOVER",
        "Predicted_AKAM",
    ]
    output_series_names = [
        "AKAM",
    ]
    initial_series = TimeSeries.create_norm_from_csv(filenames=csv_filename, input_series=None, normalize_series=input_series_names,
                                                          output_series=output_series_names)

    print(loaded_data.test_genome(initial_series, False))
    profit, b_err, s_err = loaded_data.test_genome_daily(initial_series)
    print(float(profit))
    print(b_err)
    print(s_err)

    print()
    print("Compared to:")
    check_buy_sell.main()


    print(loaded_data)  # Output: 25
