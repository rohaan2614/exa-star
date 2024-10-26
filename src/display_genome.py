import dill
import src
import sys
from exastar.time_series import TimeSeries
from exastar.genome import dt_genome
from exastar.genome.component.dt_node import DTNode
from exastar.genome.visitor import graphviz_visitor_dt

# sys.path.append('src/exastar')

# from time_series.time_series import TimeSeries
# Load the object from a file using dill



def run_sim(inital_series):
    val = 1000.0  # Set requires_grad=True
    held_shares = 0.0
    tot_val = 1000

    for i in range(10):
        loaded_data.reset()
        out = loaded_data.forward(initial_series, i)
        print(out)

        for parameter_name, value in out.items():
            if value > 0:
                if parameter_name[0] == "B":
                    money_to_purchase = value * val
                    price = initial_series.series_dictionary[parameter_name[2:]][i]
                    bought = money_to_purchase / price
                    val = val - money_to_purchase
                    held_shares = held_shares + bought

                elif parameter_name[0] == "S":
                    money_to_sell = value * val
                    price = initial_series.series_dictionary[parameter_name[2:]][i]
                    sold = money_to_sell / price
                    val = val + money_to_sell
                    held_shares = held_shares - sold
            print(initial_series.series_dictionary[parameter_name[2:]][i])
            tot_val = val + held_shares * inital_series.series_dictionary[parameter_name[2:]][i]
    print(val + held_shares * inital_series.series_dictionary[parameter_name[2:]][len(inital_series)])



with open('./output/test/3.genome', 'rb') as file:
    loaded_data = dill.load(file)
# with open('./output/test/best/484.genome', 'rb') as file:
#     loaded_data = dill.load(file)
    # Use the loaded lambda function
    graph = graphviz_visitor_dt.GraphVizVisitorDT("src", "test", loaded_data)
    test = graph.visit()
    test.render(directory=".")

    # csv_filename = (["temp_run.csv"])
    # input_series_names = [
    #     "Predicted_AKAM",
    # ]
    # output_series_names = [
    #     "AKAM",
    # ]
    # initial_series = TimeSeries.create_truncated_from_csv(filenames=csv_filename, input_series=input_series_names,
    #                                                       output_series=output_series_names, length=10)

    #run_sim(initial_series)


    print(loaded_data)  # Output: 25
