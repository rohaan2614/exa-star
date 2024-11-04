import pandas as pd
def main():
    """
    Quick method to test file for accuracy.
    """
    csv_dict = pd.read_csv("test_run.csv", encoding="UTF-8")

    price = csv_dict["AKAM"]
    buy_sell = csv_dict["Predicted_AKAM"]

    b_error = 0
    s_error = 0
    val = 0
    for i in range(len(buy_sell)-1):
        shift = price[i + 1] - price[i]
        if buy_sell[i] > 0:
            val += shift
            if shift < 0:
                b_error += 1
        if buy_sell[i] < 0:
            val -= shift
            if shift > 0:
                s_error += 1
    print(val)
    print(b_error)
    print(s_error)


if __name__ == "__main__":
    main()