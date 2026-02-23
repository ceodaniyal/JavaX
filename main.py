from generate_chart import ChartGenerator

def main(data_path):

    df = ChartGenerator.read_data(data_path)
    chart_generator = ChartGenerator(df)
    chart_generator.run()
    
if __name__ == "__main__":
    data_path = "Transactions.xlsx"
    main(data_path)