from generate_chart import ChartGenerator


def main(data_path):

    df = ChartGenerator.read_data(data_path)
    chart_generator = ChartGenerator(df)
    chart_generator.run()
    
if __name__ == "__main__":
    data_path = "Transactions.xlsx"
    main(data_path)

def main(data_path, user_query):

    df = ChartGenerator.read_data(data_path)
    chart_generator = ChartGenerator(df)
    chart_generator.run_ifelsequery(user_query)
    
if __name__ == "__main__":
    data_path = "Transactions.xlsx"
    
    # Example dynamic query (can later take from input())
    user_query = "Create charts focusing on sales trends and category distribution."
    
    main(data_path, user_query)
