from sentence_transformers import CrossEncoder
from Labeling_data_ingestion.data_handler.process_data import load_data
from Labeling_data_ingestion.config import TO_FILTER_PATH
def filtered_ranking(PATH):
    df = load_data(PATH)
    
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    
    pairs = list(zip(df['theme'], df['title']))

    df['ranking'] = model.predict(pairs)
    
    df.to_csv("data/ranked/ranked_data.csv")

filtered_ranking(TO_FILTER_PATH)